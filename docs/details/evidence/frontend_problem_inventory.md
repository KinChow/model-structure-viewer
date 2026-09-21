# 前端问题总排查（对标 vLLM + SGLang 两框架）—— 只摸排，本阶段不修

**方法**：SGLang 本地源码 + H20 真机减层实测；vLLM 取主干源码（github raw 逐 cache 核 dtype，无本地 vLLM）。
**分类**：`[A]` 真 bug（前端 vs 两框架都错）；`[B]` 框架分叉（vLLM≠SGLang，前端只对一个）；`[C]` 干净（对两框架都对）；`[D]` 缺项（两框架都有、前端没建模）。

## 一、显存 / KV / state 逐字节 dtype

| cache | SGLang | vLLM（源码核） | 前端 | 分类 |
|---|---|---|---|---|
| DSA index（`dsa_sparse_mla`） | fp8(uint8,1B)+fp32尺度/128 | **同：`DeepseekV32IndexerCache dtype=torch.uint8`+fp32尺度/128** | bf16(2B) | **[A] 真 bug** — 9 模型(glm5_next+deepseek_v32)，index 高估~1.94×、KV/token +10.7% |
| 线性 recurrent state · **KDA**（kimi_k3；glm5_next 有 `kda_layers`） | fp32(4B) | **同：`kda_state_dtype` auto→torch.float32** | bf16(2B) | **[A] 真 bug** — state 低估~1.93×(48%) |
| 线性 recurrent state · **GDN/Mamba2**（qwen3_5、qwen4_exp） | fp32（默认；qwen3_5 config 显式 float32） | **bf16**（`_mamba_state_dtype` auto→model dtype） | bf16 | **[B] 框架分叉** — 前端=vLLM 对、SGLang 错 |
| 线性 conv state | bf16 | bf16 | bf16 | [C] 干净 |
| MLA latent（kv_lora+qk_rope） | bf16(auto→model) | bf16(auto→model) | bf16 | [C] 干净 |
| GQA KV | bf16(auto→model) | bf16(auto→model) | bf16 | [C] 干净 |
| MiniMax 块稀疏 index | 随主 KV=bf16 | 未核（vLLM 支持存疑） | bf16 | [C] 干净(SGLang 侧) |
| dsv4/V4.1 压缩 KV+index | fp8/fp4（模型设计） | fp8_ds_mla 默认 / nvfp4 opt-in；index fp8 默认/mxfp4(Blackwell) | 建模设计 fp4（V4.1 890 B/token 精确） | 参考/flag 依赖，**非 bug** |
| W8A8C8 int8 KV（观察3） | kv_cache_scheme 仅 float/8bit→fp8 | **int8 scheme 直接拒绝**（`validate_kv_cache_scheme` 只认 num_bits8+type float→fp8） | bf16 | **降级**：两框架都不经此路走 int8 → 前端 bf16 大概率无碍 |
| MTP / 投机 draft 的 state/KV | 分配额外 draft 缓冲 | 分配 | 不建模 | [D] 缺项（静态工具取舍） |

## 二、结构 / 并行 / 通信

| 项 | vLLM | SGLang | 分类 |
|---|---|---|---|
| MoE 专家分片 | EP-XOR-TP（EP 开则 expert-TP=1，每卡整专家） | EP×moe_tp 混合（专家分组内再 TP 切 intermediate） | **[B] 分叉** — 每卡专家权重元素数不同；MSV 有 moe_tp 轴（偏 SGLang），vLLM 无此轴 |
| shared expert 融合 | 默认独立 MLP（不进 all-to-all） | DeepEP 复制成每 EP rank 一个额外 routed 专家（256+EP、topk 8→9） | **[B] 分叉** — 专家计数 + all-to-all 字节都变 |
| MTP / next-n | 一层（enorm/hnorm/eh_proj/shared_head + 一个 MLA 解码层） | 同（vLLM 源码注释 "Matches SGLang"） | [C] 干净 |
| attention-kind（linear/full 分层） | 读 `layer_types`/`layer_type` | 读 `layers_block_type`/`full_attention_interval` | [C] 干净（字段名异、逐层口径同） |
| dense TP all-reduce（Megatron 每层 2 次） | 同 | 同 | [C] 干净 |
| PD KV 传输字节 | dedup MLA latent | dedup MLA latent | [C] 干净（字节口径一致） |
| PD 跨 TP 布局重排 | 机制/约束（hetero-TP） | 机制/约束 | **[B] 分叉**（若前端建模重排/兼容契约） |

## 三、结论（本阶段只摸排，不改代码）

- **确定要修（真 bug，vs vLLM+SGLang 都错）**：
  1. **DSA index** 应按 fp8(1B)+fp32 尺度/128 计（现 bf16）—— glm5_next/deepseek_v32 共 9 模型。
  2. **KDA recurrent state** 应按 fp32 计（现 bf16）—— kimi_k3、glm5_next（KDA 家族，两框架默认都 fp32）。
- **需要"框架轴"，不是简单 bug（vLLM≠SGLang）**：
  3. **GDN/Mamba2 ssm dtype**：vLLM bf16 / SGLang fp32（+ SGLang 读 config.mamba_ssm_dtype，vLLM 不读 HF 字段、靠 server-arg）。前端 bf16 对 vLLM 对、对 SGLang 错。
  4. **MoE EP×moe_tp** 每卡专家权重布局；5. **shared-expert 融合**（专家数/all-to-all 字节）；6. **PD 跨 TP 重排**。
  → 这几项正是"做不出一个通用字节数"的根源；正解 = **元素口径当通用内核 + 模型 config 量化自动读 + 框架预设(vLLM/SGLang) + 用户覆盖**。
- **干净（对两框架都对）**：MLA latent、GQA KV、conv state、MTP 结构、attention-kind、dense all-reduce、PD KV 字节。
- **缺项**：MTP/投机 draft 的 state/KV 前端完全不建模（是否建模属设计取舍）。
- **降级**：W8A8C8 int8 KV —— vLLM/SGLang 的 kv_cache_scheme 都只认 fp8(float/8bit)、拒 int8，前端 bf16 大概率无碍（真机 serve 量化 ckpt 可最终确认）。

**证据来源**：SGLang 本地 `mem_cache/*`、`configs/mamba_utils.py`、`index_key_cache.py` + H20 真机（`sglang_glm5next.md`/`cache_dtype_audit.md`）；vLLM 主干 `model_executor/models/deepseek_v2.py`(Indexer uint8+fp32尺度)、`layers/mamba/mamba_utils.py`(`_mamba_state_dtype` auto→model dtype、`kda_state_dtype` auto→fp32)、`layers/attention/{attention,mla_attention}.py`、`fused_moe/config.py`、`deepseek_mtp.py`。vLLM 侧为 web 读源（无行号，逐段 verbatim 核对）。

## 复核（2026-09-21）：shared-expert 权重（[B] #2 结论——已被 H20 实测更正）

针对上表 [B]「shared expert 融合」项把**权重侧**核到底。**先前（基于本摸排稿）曾推测「SGLang-DeepEP 复制 shared 到每 EP rank → 前端 ÷tp 低估 ~×ep」，此推测已被 H20 真机推翻**——见 `parallelism/deepep_shared_expert_h20.md`。

- **H20 真机（关键）**：DeepEP 的 shared-expert **fusion 默认关**，且 **`moe_ep_size>1` 时在 NV 上强制关**（日志 `DeepEP: fusion off by default`；源码 `deepseek_v2.py: shared_experts_fusion_disable_reason`）。故默认 DeepEP/EP 场景 shared expert 是**独立本地 MLP、不复制成 routed 专家、不进 all-to-all**。
- **前端权重口径（纯代码核）**：shared expert 复用 `structure/operators/ops/index.js: mlpOperatorSpecs`（dense MLP），三投影权重全 `weightMatrixDecl("tp", …)` → `cost/sharding.js: declaredClassDivisor("tp")` = **÷tp**。
- **对账结论（权重侧）**：shared=本地 TP-MLP → 前端 `÷tp` **正确**，对 **vLLM / SGLang-非DeepEP / SGLang-DeepEP 默认**三者都对，**无低估**（先前的 ×ep 低估推测作废）。
- **真正的口径问题在 all-to-all 字节侧（且已修）**：前端 C3c 曾**默认**把 shared 折进 dispatch（`+n_shared`）→ 在默认 DeepEP/EP 场景**高估** a2a 字节。已改为默认不折叠、仅 `enforceSharedExpertsFusion===true`（对应 SGLang `--enforce-shared-experts-fusion`）+ sglang + `sharedExperts>0` 才 `+n_shared`；vLLM/neutral 恒不折叠。详见 `deepep_shared_expert_h20.md`（含单测 16/24 与 438/438、60/60、docs:check 全绿）。
- **[B] shared-expert 判定**：**已闭合**——权重 ÷tp 正确、a2a 字节高估已修（默认关 + opt-in）。DeepEP 前向本身也在 H20 跑通（ABI/构建障碍相对 A100 解除），完整出 token 受减层 dummy 的 MLA 维度/量化 kernel 约束、非通信问题。
