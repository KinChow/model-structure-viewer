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

## 复核（2026-09-21）：shared-expert 权重复制（[B] #2 权重侧结论）

针对上表 [B]「shared expert 融合」项，把**权重侧**核到底（此前只确认了 all-to-all 字节侧）。

- **前端现状（纯代码核）**：shared expert 复用 `structure/operators/ops/index.js: mlpOperatorSpecs`（dense MLP），三投影权重全部 `weightMatrixDecl("tp", …)` → class `tp` → `cost/sharding.js: declaredClassDivisor("tp")` = **÷tp**（`attnMode=dp` 的 attention 叶除外，shared_expert 路径不在其中）。即**前端 shared-expert 每卡权重恒按 ÷tp**，无框架分叉门控。
- **对两框架**：
  - **vLLM**：shared expert = 独立 dense MLP、随 TP 切 → ÷tp。**前端 == vLLM**。
  - **SGLang 非 DeepEP**：shared expert 随 attention TP 切 → ÷tp。**前端 == SGLang(非 DeepEP)**。
  - **SGLang + DeepEP**：shared expert **复制成每 EP rank 一份额外 routed 专家**（每 rank 持整份 shared 权重、组内再 ÷moe_tp）。此时每卡 shared 权重 ≈ full/moe_tp，而前端给 ÷tp（tp 含 ep 因子）→ **前端在 SGLang-DeepEP 下低估 shared-expert 每卡权重约 ×(tp/moe_tp)=×ep**（如 `--tp8 --ep8` 低估 ~8×）。
- **结论**：这是 **[B] 框架分叉（SGLang-DeepEP 专属），非通用 bug**——对 vLLM 与 SGLang-非DeepEP 都正确。comm 侧的 DeepEP shared 融合（`comm.js sharedFused`，topk+n_shared 的 all-to-all 字节）已建模，**唯独权重侧的 EP 复制未建模**。
- **量级/影响**：shared expert 通常 1–2 个、intermediate 与单个 routed 专家同量级，占模型总权重很小；但在高 ep + 多 shared 时，每卡权重会被低估该分量的 ~ep 倍，影响显存 fit 的边界判断。
- **是否修 / 环境**：属真但窄的口径缺口。修法 = 框架门控（`frameworkProfile==="sglang"` 且走 DeepEP 且 ep>1 时，shared-expert 权重按 EP-rank 复制：除数取 `moe_tp` 而非 `tp`）。**运行时坐实需 Hopper + 可跑 DeepEP**（A100 CUDA-13/Ampere 编不出 DeepEP），本轮**只摸排、不改代码、不伪造运行时**，与仓库既有纪律一致。
