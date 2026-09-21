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

## 四、全量 Chrome E2E 复核（2026-09-22）

### 方法

- 现有 Playwright 全量用例：`frontend/e2e/viewer.spec.js` 的「每个内置模型都能展开父节点并保持可计算图」，桌面 Chrome，60 个内置模型。
- 结果：**60/60 通过**；图节点/边、展开、Cost 面板、Roofline bound 均可用。
- 追加移动端 Chrome 扫描（390×844）：**60/60 均可打开**，没有 page error、未知 bound、缺失 Roofline/Stage 时间或 `NaN/undefined/[object Object]`。
- 追加只读扫描：逐模型打开详情、展开 Cost，检查 page error、console error、请求失败、`unknown` bound、Roofline 时间、Stage HBM 时间、`NaN/undefined/[object Object]`。
- 追加扫描结果：桌面和移动端各 **60/60 均显示 Roofline 时间和 Stage HBM 时间**，均无页面异常。

### 新增问题

| 模型 | 现象 | 证据 | 分类 |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V4.1-Flash` | Provider 模型列表中不显示发布日期；该模型是 `models/catalog.json` 中唯一缺少 `release_time` 的内置模型，因此“按最新发布”排序时也只能落入无日期项 | `models/catalog.json` 条目缺少 `release_time`；`frontend/src/structure/catalog/modelOrdering.js:4-12` 对缺失时间返回空字符串 | **[P2] 前端展示/目录数据缺失** |

> 这里的“没有显示时间”确认指**模型发布日期**。V4.1-Flash 的 Cost 面板并非没有时间：全量 Chrome 扫描中可看到 Roofline 下界时间和 Stage HBM 时间。

### 其他观察

- `moonshotai/Kimi-K3` 扫描中出现 `ERR_FAILED`，仅来自测试主动 abort 的 HF/ModelScope 远程请求；模型仍由本地 built-in config 正常打开，页面无 `pageerror`、无产品错误提示，不登记为产品问题。
- 本轮未发现其他模型出现空标题、未知 Roofline、缺失 Cost 时间、页面异常或 `NaN/undefined/[object Object]`。

## 五、框架条件化修复闭环（2026-09-22）

> 上述“一～四”保留为历史摸排记录，并非本次修复后的当前状态。
> 当前公式与固定源码版本以 [framework accounting](../framework_accounting.md)
> 为准，完整命令、日志、截图见
> [本地验收记录](../../../artifacts/framework-accounting/README.md)。

### 根因与修复

| 问题 | 根因 | 本次处理 |
|---|---|---|
| V4.1-Flash 无发布日期/最新排序异常 | catalog 唯一缺失 `release_time` | 补 Hub `createdAt` 快照 `2026-09-10T02:17:58.000Z`；60 模型完整性校验；不增加页面在线查询 |
| 草稿显存只进入展示、Fit/Max Context/PD 与总量不一致 | UI 独立相加与成本层驻留口径分离 | 建立 main/draft/shared 唯一 cache pool 账本，统一总 KV/state、Fit、Max Context、PP stage 和 PD 两侧投影 |
| 共享缓存可能重复累计 | 直接累计叶子而没有存储归属 | 相同 `cache_pool_id` 去重；冲突保留较大上界并列为 evidence gap；共享映射本身不作为存储共享证据 |
| vLLM TP4 误判 invalid | 对 EP 未启用的 MoE 也注入 `moeTp=1` | dense/EP-off 保持原计划；仅明确 EP 启用时应用 vLLM effective plan，且不污染可编辑 base plan |
| shared fusion 参数失效 | 函数缺省 `false` 遮盖 plan 的显式 `true` | option > camel-case plan > snake-case plan > false；仅 SGLang opt-in 生效 |
| KV fallback 覆盖显式 dtype | 默认 bytes 被当作强制 override | 显式 KV/index dtype 优先；UI 明示“默认”与 fallback；DSA FP8+scale 保持 |
| KDA/GDN state 与 Roofline 口径不统一 | dtype 选择分散在结构、驻留与 action traffic | 从框架 profile/架构注册表解析 dtype，同一解析同时供驻留和动作计数使用 |

### 对历史推断的源码修正

- 当前审计版本的 vLLM Qwen3.5 路径会读取显式 `mamba_ssm_dtype`；
  上文“vLLM 不读 HF 字段”的旧观察不应推广到当前版本。
- vLLM 的 KDA auto 为 FP32，GDN auto 跟随 model dtype；不能把 vLLM
  所有 recurrent state 都统一为 BF16。SGLang 默认 FP32，显式配置仍优先。
- 当前上游 DSpark 有独立草稿 SWA/ring。共享 full-to-SWA 映射不代表共享
  KV 张量，所以不能为了贴近某次实测把草稿 KV 直接设零。
  runtime profile 采用有界逻辑窗口；neutral 保留 config-faithful 上界。
  显式声明的共享 pool 仍只计一次。

### 验收结论

- 代码提交：`9909da3`、`143ac21`、`a625758`、`9e757d2`，均只在本地 main。
- 最终 Node **452/452**、Python **183/183**、模型 **60/60**；
  docs:check、build、原则护栏通过。
- 桌面 Chrome **27 passed**；移动 Chrome **22 passed / 5 个原有桌面专属 skip**。
  新增定向 7 测试在两端全部执行，两端各自的 **60 模型扫描均通过**。
- 这里只验证理论公式与浏览器行为；没有重新执行 GPU benchmark。
  backend packing、分页/headroom、压缩器状态与 workspace 仍为明确 evidence gap，
  不宣称已完成逐字节真机对齐，也没有新增校准系数。
