# 算子动作向量注册表（W5-1 切装后实况）

> 逐算子对照审查：[`operators_reference.md`](./operators_reference.md)——每算子的触发面探针实证、公式实现位置、来源三级标注与对齐勾选。
>
> **最后对齐：2026-09-10（P4/P5/P10 后）**。历史叙述（W1-W5 校准过程）保留为方法论存档，以「历史记录」标注；已完成的验收数字是棘轮基线，只许向好。

本文是 `frontend/src/structure/operators/formulas/`（`index.js` 注册表 + `counts.js` 共享实现 + `extractor.js` 提取）的实现规格：
分类、公式、共享实现与假设。实现以本文为准；修改公式先改本文。

**逐条公式来源与单位约定见 [`principles.md`](../principles.md) §3.1 / §3.7**：
`matrix` 存 MACs（aten 公式含 2×，抄时换算）、`vector` 存 flop、`sfu` 存操作次数、
`bytes` 为每次前向 compulsory traffic（权重读一遍 + 输入 + 输出；**默认无 phase 分支，
六类例外显式分相位——见下节**）。

## 相位口径

默认无 phase 分支：decode 的 memory-bound 现象由 T=1 自然涌现。该假设对逐元素类与投影类成立，
对下列六类不成立，必须显式分相位（口径段：`counts.js:16-21`）：

1. **因果可见长度**：打分对数走 `scoredPairs`（`counts.js:31-35`）——prefill 因果三角
   T(T+1)/2 + 前缀 T·(S−T)，decode 全长 S；手算校验见
   `__tests__/counts.test.js`「因果对数解析检查」（暴力求和独立 oracle，防同义重复）。
2. **MoE 专家权重流量**：`fusedMoeMlpCounts` 权重读被触达专家数 min(k·T, E)。
3. **MLA absorbed/materialized**：latent 共享时 K/V 读宽取 max(kWidth, vWidth)（cache 只存一份）。
4. **线性注意力 chunked 状态流量**：state 交互次数 = steps（prefill ⌈T/64⌉、decode 每 token）。
5. **conv/递推 state cache**：decode 相位的卷积环形历史读写。
6. **KV 读语义**：decode 的 K/V 读即读 KV cache；`kvRead`/`indexRead` 子桶单列（见 F2）。

## 全局假设（每条的 counts 注释须引用）

| 编号 | 假设 | 依据 |
|---|---|---|
| A1 | split / view 类重排**零流量**（fused projection 拆分是视图，不发生拷贝） | 2026-09-07 拍板 |
| A2 | softmax 按**融合单遍**实现，logits 读 1 遍；多遍未融合读放大不建模 | 2026-09-07 拍板 |
| A3 | rope 的 sin/cos **查表**，SFU ≈ 0 | 常规实现 |
| A4 | 复合节点的分解假设（见「复合节点」表）逐条标注 | §3.1 分解声明 |
| A5 | SFU 计数约定：sigmoid = 2（exp + rcp）、exp = 1、rsqrt = 1、div = 1；elementwise/vector 操作逐 flop 计 | 2026-09-07 统一口径 |
| A6 | 线性注意力递推核（外积 / delta matvec / query）按 per-token 计；**状态流量**按 chunked steps 计（显式近似） | Gated DeltaNet arXiv 2412.06464 |
| A7 | 融合算子（MegaMoE / fused gate+up / megakernel 等）按**语义分解**计数（matrix/vector/sfu 与融合无关）；bytes 按未融合口径（保守），融合收益记 attributes.implementation，不做流量折算 | TritonMoE arXiv 2605.23911（fused gate+up 省 35% 流量）；Megatron 2026 roadmap |

记号：`T`=tokens（phase 决定），`H`=hidden，`D`=head_dim，`I`=intermediate，
`S`=可见 key tokens，`E`=专家数，`k`=topk，`b`=每元素字节（**激活宽**）。

- **激活宽 b 主链恒 2**（bf16/fp16；`compute.js:23` `bytesPerElement: 2`）。
- **权重宽不走 b**：未量化参数的 dtype 单源在 `formulas/paramDtypes.js`
  （FP32_PARAMS：`gdn_decay` / `mhc_base` / `mhc_scale` / `mhc_fn` = 4B，其余默认 2B）；
  量化张量（FP8 块量化 / MXFP8 / GPTQ）走 `cost/quantBytes.js` 的 per-matrix 精确计算；
  checkpoint 证据在场时以 safetensors 头部逐 tensor dtype（`weight_dtypes`）为准。
  两通道由锚 1（dtype-aware，见「权重声明协议」节）逐叶对账。

## 已删除的死条目（2026-09-07 盘点确认，零引用）

`sigmoid`（仅作为属性值）、`kda_decay`（已内含于 gated_delta_attention）、
`kimi_kda`（与 gated_delta_attention 公式逐字相同，路径实际用后者）、
`kimi_kda_output_gate`（实际用 gated_rmsnorm）、`kimi_fused_qkvg_split`（实际用 split）、
`dsv4_output_projection`（wo_a/wo_b 实际用 linear，`ops/index.js:423,430`）。

## 共享 counts 实现（counts.js，F1-F9 规格族）

### F1 线性（linearCounts，`counts.js:48-68`）

matrix = T·out·in·xf；vector = bias ? T·out : 0；sfu = 0；
bytes = { weightsShared ? 0 : (out·in + (bias ? out : 0))·(weightBytesPerElement ?? b),
T·in·b, T·out·b }。
aten: `aten.mm`（torch `mm_flop` = m·n·2k FLOPs → 换算 MACs）。
logicalShape = [out, in]（weight 逻辑形状；packed 存储形状由提取层换算，qweight 无逻辑形状 → null）。

三个 W5 签名位：

- **weightBytesPerElement**：权重的字节宽可以不同于激活（fp32 的 mHC 混合矩阵等，paramDtypes
  登记；mHC ctx 传 `paramBytes("mhc_fn")`=4）；未传则跟随激活字节宽。
- **weightsShared**：本次 GEMM 复用**别处已计过**的同一份权重（如 mhc_post 复用最后一层的
  hc_ffn_fn）——算力照计、权重字节不重复计（权重字节恒等式的口径是「该相位应读一遍」）。
- **bias 计 weights**：bias 也是要从 HBM 读的权重（out 个），2026-09-09 补齐——此前只记
  权重矩阵，与本模块自己的原子分解（add 原子带 weightElements: out）及 compulsoryBytes
  口径不一致。

### F2 选择集注意力（attentionCounts，`counts.js:84-103`）

- matrix = heads·scoredPairs·(headDim + valueDim)（scores 点积 + context 加权各一段；
  **因果对数，非稠密 T·S**——prefill 系统性高估约 2 倍的口径已在 W3 修正）。
- vector = **4·scores**（1 次 1/√d 缩放乘 + softmax 三段减 max/累加/除；W5 补 scale 段——
  此前 3·scores 使融合分解恒等式实测 fused/decompose = 0.75）。
- sfu = **2·scores**（exp + rcp，A5）。
- bytes（F2 闭式）：weights = 0；
  actIn = (q + k + v + 2·scores)·b——Q 读（heads）、K/V 读（kvHeads；decode 即读 KV cache）、
  scores/probs 写+读各一遍（A2）；
  actOut = (2·scores + context + kvWrite)·b——scores/probs、O 写、**新算 K/V 写回 cache
  （kvHeads·T·(D+dv)：prefill 全量、decode 1 token）**。
  aten: `aten.bmm` ×2 + `aten._softmax`。

S 的取法与 kvHeads 由条目/提取器决定：

| 变体 | 运行时 id / case | S | kvHeads | headDim | valueDim | 备注 |
|---|---|---|---|---|---|---|
| MHA / GQA | matmul（scores/context 两叶） | seq / 上下文全长 | =heads / config.kvHeads | D | D | matrix 不随 kvHeads 变（每个 query head 做完整点积），只有 K/V 流量随 kvHeads 缩小 |
| MQA / SWA | dsv4_swa_attention | min(S, slidingWindow) | 1 | D | D | KV 读/写宽 D（swa 缓存每 token 一份 headDim 宽 latent，K/V 共享） |
| 块稀疏 | minimax_sparse_attention | min(可见, (topk+init+local)·blockSize) | config.kvHeads | D | D | 选中必须夹到可见长度（W5）；计 kvWrite |
| QSA | qsa_sparse_attention | qsaIndexerBudget | config.kvHeads | D | D | 计 kvWrite（paged cache 写回在模板内无叶承担） |
| DSA 吸收式 | dsa_sparse_mla | index_topk | 1（共享 latent） | kv_lora_rank+rope | kv_lora_rank | 读宽取 max(kWidth, vWidth)（W5 防双计）；无 kvWrite（latent 写归 kv_a_proj） |
| DSV4 C4 稀疏 | dsv4_sparse_mla | index_topk | 1 | D | D | + 原始滑窗混合读（[t-128,t]）；无 kvWrite |
| DSV4 压缩 | dsv4_compressed_attention | ⌈S/ratio⌉ | 1 | D | D | + 滑窗读；压缩态写归 compressor 叶（无 kvWrite）；matrix 走旧链镜像 |

（打分式变体覆盖矩阵 2026-09-07 补；linear attention 家族不经 F2，见 F7b。）

### F2 bytes 子桶：kvRead / indexRead（⊆ actIn）

- W6 起 extractor 的 matmul scores/context 叶与各 sparse attention case 在 bytes 里附
  **kvRead** 子桶——它是 actIn 里**从 KV cache 读的那部分**（不含 Q、top-k 索引、scores
  中间量）。KV 读恒等式只能拿这一项跟 cache 容量口径比，拿 actIn 总量比就只能留松量。
- `sparseIndexerCounts`（`modules.js:748-761`）附 **indexRead** 子桶：indexer 必须扫全长
  选 top-k（主注意力按稀疏预算读），两个乘子不同，与 kvRead 分开比。
- **两子桶 ⊆ actIn，不额外累加；且不进模型级 actions**——`compute.js:50-63` 的 actions
  只透传 matrix/vector/sfu/computeDtype + bytes 三元组（weights/actIn/actOut），子桶在
  叶级 counts 可得、聚合层不携带。

### F3 归一化（rmsnormCounts，`counts.js:110-128`）

- rmsnorm：matrix = 0；
  **vector = 4·T·H − T**（W5：均方求和是每组 hidden 个元素做 hidden−1 次加法——x²、求和、
  缩放乘、乘 weight 四段，减每 token 少的一次加法）+ weightOne ? T·H（gemma 的 (1+w) 加法）
  + gated ? T·H（门乘）；
  sfu = T（rsqrt）+ gated ? 2·T·H；
  bytes = { (weightWidth ?? H)·(affineBias ? 2 : 1)·b, (T·H + gated ? T·H : 0)·b, T·H·b }。
- **weightWidth（逐头 norm 末维）与 affineBias（LayerNorm 有 bias，权重 2×宽度）是两个
  W5 签名位**：逐头归一化（q_norm/k_norm = `RMSNorm(self.head_dim)`、GDN 输出门 =
  `RMSNormGated(self.head_v_dim)`）必须显式传最后一维，否则权重被放大 heads 倍
  （extractor `normWeightWidth`，`extractor.js:85-91`）。
- 分解声明：`mul / reduce_sum / rsqrt / mul`（+ weightOne 加法 + gated sigmoid/mul），
  无单一 aten 对应。

### F4 门控乘（gateCounts，`counts.js:133-144`）

matrix = 0；vector = T·W；sfu = 2·T·W（sigmoid = exp + rcp，A5）；
bytes = { gateProjection ? gateProjectionInput·W·b : 0, (T·W + gateProjection ? T·gateProjectionInput : 0)·b, T·W·b }。
gateProjection 是 counts 侧能力（modules gate 分解用）；主链 4 个门控叶
（attention/mla/linear_attention/shared_expert）均不带投影——投影由独立 linear 叶计。

### F5 逐元素激活（swigluCounts，`counts.js:147-154`）

swiglu：matrix = 0；vector = 2·T·I；sfu = 2·T·I（silu = x·sigmoid(x)：sigmoid 2 SFU + 1 mul，A5）；
bytes = { 0, 2·T·I·b, T·I·b }。aten: `aten.silu` + `aten.mul`。
- **vision_activation 同式复用**（`extractor.js:737-738`）：φ 由 hidden_act 决定是语义说明，
  计数与 swiglu 完全同式——**未按 gelu/exp 差异分档**（历史文档「gelu → sfu 含 exp」未落
  实现，如实登记；如需分档属后续条目）。
- **路由专家叶已拆独立条目 fused_moe_mlp**（N2-4 W-A），本式只服务 dense/vision 纯激活，
  **不再乘 expertFraction**。

### F6 旋转位置（ropeCounts，`counts.js:165-172`）

rope：matrix = 0；vector = 3·T·ropeDims（每维对 4 乘 2 加 = 3 flop/元素）；sfu = 0（A3 查表，精确零）；
bytes = { 0, 2·T·ropeDims·b, **T·ropeDims·b** }。
- **actOut = T·ropeDims·b（W5 修正）**：读 = 数据 + sin/cos 两份，写 = 数据一份——此前
  actOut 2·T·ropeDims 把 q 与 k 当两个张量，与 vector 的单总量口径不一致。
- **ropeDims = 每 token 被旋转元素总数 = (heads + kvHeads)·D·factor**（W5 修正：此前只传
  单头 head_dim，等于只算一个头；vision 塔 MHA = 2·heads）；factor =
  attributes.partial_rotary_factor ?? config.partialRotaryFactor ?? 1（`extractor.js:708-718`）。
  分解声明，无 aten 对应。

### F7a 因果短卷积（causalConvCounts，`counts.js:175-182`）

规格（注册表锚）：matrix = T·C·w；vector = T·C（silu）；sfu = 2·T·C；
bytes = { C·w·b, T·C·b, T·C·b }。aten: `aten.conv1d`（C_out·C_in·k·T FLOPs 含 2× → MACs）。

**运行时权威路径 = extractor 专用 case（`extractor.js:784-805`）**：
- matrix = T·width·kernel（width = 2·keyProjection + valueProjection）；
- **vector = 0、sfu = 0**（SiLU 段在运行时叶未计——G2 双轨差②）；
- bytes.weights = width·kernel·b（W3-⑥a：核权重每次前向读一遍）；
- decode 相位加 conv state 读写 width·(kernel−1)·b（W3-⑥b；prefill 的窗口在片上滑动，
  不额外落 HBM）。

**双源问题登记（代码侧待裁决，不改代码）**：注册表 `causalConvCounts`（含 SiLU
vector/sfu）与 extractor case（vector/sfu = 0）同名双轨——`formulas/index.js` 的
causal_conv1d.counts 是规格锚，主链不经过它。裁决前以运行时为准（G2 口径）。

### F7b 线性注意力递推状态（linearAttentionStateCounts，`counts.js:220-237`）

覆盖全部 linearAttentionMode 变体（generic = plain；qwen3_5 / qwen4_exp / kimi / kimi_k3 /
glm5_next = delta）：

| 变体 | linearAttentionMode | 状态更新 | delta |
|---|---|---|---|
| generic gated LA | generic | decay⊙S + k^Tv（plain） | false |
| Qwen3.5 / Qwen4Exp GDN | qwen3_5 / qwen4_exp | gated delta rule | true |
| Kimi / Kimi-K3（KDA） | kimi / kimi_k3 | gated delta rule | true |
| GLM-5.3-Flash | glm5_next | gated delta rule | true |

- matrix = (delta ? 3 : 2)·T·heads·dk·dv（外积 + query；delta 另加 S_{t-1}k matvec——
  2026-09-07 数学修正：matvec 属矩阵 MACs）。keyDim/valueDim 为每头维度，heads 显式
  （state = heads·dk·dv，多头下 state 流量是主导项）。
- **vector = steps·state（decay 逐元素乘）；sfu = steps·heads·(delta ? 3 : 1)**
  （每步每头 exp；delta 另加 beta 的 sigmoid 2 SFU）。
- bytes = { 0, 2·steps·state·b, steps·state·b }（递推状态读+写主导）。
- **stateSteps（chunked 语义）**：chunked 实现每块与状态交互一次；不传退回 per-token
  （steps = T，A6 下界）。
- 执行形态假设：递推核三段（外积 / delta matvec / query）按 per-token 计，chunked 实现
  总量等价（仅流量分布不同）；投影打包差异（fused qkvz vs 分离 beta/decay）是 linear
  节点（F1）；decay 参数化（safe gate / lower_bound / A_log / dt_bias）是属性级。

运行时 state_update 叶（extractor `stateUpdateCounts`，`extractor.js:330-371`；
gated_delta_attention case `:825-826` 与 linear_attention 的 /state|recurrent/ 路径 `:820-822`）：

- steps 分相位：decode = batch（每 token 一次）；prefill = ⌈T/64⌉（LINEAR_ATTENTION_CHUNK
  = 64，`extractor.js:318`，业界 chunked 默认）——这是显式近似执行形态假设，记在保留容差清单。
- matrix 走 linearStateUpdateMacs（旧链镜像，按 linearAttentionMode 分派）。
- **decay 标量参数计 weights：gdn_decay fp32 4B（`paramDtypes.js:22`）**——qwen GDN =
  2·heads（dt_bias 与 A_log 都是 num_v_heads，qwen_gdn_linear_attn.py:467-475）；
  GLM5-Next / K3 KDA = heads + heads·vd（dt_bias = projection_size、A_log = num_heads）。
  判据用节点 attributes.model_kind 字段（archs 显式登记，非子串猜测）。
- actIn/actOut = stateBytes·steps（stateBytes 含 conv 环形历史元素）。

### F8 MoE 路由与分发（topk / moeDispatch / moeCombine / add / hashRoute）

- **topk**（`counts.js:244-254`）：matrix = 0；
  **vector = T·E + (normTopkProb ? T·(k−1) : 0)**（top-k 扫描 + 归一化权重求和——W5 补
  求和段，此前分解恒等式实测 fused/decompose = 0.9734）；
  sfu = normTopkProb ? T·k : 0（除法，A5）；
  **bytes = { 0, T·E·b, T·k·4 }**——选中专家 id 是 int32（4B），与激活 b 无关。aten: `aten.topk`。
- **moe_dispatch**（`counts.js:256-261`）：全零计算；bytes = { 0, T·H·b, T·k·H·b }（gather）。
  aten: `aten.index_select`。
- **moe_combine**（`counts.js:263-270`）：**vector = 2·T·k·H**（scatter + 加权合并
  y = Σ w_e·y_e 的乘 + 累加，2026-09-07 规格修正的诚实数学）；sfu = 0；
  bytes = { 0, (T·k·H + T·k)·b, T·H·b }。
- **moe_add**（addCounts）：vector = T·H；bytes = { 0, 2·T·H·b, T·H·b }。
- **dsv4_hash_route**（hashRouteCounts，`counts.js:316-318`；`extractor.js:838-848`）：
  **weights = 0——tid2eid 是 buffer 非参数**（2026-09-09 分类裁决：Megatron-Bridge 明文
  "Buffers are not parameters"，MaxText 同；出处 Hash Layers, Roller et al. 2021）；
  **表的常驻容量（vocab·k·4B int32）由 derivedBufferBytes 单独计入显存**，不进权重字节
  恒等式。流量与 gather 原子逐位同构：actIn = actOut = T·k·b（读 topk 个专家 id、写 topk 个）。

### F9 重排与视觉（rearrangeCounts / addCounts）

- **split 家族（全零计算）**：split / mla_kv_split / qwen_qkvz_split / attention_qkv_split
  四 id 共用 `rearrangeCounts()`——bytes = **0**（A1 view 语义，显式登记为零而非漏算）。
  aten: `aten.split`（视图语义，flop_counter 无成本条目）。
- **vision_merge**：copy = true 真拷贝（A1 豁免不适用）；**G1 修正（2026-09-09）**：in/out
  元素各乘自己的 token 数（out 侧 token 数 ÷ mergeSize²）——此前单 token 宽漏乘 T_v
  （Qwen3.5-0.8B actIn 少 576×）。
- **vision_position**：addCounts——vector = T·H_v；bytes = { 0, 2·T·H_v·b, T·H_v·b }。

### 补充共享实现（F 编号外）

- **fusedMoeMlpCounts**（`counts.js:283-296`）：MoE 路由专家融合叶——matrix = T·k·3·EH·EI
  （gate/up/down 三段 GEMM 各 EH×EI）；激活段走 F5（tokens·k）；**weights =
  3·min(k·T, E)·EH·EI·b**——权重读被触达专家数：prefill 大 T（k·T ≥ E）全 E 份、decode
  T=1 只 k 份（融合形式与相位无关，相位差异由 tokens 自然涌现——「T=1 自然涌现」真正
  成立的情形）。对标 vLLM FusedMoE（打包 w13/w2）与 SGLang fused_moe 专家内核。
- **sinkhornCounts**（`counts.js:194-205`）：mHC comb/Sinkhorn 段——softmax(T·n²) +
  (iterations−1) 轮行/列归一化（每轮两方向各一次 div + add）。出处：vLLM deepseek_v4
  tilelang_kernels.py@92-142 / torch.py@62-98（kernel 取证 2026-09-09）。
- **softmaxCounts**（`counts.js:322-324`）：给独立 softmax 节点（融合注意力的 softmax 在
  F2 内含）；与 softmax 原子逐位同构（委托 atoms.js，M11.5 单处化防抄写漂移）——
  vector = 3E、sfu = 2E、actIn/actOut = E·b。
- **addCounts**（`counts.js:303-305`）：与 add 原子逐位同构（scratch 证明 120/120 组
  Object.is 相等，护栏 `__tests__/countsAtomsConsistency.test.js`）→ 直接委托。
- **scoredPairs / causalDensity**（`counts.js:31-41`）：相位助手（因果对数 / 稀疏密度），
  唯一实现在 counts.js，extractor 与 modules 共用。
- **dsv4VisibleKeys**（`counts.js`）：DSV4 主注意力可见 key 数（ratio=0 夹 sliding_window；
  ratio=4 为 ceil(T/4)+window 再夹 dsaIndexTopk；ratio=128 为 ceil(T/128)）。叶 counts
  与 T4 期望侧共用，禁止再抄一份。

## 复合节点（分解声明）

以下条目的 counts = sumCounts 组合已知实现；它们都是叶子 operatorSpec，子节点无独立
算子，无双计。**分解以 `formulas/index.js` 的 counts 组合为唯一实况**——本表逐条给出
该组合；`modules.js` 的 fused/decompose 是同一组合的模块层视图（护栏：融合 ≡ Σ 分解、
bytes 差额 == 驻留中间量，`__tests__/identities.test.js` 容差 0）。

| 条目 | 分解（index.js counts 组合） | ctx 要点（extractor ctxBuilders） |
|---|---|---|
| mla_query_compress | **F1(qa)**（`index.js:310`） | **norm 与 q_b 都不在本叶内**（q_a_norm / q_b_proj 是独立叶，算进来就是双计——q_b 由 2026-09-07 参数审计、norm 由 2026-09-09 权重字节逐层归因抓出）；ctx 的 norm 键为残留，counts 不消费 |
| mla_kv_compress | F1(proj) + F9(split view)（`index.js:321`） | out 维以节点 output_shape 为权威（MLA latent = kvLoraRank+qkRopeHeadDim；DSV4 压缩 = 2·headDim·k），config 组合仅作无形状回退；latent cache 写 = 本叶 actOut |
| attention_residual | softmax(aggregate) + add(mix)（`index.js:358`） | 两个 norm 与两个 [H→1] 打分投影是独立叶（self_attention_res_norm / mlp_res_norm / *_res_proj，vLLM kimi_k3/amd/linear.py:562-580），算进来就是双计（K3 每层 +14,336）；ctx 的 norms/scoreProj 键残留不消费 |
| hyper_connection | F3(grouped, weightOne) + F1(mixDown) + F4(silu) + F1(mixUp) + F4(gate) + F1(inject) + add(combine) **七段**（`index.js:376-384`） | 形状全来自 vLLM GatedResidual（hyperconnection.py:140-193）：hyper_hidden = hc_count·hidden；hc_use_combine === false 的相位（最终 mixer）无 block_inject_weight → inject tokens=0 + weightsShared |
| mhc_pre | gate(mix) + F1×3(fn/base/scale) + F3(norm) + sinkhorn + add(merge)，**computeDtype: "tf32"**（`index.js:265`） | N2-1：pre-GEMM 在 Hopper/Blackwell + DeepGEMM 上跑 TF32（tilelang_kernels.py:686-711），roofline 按 tf32 费率拆算；fn/base/scale 是 fp32 参数（paramDtypes：mhc_fn/mhc_base/mhc_scale = 4B），base/scale 传 tokens=0（纯参数无激活流量）；attn_norm 权重融进内核（无独立 input_layernorm 叶）记 norm 段；混合矩阵形状 mix_hc = (2+hc_mult)·hc_mult、hc_dim = hc_mult·hidden（model.py:709-752，此前写 [H, hc_mult] 小 24 倍已修） |
| mhc_fused_post_pre | gate(post) + add(inject) + gate(pre) + F1×3 + F3 + sinkhorn，**computeDtype: "tf32"**（`index.js:276`） | 同上；ffn_norm 权重融进 fused 内核（model.py:705）；A7 融合收益记 implementation 不折算 |
| mhc_post | F1(combine, **weightsShared**) + add(inject)（`index.js:286`） | 最终 hc_post 复用**最后一层** hc_ffn_* 参数再算一遍（model.py:1074-1097），没有自己的参数——算力照计、权重字节不重复计 |
| mhc_contract | add（`index.js:295`） | GLM-5.3-Flash 末层 n 流平均收缩 |
| ple | F8(hash embed) + F1(kv) + F3(norm) + F7a(conv) + add（`index.js:394`） | Qwen4Exp PLE（Qwen modeling 未入库，离线取证）；kv = [2·pleEmbedDim, H]；conv = (pleEmbedDim, pleNgramSize) |
| qsa_indexer / dsa_indexer / dsa_kpool_indexer / dsv4_indexer / minimax_sparse_indexer | **sparseIndexerCounts 一份参数化实现**（`modules.js:748-761`，分解见 `:667-696`）——四组参数：QSA（key 池化 compress_ratio>1 时、等权求和）/ DSA（逐 token、逐头 weights_proj）/ DSA-kpool（key 池化 + pool 粒度 topk + tail）/ MSA（score 池化块 max）；dsv4_indexer 与 DSA 同参 | 五个 operator_id 不共用条目（算法出处不同），共用实现（W2 改判）；共同点：无 value 通路、无 softmax（ReLU + 逐头求和）、index k 单头、scores fp32（scoreBytes=4）；indexRead 子桶 ⊆ actIn |

## 逐条清单（49 条）

> 在用键集 = `formulas/__tests__/counts.test.js:259`（「注册表完整性」测试与 FORMULAS
> 键集**双向锁定**，新增/删除条目必须同步）。相对 W5 快照的 42 条差集：
> **+fused_moe_mlp**（N2-4 W-A 从 swiglu 拆出路由专家语义）、**+residual_add**（W4）、
> **+dsa_indexer / dsa_kpool_indexer / dsv4_indexer**（W2 从 qsa_indexer 按算法出处拆分）、
> **+dsa_sparse_mla / dsv4_sparse_mla**（W2 从 qsa_attention 拆分）、
> **qsa_attention → qsa_sparse_attention**（更名）。
>
> **机器段化待办（登记）**：本清单为手写对齐；后续可仿 P9 的
> `scripts/gen-model-reference.mjs` 手法把清单节并入 `docs:check` 机器段（本波只登记，
> 不改代码）。

| # | 条目 | 分类 | 实现 | 备注 |
|---|---|---|---|---|
| 1 | linear | 计算+访存 | F1（`counts.js:48`；`extractor.js:426-453`） | aten: mm；routed ×xf；logical_weight_shape 优先 + derived 回退；bias 已接线 |
| 2 | matmul | 计算+访存 | F2 分解两叶（`extractor.js:454-519`） | aten: bmm ×2；scores 叶 + context 叶；kvRead 子桶 |
| 3 | softmax | 仅访存 | F2 内核 / softmax 原子（`counts.js:322`） | elements = heads·scoredPairs（`extractor.js:699-707`）；A2 |
| 4 | split | 仅搬运（零） | F9（`counts.js:330`） | A1 view |
| 5 | causal_conv1d | 计算+访存 | F7a 规格 `counts.js:175`；运行时专用 case（`extractor.js:784-805`） | 双源待裁决：运行时 vector:0/sfu:0，含核权重读 + decode conv state |
| 6 | rope | 仅访存 | F6（`counts.js:165`；`extractor.js:708-718`） | A3；ropeDims = (heads+kvHeads)·D·factor |
| 7 | vision_position | 仅访存 | addCounts（`counts.js:303`；`extractor.js:739-740`） | |
| 8 | vision_merge | 仅搬运（真拷贝） | F9 copy（`extractor.js:741-755`） | G1 修正：in/out 元素各乘 token 数 |
| 9 | vision_activation | 仅访存 | F5 同式（`extractor.js:737-738`） | 计数与 swiglu 同式，未按 φ 分档 |
| 10 | rmsnorm | 仅访存 | F3（`counts.js:110`；`extractor.js:719-726`） | weightWidth = 逐头 norm 末维；affine_bias → 权重 2× |
| 11 | gemma_rmsnorm | 仅访存 | F3 weightOne | (1+w) |
| 12 | swiglu | 仅访存 | F5（`counts.js:147`；`extractor.js:762-766`） | 纯激活（SiluAndMul）；不乘 expertFraction |
| 13 | fused_moe_mlp | 计算+访存 | fusedMoeMlpCounts（`counts.js:283`；`extractor.js:767-783`） | gate/up/down + SwiGLU；权重读 min(k·T, E) 份；EH = latent_size→routedExpertHiddenSize→hiddenSize |
| 14 | topk | 仅访存 | F8（`counts.js:244`；`extractor.js:827-828`） | vector = TE + norm?T(k−1)；actOut = T·k·4（int32） |
| 15 | moe_dispatch | 仅搬运 | F8 gather（`counts.js:256`） | actIn = TH·b、actOut = TkH·b |
| 16 | moe_combine | 仅搬运+加权 | F8 scatter+加权（`counts.js:263`） | vector = 2·TkH |
| 17 | moe_add | 仅访存 | addCounts（`extractor.js:836-837`） | routed/shared 合并 |
| 18 | residual_add | 仅访存 | addCounts（`index.js:188-200`；`extractor.js:833-835`） | W4 补；每层两处 h = x + sublayer(x)；hidden 取 output_shape 宽 |
| 19 | linear_attention | 计算+访存 | F7b 规格；运行时按路径分派 conv/state 子叶（`extractor.js:806-824`） | 0/59 触发（通用槽位保留）；short_conv 子叶 weights=0 与 causal_conv1d 叶分工 |
| 20 | linear_attention_gate | 仅访存 | F4 | 0/59 触发（KDA 输出门 z·y 路槽位） |
| 21 | gated_delta_attention | 计算+访存 | F7b delta + state_update 叶（`extractor.js:825-826` → `:330-371`） | chunked steps；gdn_decay fp32 4B 标量权重；matrix 走 linearStateUpdateMacs 镜像 |
| 22 | gated_rmsnorm | 仅访存 | F3 gated（`extractor.js:727-731`） | 逐头门控；sfu = T + 2TH |
| 23 | mhc_pre | 分解 | gate+F1×3+F3+sinkhorn+add，tf32（`index.js:254-266`） | 见复合节点表 |
| 24 | mhc_fused_post_pre | 分解 | gate×2+add+F1×3+F3+sinkhorn，tf32（`index.js:267-277`） | post+pre 层间融合 |
| 25 | mhc_post | 分解 | F1(weightsShared)+add（`index.js:278-287`） | 复用最后一层 hc_ffn 参数 |
| 26 | mhc_contract | 分解 | add（`index.js:288-296`） | n 流平均收缩 |
| 27 | mla_query_compress | 分解 | F1(qa)（`index.js:297-311`） | norm / q_b 独立叶防双计 |
| 28 | mla_kv_compress | 分解 | F1 + F9(view)（`index.js:312-322`） | latent cache 写 = 本叶 actOut |
| 29 | mla_kv_split | 仅搬运（零） | F9 | A1 |
| 30 | mla_output_gate | 仅访存 | F4 | 仅 Kimi-K3 触发 |
| 31 | attention_residual | 分解 | softmax+add（`index.js:343-359`） | norms / 打分投影独立叶 |
| 32 | hyper_connection | 分解 | 七段（`index.js:360-385`） | 见复合节点表 |
| 33 | ple | 分解 | F8(hash)+F1+F3+F7a+add（`index.js:386-395`） | Qwen4Exp PLE |
| 34 | shared_expert_gate | 仅访存 | F4 | 16 模型 |
| 35 | qsa_indexer | 分解 | sparseIndexerCounts（`modules.js:748`；`extractor.js:882-893`） | QSA 参数组（key 池化、等权） |
| 36 | dsa_indexer | 分解 | sparseIndexerCounts（`extractor.js:894-905`） | DSA 参数组（逐 token、逐头权重） |
| 37 | dsa_kpool_indexer | 分解 | sparseIndexerCounts（`extractor.js:906-917`） | kpool 参数组（pool 粒度 topk + tail） |
| 38 | dsv4_indexer | 分解 | sparseIndexerCounts（`extractor.js:918-929`） | 同 DSA 参数组（C4 压缩 latent 打分） |
| 39 | qsa_sparse_attention | 计算+访存 | F2 变体（`extractor.js` qsa/dsa/dsv4 分族） | S = qsaIndexerBudget；计 kvWrite；top-k 索引读 T·selected |
| 40 | dsa_sparse_mla | 计算+访存 | 同上 | latent 共享（读宽 max(k,v)）；无 kvWrite |
| 41 | dsv4_sparse_mla | 计算+访存 | 同上 | MQA 行 + 原始滑窗混合读；无 kvWrite |
| 42 | qwen_qkvz_split | 仅搬运（零） | F9 | A1 |
| 43 | attention_qkv_split | 仅搬运（零） | F9 | A1 |
| 44 | attention_output_gate | 仅访存 | F4 | 29 模型 |
| 45 | minimax_sparse_indexer | 分解 | sparseIndexerCounts（`extractor.js:930-942`） | MSA 参数组（score 池化块 max） |
| 46 | minimax_sparse_attention | 计算+访存 | F2 块稀疏（`extractor.js:589-626`） | 选中夹到可见长度；计 kvWrite |
| 47 | dsv4_hash_route | 仅搬运 | gather（`counts.js:316`；`extractor.js:838-848`） | tid2eid = buffer 非参数：weights = 0；容量走 derivedBufferBytes |
| 48 | dsv4_swa_attention | 计算+访存 | F2 滑窗 MQA（`extractor.js:627-663`） | KV 读/写宽 = D（K/V 共享 latent） |
| 49 | dsv4_compressed_attention | 计算+访存 | F2 压缩读（`extractor.js:664-698`） | matrix = 旧链镜像 legacyDeepseekV4AttentionMacs；压缩态写归 compressor 叶 |

## 结构级缺口 —— 已清账（M11/W4，2026-09-09）

W1 时点在 principles §10 登记两条结构级缺口，均已修复：

- ~~**残差加法无算子节点**~~ ✅ 已清（W4）：`residual_add` 算子
  （`formulas/index.js:188-200`，counts = addCounts；每层 attention 与 FFN 子块后各一次
  h = x + sublayer(x)，`extractor.js:833-835`，hidden 取 output_shape 宽）。
- ~~**embedding gather 无算子节点**~~ ✅ 已清（M11，流量侧）：embedding 是结构节点
  （无 operatorId），extractor `type === "embedding"` 分支（`:407-415`）与 linear case 的
  embed 排除分支（`:436-445`）计 gather 流量 actIn/actOut = T·H·b、matrix 恒 0；权重容量
  不进 counts.bytes.weights，走 P4 声明 + 登记例外（**声明 = 驻留、流量按行计**，
  `embedding.js:13-16`，见下「权重声明协议」）。

量级备注（方法论存档，2026-09-07）：残差加法每层 2 次 × 3TH·b，60 层 H=8192 bf16 ≈
6 MB/token，相对权重流量（GB 级）可忽略——当时「暂缓」是量化后的决定，后由 W4 补齐
为精确计费。

## 权重声明协议（P4，2026-09-10 落地）

**weightMatrices 权重声明**（schema v2，`ops/index.js:65-100` `weightMatrixDecl`）：

```text
{ class, shape, count, matrices, split, quantizable, param_dtype }
```

- **class** ∈ tp | ep | vocab | replicated——分片亲和，与 parallel.js 的分片轴对应。
- **shape**：张量形状数组（safetensors/state_dict 的成熟表示——向量 [heads]、卷积核
  [width, kernel]、矩阵 [out, in]）；out = shape[0]、in = 其余维乘积为派生字段，供现有
  消费者零改动使用。
- **count × matrices × out × in = 该组全部元素**，与叶 counts.bytes.weights 逐位可对账
  （锚 1，`modelIdentities.test.js`；声明写错立即红）。
- **split**：切分维度，命名对应 vLLM 并行类——`MergedColumnParallelLinear`（gate_up/qkv，
  沿 output 切）→ "output"，`RowParallelLinear`（down/o_proj，沿 input 切）→ "input"，
  replicated → null（vLLM `linear.py` create_weights 的 `ModelWeightParameter(input_dim=1,
  output_dim=0)`：切分轴是权重参数的一级属性）。当前消费者只算 ÷tp 总量比例，split 是为
  维度级建模（w1/w3 列切、w2 行切）预留的一级字段，不改变现有行为。
- **quantizable**：量化方案只作用于 Linear 权重矩阵（HF quantization_config targets:
  ["Linear"]，vLLM/SGLang 同）；norm scale/bias/衰减参数显式 false。
- **param_dtype**：引用 `formulas/paramDtypes.js` 的 FP32_PARAMS 键（不携带字节数——dtype
  知识仍单源在登记表），供锚 1 的 dtype-aware 判据使用。

接线与执法：

- 归一化族与线性族的声明由 `operatorSpec` 工厂按形状**自动产出**（`ops/index.js:30-34`、
  归一化族 `:110-120`），显式传 weightMatrices 的调用点覆盖自动值。
- **锚 1 dtype-aware**（MAINTENANCE 棘轮）：声明元素 × (param_dtype ? paramDtypes 字节宽
  : 2B) == 叶 counts.bytes.weights，全目录逐叶容差 **0**（18401 声明叶、违例 0）。
- **登记例外 1 处：embedding**（`embedding.js`）——声明 = **驻留**（本叶
  `vocab_size×hidden_size`：主词表 = 模型 vocab×hidden，PLE ngram 表 =
  padded_vocab×head_dim），gather 流量按行计（actIn/actOut = T·H·b、
  counts.bytes.weights = 0），声明不等于该相位读量。主词表 vocab 亲和 =
  ParallelLMEmbedding；ngram 表 replicated（HF `_no_placement_params`）。
  quantizable = false（vLLM ParallelEmbedding 无 quant_method）。
- **P5 回退删除**：WEIGHT_PROJECTION_RULES 与 QUANTIZABLE_OPS 已删——**无声明带权叶 =
  unknown**（不再静默回退）；P2 声明覆盖 18401/18401 带权叶全声明。

## 通信与并行项（P10，2026-09-10 落地；协议 Q4/Q6/Q7）

- **AllToAll dp>1**（`comm.js:59-66`）：无 EP 时专家被 DP 切（DP-shards-experts，协议 Q4），
  DP 副本间仍需 all-to-all；EP 启用走 ep 分支；attnMode=tp 时 token 已按 DP 复制、专家域
  含 dp——同样触发。口径标注近似（Q3：无 moe_dp 轴，用 dp 近似）。
- **PD transferSeconds**（`comm.js:98-102`）：传输时间 = per-decode-rank KV+state 字节 /
  min(两侧链路带宽)（Q7②，闭式，不建模 overlap/协议开销）。
- **kvKeepRatio**（`parallelPlan.js:32-35`）：decode 侧实际驻留 KV 比例（streaming/滑窗/
  逐出），1 = 全保留（缺省），(0, 1] 区间校验；fit 估算按比例折减、输出标注估算口径。
- **overlapUpperBound**（`roofline.js:125`）：comm 与 compute 取 max（roofline 五路聚合本
  就是 max 语义），输出带 `overlapUpperBound: true` 标记——静态上限、非调度仿真（Q7③）。

---

## 提取器规格（W5-1 切装后实况）

`frontend/src/structure/operators/formulas/extractor.js` 是**唯一** node → counts 提取路径：
`compute.js` 主链（W5-1 切装，旧 nodeMacs 分派链已删除，`compute.js:1-3`）经
`countsForNode` 查本注册表。芯片参数不得进入该文件（§3.4）。

```js
countsForNode(node, env = { config, options, path, bytesPerElement }) → counts | null
```

- env 四件套：config（normalize 后）、options（batch/sequence/phase）、path（结构化 id）、
  bytesPerElement（主链恒 2，激活宽——`compute.js:23`）。
- 返回动作向量 { matrix, vector, sfu, bytes{weights, actIn, actOut}（部分 case 另含
  kvRead / indexRead 子桶）, computeDtype? }——**无 source 字段**；matrix 无法确定时返回
  null（调用方计入 unknownComputePaths）。
- operators_reference.md 的管线节已按此签名书写（`operators_reference.md:133`）。

### 提取原则

1. **查表优先**：节点已有 `weight_shapes`（checkpoint 真值）、`input_shape/output_shape`
   （dims.js 数值形状）、`attributes`——提取器只补三样节点上没有的东西：
   ① phase 相关的 T/S；② 变体选择参数（budget/blocks/window/compress）；③ expertFraction。
2. **-1 位替换**：dims.js 约定 -1 = 自由维。T = batch·(decode ? 1 : sequence)·(vision ?
   visionTokens : 1)（`tokensFor`，`extractor.js:44-46`）；S 默认 = sequence（prefill）/
   上下文全长（decode）。
3. **结构化判据**：scores 与 context 两类 matmul 用「output_shape 与 tensorDims(config)
   的模式匹配」区分；**禁止显示名**（§3.2）；text/vision 两套 patterns（`extractor.js:116-123`）。
4. **层向**：extractor 不 import cost 层；bytesPerElement 由调用方传入（主链恒 2）；
   权重宽走 weight_dtypes / paramDtypes 通道，两通道由锚 1 对账。
5. **repeat/multiplier**：counts 返回单实例；倍乘由 walker 沿用 multiplier（`cost/traverse.js`）。
6. **expertFraction**：`expertFractionFor`（`extractor.js:54-61`）——routed expert 且该层
   非 dense 时按 k/E 缩放，只作用于 linear（fused_moe_mlp 用 topk 直接表达）；路径正则
   LAYER_INDEX_RE / ROUTED_EXPERT_RE 全仓统一在 `extractor.js:40-41`。

**旧链词汇（历史记录）**：上文与代码注释中的「与旧链对齐 / 旧链镜像」是 W1-W4 差分期的
历史词汇；W5-1 后旧链已删除，extractor 是唯一提取路径。「旧链镜像」函数段
（`extractor.js:125-372`）现状混杂：

- type === "attention" 容器按 §2.4 不计费（nn.Module 不是 kernel 边界；打分核在叶上走 FORMULAS）。
- linearStateUpdateMacs（state_update 叶 matrix）与 dsv4CompressedAttentionCounts
  （dsv4_compressed_attention matrix）仍是**权威路径**。

同段内「死/活」混杂、注释「W5 切换后随旧链一并删除」已失效——**双源问题登记为代码侧
待裁决（不改代码）**。

### 分派表（extractor switch 全量）

| 分派 | operatorId / 判据 | ctx 来源与要点 |
|---|---|---|
| 模块容器 | type === "attention" | §2.4：nn.Module 容器不计费，返回 null；打分核在叶上 |
| 结构节点 | type === "embedding" | gather：actIn/actOut = T·H·b、weights = 0（`:407-415`） |
| 线性 | linear；或无 operatorId 但 weight_shapes 有 ≥2 维形状（effectiveOperatorId 改写，`:419-423`） | `linearLogicalShape`：attributes.logical_weight_shape 优先 → weight_shapes 首个 ≥2 维 → `derivedLinearShape`（input/output 正维积）回退；packed 无逻辑形状 → null（诚实未知）；**bias = attributes.bias === true 已接线**；文本 token embed 结构化路径排除（真查表走 gather；视觉 patch embed 是 Conv3d 一次 GEMM，不排除）；routed ×xf |
| scores/context matmul | matmul | `attentionShapePatterns`：**context 先判**（含具体 heads/value 维，更具体；scores 的全 -1 通配会吞掉一切 4D 输出）+ text/vision 两套 patterns（`:454-519`）；scoredPairs 分相位；MLA latent 共享 → kvHeads=1、K 读宽 kv_lora+rope、V 读宽 kv_lora（context 叶同 latent 不再读，W5 防双计）；kvRead 子桶 |
| 融合稀疏注意力 | qsa_sparse_attention / dsa_sparse_mla / dsv4_sparse_mla（**三 id 共用 case**，`:522-588`） | S = qsaIndexerBudget（qsa）/ dsaIndexTopk（dsa/dsv4）；latentRead = 非 qsa 且 kvLoraRank>0 → kvHeads=1、读宽 max(k,v)；kvWrite 仅 qsa 计（latent 写归 kv_a_proj、C4 压缩态写归 compressor，防双计）；dsv4 滑窗混合读补记；top-k 索引读 tokens·selected |
| 块稀疏注意力 | minimax_sparse_attention | 选中 token = min(可见, (topk+init+local)·blockSize)（`:589-626`）；计 kvWrite（cache 写回在融合算子内，dense 侧由 k/v_proj actOut 计） |
| 滑窗 / 压缩 MQA | dsv4_swa_attention / dsv4_compressed_attention | ratio 由 compressRatios[layerIndex]（层索引取自节点路径）；compressed matrix 走旧链镜像；压缩态写归 compressor 叶（无 kvWrite）；c128a 混合读含原始滑窗 |
| softmax | softmax | elements = heads·scoredPairs（`:699-707`） |
| rope | rope | ropeDims = (heads+kvHeads)·D·factor（vision = 2·heads）；factor = attributes.partial_rotary_factor ?? config ?? 1（`:708-718`） |
| 归一化 | rmsnorm / gemma_rmsnorm / gated_rmsnorm | hidden = staticWidth(input_shape)；**weightWidth = normWeightWidth（逐头 norm 末维）**；affineBias = attributes.affine_bias（`:719-731`） |
| 门控 | attention_output_gate / mla_output_gate / linear_attention_gate / shared_expert_gate | width = staticWidth(output_shape)（`:732-736`） |
| 激活 | swiglu / vision_activation | intermediate = staticWidth(output_shape)；**swiglu 不再乘 expertFraction**（路由专家叶已拆 fused_moe_mlp，`:762-766`） |
| 视觉 | vision_position（add）/ vision_merge（copy，in/out 各乘 token 数） | `:737-755` |
| split 家族 | split / mla_kv_split / qwen_qkvz_split / attention_qkv_split | 全零（A1 view，显式登记，`:756-761`） |
| 卷积 | causal_conv1d | extractor 专用 case（vector:0/sfu:0、核权重读、decode conv state）——与注册表 causalConvCounts 双源（待裁决，见 F7a） |
| 递推 | linear_attention（路径分派 short_conv\|conv / state\|recurrent，`:806-824`）；gated_delta_attention（`:825-826`） | state_update 叶 = stateUpdateCounts（chunked steps、gdn_decay fp32 权重）；matrix 走 linearStateUpdateMacs 镜像 |
| MoE / 残差 | topk / moe_dispatch / moe_combine / moe_add / **residual_add** / dsv4_hash_route / fused_moe_mlp | E/k/H 来自 config 或节点 shape（payload 宽走 staticWidth(input_shape)）；hash_route weights=0（buffer）；fused_moe_mlp EH = latent_size → routedExpertHiddenSize → hiddenSize（`:767-783`、`:827-848`） |
| 复合 | mhc_* / hyper_connection / ple / attention_residual / mla_query_compress / mla_kv_compress / 五 indexer | default 分支查 FORMULAS 注册表 counts + extractor ctxBuilders（`:849-1027`）；无 operatorId 的结构节点 = 零向量；有 operatorId 但注册表未实现 = null（unknownComputePaths） |

---

## 提取器任务分解 T1-T5

> **历史记录，全部完成**（T1-T3 于 W1、T4 于 W1-M8、T5 于 W5-1 收口；下文数字为各时点
> 快照，保留作校准方法论存档。当前终态见「M8-V2 收官快照」）。

- **T1** extractor 骨架 + 线性族 → 旧链差分（linear 子集先行）✅
- **T2** attention 族（scores/context 模式匹配 + 融合变体 + kvHeads 映射表）✅
- **T3** elementwise 与其余 + 复合节点（逐条核对 attributes）✅
- **T4** 整模型恒等式 ✅（2026-09-08 二次收敛）：
  - 目录构成：37 vision（恒等式 v2 再覆盖）+ 22 MoE；**无纯 dense**，dense 字段组合
    由 T4b 合成变体覆盖（GQA untied / tied embeddings / headDim 推导 / MoE+shared+tied，
    四个变体全部精确闭合）；
  - 校准过程中修复（详见 refactor_plan.md W1 问题实录）：routed swiglu 按 k 而非 k/E、
    qb 重复计费、derived MLA 调度回退、sharedExpertIntermediateSize 通用回退（含
    kimi_k3 fused 语义）、期望侧 score 项 2× 双计、normsTerm 层数、generic-decoder 缺尾。
- **T5** 旧链差分全量 + 漏算清单产出 ✅（差分测试已常驻；旧链已于 W5-1 删除，
  漏算清单作为切换价值证明随删随出）。

历史记录（T4 时点口径，已被收官快照取代）：21 个目录 MoE 行 |ratio-1| ≤ 1.7%、
测试断言为全模型统一 2% 容差——**现终态为 0.005 + REGISTERED 空**（见下）。

与旧链的差分预期（历史记录）：matrix 逐节点全等；旧链 = 0 而新链 > 0 的节点
（mhc/hyper/ple/indexer 复合、qsa/minimax/dsv4 融合节点）输出旧链漏算清单，作为 W5
切换的价值证明——已完成使命。

## 恒等式残差 —— 已清账（2026-09-09 收官；本节为方法论存档）

> 终态：四条恒等式**容差 0**（权重字节 / KV 读分桶 / 激活流形状连续性 / 融合分解全部
> error 模式）、matrix 恒等式容差 0.005（浮点求和/取整口径）、**REGISTERED 空**
> （MAINTENANCE 棘轮表；`extractor.identity.test.js:41-42`）。

R1 逐项对账审计（2026-09-08）的归因路径，全部清零，方法保留：

- ~~counts 侧 kv_b 宽度~~ ✅ 已修（2026-09-08，M8-V1）：MLA 模板 kv_b 输出改为
  `[-1,-1,kvHeads, qkNope+vHeadDim]`；R1 恒等式 0.9914→0.9983、Kimi 0.9950→0.9990。
- ~~测试期望侧 score 项~~ ✅ 已修（2026-09-08）：期望侧每层写成 2·2·heads·T²·D（双计），
  真值 = scores + context 各 heads·T·S·D。修复后目录模型整体收紧 ~0.5-1.7%。
- ~~GLM-5/5.1/5.2/5.3 +0.3%~+0.5% 正向残差~~ ✅ 已清（2026-09-09 六波收官）：根因 =
  derivedWeights 的 DSA 分支 model_type 白名单漏 glm5_next + ops 模板 KDA 宽度错
  （低秩 decay、out_proj 输入宽），修正后全类 1.0000；权重字节 32/32 逐字节。
- ~~Kimi-K3 1.0437 / GLM-5.3-Flash 1.0909~~ ✅ 已清：latent MoE down/up 投影每层一份
  全 token 激活（期望侧误乘 k/E）、块稀疏 selected 夹到可见长度等，见收官快照。
- 设计备忘（方法论存档，保留）：若残差再扩大，可考虑「期望侧改为同一 IR 的叶子权重
  清单 × 1 MAC」——代价是恒等式从独立 oracle 退化为对账自检。成熟方案调研结论
  （2026-09-08）：PyTorch `test_flop_counter.py` / fvcore 同样以 per-op golden + 独立
  端到端真值为准；定位困难靠层级分解报告解决；Megatron-LM 混合 MoE FLOPs 计数曾静默
  出错 57.5%（arXiv 2605.20799），业界确认「手工公式随模型演化静默失效」是常态，
  两侧对账 + 外部真值是主流做法。

## M8-V2 恒等式登记 —— 收官快照（2026-09-09/10）

> 校准方法（域拆分账本、四样东西、已排除假设纪律）见
> [`identity_calibration.md`](./identity_calibration.md)（历史记录）。

- **终态（`extractor.identity.test.js`）**：TOLERANCE = **0.005**、REGISTERED = {}
  （**空**）。59 模型 matrix 恒等式 |ratio−1| ≤ 0.005（残留来自 tied embedding 与 norm
  权重项的取整口径，量级稳定；vision 域由 v2 双 token 域拆分覆盖）。DSV4 打分项按
  `compress_ratio` 分层，与叶 counts 共用 `dsv4VisibleKeys`。
- 归零路径（方法论存档，每条有实测证据，不是放宽容差）：
  - GLM-5.3-Flash 1.0904 → 0.999x：ops 模板 glm5_next KDA 两处宽度错 + derivedWeights
    的 DSA 分支白名单漏 glm5_next（11 个 DSA 层退回泛化 GQA）；
  - Kimi-K3 1.0437 → 0.9994：latent MoE 的 down/up 投影是每层一份、全 token 激活；
  - MiniMax-M3 1.0279 → 0.9995：块稀疏 selected 未夹到可见长度；
  - V4-Flash-Vision-Exp / Kimi-K2 系：登记值等于或宽于默认容差，属无效登记，移除。
- ~~「测试断言为全模型统一 2% 容差」~~（W4 时点表述）→ 已收至 **0.005 + REGISTERED 空**
  （W5 验收收口，2026-09-09）。
