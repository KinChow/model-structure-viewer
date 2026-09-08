# 算子对照参考（Operators Reference）

> **用途**：逐算子人工对齐审查手册。每个算子一节，固定模板：触发面（探针实证）、
> matrix / vector / sfu / bytes 公式与实现位置、单位换算、三级来源标注、适用全局假设、
> 已知近似与登记项、对齐勾选。
> **口径权威链**：公式规格以 [`cost_counts.md`](./cost_counts.md) 为准；本文是它的
> 逐算子展开与实证附录，两者冲突时以 cost_counts.md + 代码注释为准并回改本文。
> **基线**：commit `2e83940`（main），2026-09-08。触发面数字全部来自探针实跑（§1），
> 非代码推断。

---

## 0. 总览表

`matrix` 列：✓ = 有矩阵计费；0 = 精确零（该单元无事可做，principles §3.3）。
`bytes` 列：✓ = 三个访存分量中至少 actIn/actOut 非零；∅ = 全零（A1 view 豁免）。
`来源`：一 = 一等 aten 锚点；二 = 二等 modeling 对照；三 = 三等分解声明（可组合，如 `二+三`）。
`触发模型`：探针发射该 operator_id 叶的模型数 / 59；括号内为 leaf 节点数 / 乘数后实例数。
`对齐`：勾选框供人工逐条核销。

| 算子 | 组 | matrix | bytes | 来源 | 触发模型（节点/实例） | 对齐 |
|---|---|---|---|---|---|---|
| linear | 线性与投影 | ✓ | ✓ | 一 | 59/59（9891/28707） | - [ ] |
| matmul | 注意力·dense | ✓ | ✓ | 一 | 48/59（878/4162） | - [ ] |
| softmax | 注意力·dense | 0 | ✓ | 一 | 48/59（439/2081） | - [ ] |
| attention_output_gate | 注意力·dense | 0 | ✓ | 二 | 29/59（355/355） | - [ ] |
| attention_qkv_split | view 零流量 | 0 | ∅ | 二 | 40/59（41/1147） | - [ ] |
| mla_query_compress | 注意力·MLA | ✓ | ✓ | 三+二 | 19/59（221/1124） | - [ ] |
| mla_kv_compress | 注意力·MLA | ✓ | ✓ | 三+二 | 24/59（464/1369） | - [ ] |
| mla_kv_split | 注意力·MLA/view | 0 | ∅ | 一 | 19/59（221/1124） | - [ ] |
| mla_output_gate | 注意力·MLA | 0 | ✓ | 二 | 1/59（23/24） | - [ ] |
| qsa_indexer | 注意力·DSA/QSA | ✓ | ✓ | 二+三 | 16/59（327/698） | - [ ] |
| qsa_attention | 注意力·DSA/QSA | ✓ | ✓ | 二 | 16/59（327/698） | - [ ] |
| minimax_sparse_indexer | 注意力·块稀疏 | ✓ | ✓ | 二+三 | 2/59（2/114） | - [ ] |
| minimax_sparse_attention | 注意力·块稀疏 | ✓ | ✓ | 二 | 2/59（2/114） | - [ ] |
| dsv4_swa_attention | 注意力·V4 | ✓ | ✓ | 二（降级） | 3/59（3/6） | - [ ] |
| dsv4_compressed_attention | 注意力·V4 | ✓ | ✓ | 二（降级） | 5/59（120/122） | - [ ] |
| linear_attention | 注意力·KDA | ✓ | ✓ | 二 | **0/59**（槽位保留） | - [ ] |
| linear_attention_gate | 注意力·KDA | 0 | ✓ | 二 | **0/59**（槽位保留） | - [ ] |
| gated_delta_attention | 注意力·KDA | ✓ | ✓ | 二 | 34/59（431/1274） | - [ ] |
| gated_rmsnorm | 注意力·KDA | 0 | ✓ | 二+三 | 34/59（431/1274） | - [ ] |
| causal_conv1d | 注意力·KDA | ✓ | ✓（weights=0*） | 一 | 34/59（431/1274） | - [ ] |
| attention_residual | 注意力·K3 | ✓ | ✓ | 二+三 | 1/59（47/93） | - [ ] |
| split | view 零流量 | 0 | ∅ | 一 | 36/59（605/726） | - [ ] |
| qwen_qkvz_split | view 零流量 | 0 | ∅ | 二 | 31/59（383/1137） | - [ ] |
| rmsnorm | 归一化与激活 | 0 | ✓ | 三 | 57/59（1898/8633） | - [ ] |
| gemma_rmsnorm | 归一化与激活 | 0 | ✓ | 三 | 31/59（2179/4287） | - [ ] |
| swiglu | 归一化与激活 | 0（routed ✓） | ✓ | 一 | 59/59（2161/5774） | - [ ] |
| rope | 归一化与激活 | 0 | ✓ | 三 | 59/59（1101/2393） | - [ ] |
| topk | MoE | 0 | ✓ | 一 | 44/59（916/2565） | - [ ] |
| moe_dispatch | MoE | 0 | ✓ | 一 | 44/59（926/2580） | - [ ] |
| moe_combine | MoE | 0 | ✓ | 三 | 44/59（926/2580） | - [ ] |
| moe_add | MoE | 0 | ✓ | 一 | 41/59（873/2422） | - [ ] |
| shared_expert_gate | MoE | 0 | ✓ | 二 | 16/59（426/844） | - [ ] |
| dsv4_hash_route | MoE | 0 | ✓ | 二 | 5/59（10/15） | - [ ] |
| mhc_pre | 多流残差复合 | ✓ | ✓ | 三 | 7/59（292/341） | - [ ] |
| mhc_post | 多流残差复合 | ✓ | ✓ | 三 | 7/59（7/7） | - [ ] |
| mhc_fused_post_pre | 多流残差复合 | ✓ | ✓ | 三 | 7/59（292/341） | - [ ] |
| mhc_contract | 多流残差复合 | 0 | ✓ | 三 | 7/59（7/7） | - [ ] |
| hyper_connection | 多流残差复合 | ✓ | ✓ | 三 | 2/59（106/194） | - [ ] |
| ple | 多流残差复合 | ✓ | ✓ | 三 | 2/59（2/2） | - [ ] |
| vision_position | 视觉族 | 0 | ✓ | 三 | 38/59（38/38） | - [ ] |
| vision_merge | 视觉族 | 0 | ✓ | 三 | 31/59（31/31） | - [ ] |
| vision_activation | 视觉族 | 0 | ✓ | 一 | 36/59（65/974） | - [ ] |
| — embedding(struct) 结构节点 | embedding/输出头 | 0 | ✓（weights=0） | 三（M11-P0-5 分解声明） | 57/59（57/57） | - [ ] |
| — attention 模块容器（type=attention） | 注意力·容器 | 计费 0 | 计费 0 | — | 59/59（容器，不计费） | - [ ] |

\* 运行时 case 的 weights=0，见 causal_conv1d 节「已知近似」。

探针口径结论：59 模型共发现 **41 种 leaf 键** = 40 条 registry operatorId（42 条中
`linear_attention`、`linear_attention_gate` 为零触发通用槽位）+ 1 个结构节点
`embedding`。prefill 与 decode（S=4096）两相位 unknown 叶均为 **0**。

---

## 1. 探针方法与口径（可复现）

```bash
node --input-type=module -e "
import { normalizeConfig } from './frontend/src/structure/config/normalize.js';
import { resolveArchitecture } from './frontend/src/structure/registry/resolveArchitecture.js';
import { buildNetwork } from './frontend/src/structure/model_executor/models/index.js';
import { createStructureIr } from './frontend/src/structure/ir/createStructureIr.js';
import { materializeModelStructure } from './frontend/src/structure/materializers/toStructureNode.js';
import { computeNodeCosts } from './frontend/src/cost/compute.js';
// 遍历 models/catalog.json 全部 59 模型：config → normalize → resolve → buildNetwork
// → createStructureIr → materialize → computeNodeCosts(root, normalized,
//   { batch:1, sequence:128, phase:'prefill', graph })，逐叶收集
// attributes.operator_id（含 type==='embedding' 结构节点）。
"
```

统计口径：

- **leaf** = 无 children 的节点；父节点（模块/容器）不携带动作向量
  （`compute.js:43-48`，aggregate 链由子节点累加）。
- **operator_id 分派**：`type==='operator'` 取 `attributes.operator_id`；
  `type==='embedding'` 记为结构节点 `embedding`；
  `type==='attention'` 为注意力模块容器（恒有子叶，自身不计费，见 §3.8）。
- **节点数 vs 实例数**：节点数 = 结构树中 leaf 出现次数；实例数 = Σ multiplier
  （层组 repeat 倍乘，`traverse.js:40-55`）。下文记作「节点/实例」。
- **相位**：prefill（T=seq）与 decode（T=1、S=上下文全长）均跑过，unknown 叶皆 0
  （与 `builtinModels.test.js:54-55` 的 `computeComplete` 断言一致）。
- 数值快照存档：`/tmp/probe-out.json`、`/tmp/probe2-out.json`（临时文件，本文已内联
  全部所需数字）。

---

## 2. 记号、单位与全局假设速查

- 记号：`T`=tokens（phase 决定）、`S`=可见 key tokens、`H`=hidden、`Nh`=query 头数、
  `kvH`=KV 头数、`D`=head_dim、`dv`=value head_dim、`I`=intermediate、`E`=专家数、
  `k`=topk、`n`=流数、`b`=每元素字节（主链固定 `bpe=2`，`compute.js:23`）。
- 单位（principles §3.1）：`matrix` 存 **MACs**（aten FLOPs 公式含 2×，抄时换算）、
  `vector` 存 flop、`sfu` 存操作次数、`bytes` 为每次前向 compulsory traffic
  （权重读一遍 + 输入读 + 输出写，读写各一次口径，**无 phase 分支**）。
- 全局假设 A1–A7 全文见 [`cost_counts.md`](./cost_counts.md)「全局假设」表：
  A1 split/view 零流量；A2 softmax 融合单遍；A3 rope sin/cos 查表；
  A4 复合分解逐条标注；A5 SFU 口径（sigmoid=2、exp=1、rsqrt=1、div=1）；
  A6 线性注意力 per-token 递推下界；A7 融合算子按语义分解、流量不折算。
- 九个共享 counts 实现 F1–F9 全部在 `frontend/src/structure/formulas/counts.js`
  （下称 counts.js），本文每节「实现」字段给出 file:line。

---

## 3. 注意力族

### 3.1 dense 分解链（scores / softmax / context 三叶计费）

#### matmul — MatMul

- **触发面**：48/59 模型（878 节点/4162 实例）。凡注意力走**分解链**的模型每层发射
  scores + context 两叶：MiniMax-M2.7（GQA 文本塔 ×62）、GLM-4.7（GQA ×89+3）、
  Qwen3.5/3.6/3.8 全系 full-attention 层（attention_kind=qwen35_full）、
  MLA 家族的文本塔（DeepSeek-R1/V3.1、Kimi 全系 ×60 或 ×61、Kimi-K3 ×24）、
  以及**全部视觉塔**（MiniMax-M3 ×32、Qwen 系 ×12–28、Kimi ×27、GLM-5.3-Flash ×24、
  V4-Vision-Exp ×32）。缺 matmul 的 11 个模型 = 文本侧整体融合的稀疏家族且无视觉塔
  （DeepSeek-V3.2、V4×4 非 vision、GLM-5/5.1/5.2/5.3 系 6 个）。
- **matrix**：scores 叶 `Nh·T·S·D` + context 叶 `Nh·T·S·dv`（prefill S=T、decode T=1）。
  实现：extractor case "matmul"（`extractor.js:380-421`），scores/context 用
  output_shape 模式匹配区分（`extractor.js:88-101`；context 模式先判防吞）。
- **vector / sfu**：精确零（两叶都是纯 GEMM；softmax 归 softmax 叶）。
- **bytes.weights**：0——注意力无权重，投影由 linear 叶计费。
- **bytes.actIn / actOut**：scores 叶 actIn=`(Nh·T·D + S·Nh·D)·b`（读 Q、K）、
  actOut=`Nh·T·S·b`（写 scores）；context 叶 actIn=`(Nh·T·S + S·Nh·dv)·b`
  （读 scores、V）、actOut=`Nh·T·dv·b`（写 context）（`extractor.js:394-419`）。
- **单位与换算**：matrix 为 MACs；`aten.bmm` 的 `2·m·n·k` FLOPs 已 ÷2。
- **来源**：一等 `aten::bmm` ×2（`formulas/index.js:39-41` ref 注释）。
- **全局假设**：无特别引用（一阶读写各一次为本仓默认口径）。
- **已知近似/登记**：
  - **同名双轨**：registry 条目 `matmul.counts=attentionCounts`（F2 融合口径，
    `index.js:37-46`）在运行时被手搓 case 遮蔽（switch 提前 return），注册表引用是
    §3.1b 护栏认证过的「死引用」——护栏与迁移方向见 §11。
  - K 读宽按 `Nh·D`（repeat_kv 物化口径）而非 `kvH·D`：分解链把 GQA 的 K/V 扩展
    视为叶间真实张量传递；融合口径（F2）按 kvH 缩。两口径并存是有意分工
    （`counts.js:35-36` 注释），对齐时注意 dense 分解链不随 kvH 缩流量。
  - decode 的 keyTokens=options.sequence（上下文全长），与 F2 decode 行为一致。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### softmax — Softmax

- **触发面**：与 matmul 完全同集合，48/59 模型（439 节点/2081 实例）——分解链每对
  scores/context 之间一叶。
- **matrix**：精确零（归约+逐元素，不用矩阵单元）。
- **vector / sfu**：`vector=3·elements`（max/sum 归约 + 乘）、`sfu=2·elements`
  （exp + div），elements=`Nh·T·S`。实现：`softmaxCounts`（`counts.js:202-209`），
  extractor case "softmax"（`extractor.js:563-570`）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：各 `Nh·T·S·b`（读 logits 一遍、写 probs 一遍，A2 单遍）。
- **单位与换算**：vector=flop；sfu=操作次数（A5：exp=1、div=1）。
- **来源**：一等 `aten::_softmax`；torch flop_counter 明确不数 softmax，本仓**有意
  超越**计 vector/sfu/bytes（`index.js:49-51` ref 注释）。
- **全局假设**：A2（融合单遍；多遍读放大不建模）、A5。
- **已知近似/登记**：flash kernel 在线 softmax 不落地全量 scores/probs 时，本叶与
  F2 系融合注意力内的 `2·scores` 项同属「理论口径」——保持一致优先
  （refactor_plan M11「A2 口径声明」）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### attention_output_gate — Attention Output Gate

- **触发面**：29/59 模型（355 节点/355 实例；每 full-attention 层一个显式节点，
  不进层组乘数）——Qwen3.5/3.6/3.8 的 qwen35_full 模板
  （`ops/index.js:288-292`，fused QKV 投影里第 2 段 gate 经 sigmoid 调制输出）。
- **matrix**：精确零。
- **vector / sfu**：`vector=T·W`、`sfu=2·T·W`（sigmoid = exp+rcp，A5）。实现：
  `gateCounts`（`counts.js:84-95`），extractor 四门控共用 case（`extractor.js:580-584`）。
- **bytes.weights**：0（gate 向量来自 qkv_gate_proj 输出切片，无独立投影权重；
  gateCounts 的 `gateProjection` 分支在本仓四门控叶均未启用）。
- **bytes.actIn / actOut**：actIn=`T·W·b`（gate+context 拼接读，按宽 W 一阶计）、
  actOut=`T·W·b`。
- **单位与换算**：W = output_shape 末维（= Nh·D）。
- **来源**：二等 modeling 对照——vLLM/SGLang `fused_sigmoid_mul`
  （`ops/index.js:290` implementation 指针；`index.js:398-401` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：`activation: sigmoid|none` 由 plan 决定（`ops/index.js:291`）；
  activation=none 的模型本叶仍存在但语义为恒等——探针未区分（登记）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 3.2 MLA（DeepSeek / Kimi / GLM MLA）

#### mla_query_compress — MLA Query Compression

- **触发面**：19/59 模型（221 节点/1124 实例）：DeepSeek-R1/V3.1/V3.2（2 节点×61）、
  Kimi-K2 全系/K2.5/2.6/2.7（2×61）、Kimi-K3（23×24）、GLM-5/5.1（2×78）、
  GLM-5.2/5.3 系（38×78）、GLM-5.3-Flash×2（11×11）——即全部 MLA/DSA 文本塔的
  q_a 投影 + q_a_layernorm（q_b 由独立 q_b_proj linear 叶计）。
- **matrix**：`= T·qLora·H`（q_a 投影 GEMM；q_b 不在本叶）。实现：ctxBuilder 组合
  `F1(qa) + F3(norm)`（`index.js:265-276` 注册表；`extractor.js:682-687` ctx）。
- **vector / sfu**：norm 段 `4·T·qLora` + `T`（rsqrt）；sfu=`T`。
- **bytes.weights**：`(qLora·H + qLora)·b`（q_a 权重 + norm 权重）。
- **bytes.actIn / actOut**：actIn=`(T·H + T·qLora)·b`（x 读 + norm 读）、
  actOut=`2·T·qLora·b`（投影写 + norm 写；下游 q_b 再读）。
- **单位与换算**：F1 的 `2×` 已换算 MACs；F3 vector 逐 flop。
- **来源**：三等分解声明 + 二等对照 models/deepseek-ai/DeepSeek-V3.1/
  modeling_deepseek.py（q_a_proj :661、q_a_layernorm :664、q_b_proj 调用点 :769）
  （`index.js:267-269` ref 注释）。
- **全局假设**：A4（分解声明）、A5。
- **已知近似/登记**：组合里**不含 q_b**——2026-09-07 审计发现含 qb 会与独立
  q_b_proj 叶双计（Kimi/GLM 各 +19M/+25M 参数/层）（`extractor.js:683-686` 注释）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### mla_kv_compress — MLA KV Compression

- **触发面**：24/59 模型（464 节点/1369 实例）：上列 19 个 MLA 模型 + DeepSeek-V4×5
  （V4 的 compressor 叶，Flash 系 41 节点/模型、Pro 系 60×61）。kv_a 投影 + latent
  拆分前的投影段；**latent/压缩态 cache 写回由本叶 actOut 计**。
- **matrix**：`= T·outW·H`；outW 以节点自身 output_shape 为权威
  （MLA latent = kvLora+rope；V4 压缩 = 2·D·k）。实现：ctxBuilder `F1(proj)`
  （`index.js:277-287`；`extractor.js:688-701`）。
- **vector / sfu**：精确零（组合仅 F1）。
- **bytes.weights**：`outW·H·b`。
- **bytes.actIn / actOut**：actIn=`T·H·b`；actOut=`T·outW·b`（= latent/压缩态 cache 写）。
- **单位与换算**：F1 MACs 换算同上。
- **来源**：三等分解声明 + 二等对照 DeepSeek-V3.1 modeling_deepseek.py
  kv_a_proj_with_mqa（:669）（`index.js:279-281`）；V4 compressor 语义
  /tmp/m11-formulas/dsv4.md §(b)6。
- **全局假设**：A4。
- **已知近似/登记**：V4 ctx 失配已修（2026-09-08）——旧式用
  `kvLoraRank+qkRopeHeadDim` 拼 out 维，V4 无 kvLoraRank 得 out=64，compressor macs
  差 16–32×；现以 `staticWidth(output_shape)` 为权威（`extractor.js:689-701` 注释，
  refactor_plan 缺陷 3）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### mla_output_gate — MLA Output Gate

- **触发面**：1/59 模型——moonshotai/Kimi-K3（23 节点×24，每 MLA 层一个）。ref 注释
  「目录仅 Kimi-K3 发射此叶」与探针一致。
- **matrix**：精确零。
- **vector / sfu**：`vector=T·W`、`sfu=2·T·W`（sigmoid 门乘，A5）。实现：`gateCounts`
  （`counts.js:84-95`；`extractor.js:580-584`）。
- **bytes.weights**：0（门来自融合投影输出，本叶无独立 W_g；registry 公式
  `O'=sigmoid(W_g x)·O` 中的 W_g 若独立成叶应由 linear 计）。
- **bytes.actIn / actOut**：各 `T·W·b`。
- **单位与换算**：W = output_shape 末维。
- **来源**：二等对照 models/moonshotai/Kimi-K3/modeling_kimi_linear.py
  （mla_use_output_gate :398、门乘 :470）（`index.js:299-301`）。
- **全局假设**：A5。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 3.3 DSA / QSA 稀疏注意力（三分支共用 qsa_attention case）

#### qsa_indexer — QSA Indexer

- **触发面**：16/59 模型（327 节点/698 实例），与 qsa_attention 完全同集合：
  Qwen3.8-Flash-Next×2（qsa，12×12）、DeepSeek-V3.2（dsa_sparse_mla，2×61）、
  DeepSeek-V4×5（dsv4_sparse_mla，Flash 系 21、Pro 系 30）、GLM-5/5.1（2×78）、
  GLM-5.2/5.3 系（38×78）、GLM-5.3-Flash×2（11×11）。
- **matrix**：indexer 打分 `Nh_i·T·S·D_i` ×2（F2 双 bmm，S=上下文全长）。实现：
  ctxBuilder 组合 `F2(score) + F8(topk)`
  （`index.js:352-363`；`extractor.js:702-705`：indexerNHeads/indexerHeadDim 独立参数）。
- **vector / sfu**：F2 段 `3·Nh_i·T·S` + `2·Nh_i·T·S`；topk 段 vector=`T·S`
  （experts=S）、sfu=`T·budget`（normTopkProb 除法）。
- **bytes.weights**：0（indexer 投影 W_q/W_k 由相邻 linear 叶计）。
- **bytes.actIn / actOut**：F2 段一阶读写（Q/K 全上下文读、scores 读写）+ topk 段
  actIn=`T·S·b`（全量打分读）、actOut=`T·budget·b`（**top-k 索引写出**，供
  qsa_attention 读）。
- **单位与换算**：budget=indexerBudget（`index_topk`/`indexer_budget` 直读，
  normalize.js:180）。
- **来源**：二等（vLLM.SparseAttnIndexer / DeepseekV4Indexer，
  ops/index.js:393-405 implementation 指针；V3.2/V4 modeling 未入库——离线取证）+
  三等分解（F2+F8）；Qwen3.8-Flash-Next / GLM-5.3-Flash 的 indexer 源码已于
  2026-09-08 入库升级为二等：models/Qwen/Qwen3.8-Flash-Next/modeling_qwen4_exp.py
  Qwen4ExpTextQSAIndexer（:671-687）、models/zai-org/GLM-5.3-Flash/
  modeling_glm5_next.py Glm5NextTextIndexer（:739-880）。依据
  /tmp/m11-formulas/evidence-qsa-glm.md 裁决一/二。
- **全局假设**：A2（indexer 打分的 softmax 段）、A4。
- **已知近似/登记**：topk actOut 宽度实为 `budget+compress_ratio-1`（Qwen :722 /
  GLM :870-872 的尾块补选），现按 `T·budget` 上限计——登记为理论口径。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### qsa_attention — QSA Sparse Attention

- **触发面**：同 qsa_indexer，16/59 模型（327 节点/698 实例）。三种
  attention_kind 共用本 case：`qsa`（Qwen3.8-Flash-Next×2 逐头 GQA/MHA）、
  `dsa_sparse_mla`（DeepSeek-V3.2 + GLM-5 系×8，MLA latent）、
  `dsv4_sparse_mla`（V4×5，MQA 压缩态）。
- **matrix**：`Nh·T·S_sel·(D+dv)`，S_sel=`min(S, indexerBudget)`。实现：extractor
  case "qsa_attention"（`extractor.js:422-467`；矩阵镜像 qsaCoreMacs
  `extractor.js:186-193`）。
- **vector / sfu**：精确零（融合核；softmax 段以 bytes 中间量体现）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**（F2 整体口径，按 kind 分派读宽，`extractor.js:428-465`）：
  - `dsa_sparse_mla`：kvH=1，K 读宽 `kvLora+rope`、V 读宽 `kvLora`，**无 kvWrite**
    （latent cache 写由 kv_a_proj linear actOut 计）；
  - `dsv4_sparse_mla`：kvH=config.kvH（=1），读宽 D/dv，无 kvWrite（压缩态写归
    compressor 叶）；
  - `qsa`：kvH=config.kvH，读宽 D/dv，**计 kvWrite**=`kvH·T·(D+dv)`（paged cache
    写回模板内无叶承担）。
  公式：actIn=`(Nh·T·D + kvH·S_sel·(kW+vW) + T·S_sel + 2·Nh·T·S_sel)·b`；
  actOut=`(2·Nh·T·S_sel + Nh·T·dv + kvWrite)·b`。
- **单位与换算**：`T·S_sel` 项 = top-k 索引读（int32 按 b 计，由 qsa_indexer topk
  actOut 写、本叶读）；`2·scores` ×2 = A2 的 scores/probs 写+读。
- **来源**：二等 modeling 对照（FlashMLA-sparse 吸收式核按 latent 读宽、逐头变体按
  kvH——变体矩阵见 cost_counts.md F2 表）；bytes 结论 /tmp/m11-formulas/qsa.md
  §2.3-2.4、§4.3；GLM-5.3-Flash 证据改判 qsa→dsa_sparse_mla 见
  /tmp/m11-formulas/evidence-qsa-glm.md 裁决二（2026-09-08）。
- **全局假设**：A2（4·scores 记本节点——稀疏模板无独立 softmax 叶）、A5。
- **已知近似/登记**：prefill 因果三角未折减（与 F2 全仓口径一致，保守）；
  并集口径 vs 预算口径偏差见 qsa.md §5.2；kvWrite 取舍规则一句话版：
  「模板里已有叶写了 cache 条目的（latent=kv_a_proj、压缩态=compressor），融合核
  不再写；没有的（逐头 K/V）由融合核补记」（qsa.md §4.3）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 3.4 MiniMax M3 块稀疏

#### minimax_sparse_indexer — MiniMax M3 Block Indexer

- **触发面**：2/59 模型——MiniMax-M3 / M3-MXFP8（1 节点×57，稀疏层组整体乘数）。
- **matrix**：index 头块打分 `Nh_i·T·S·D_i` ×2（F2，S=上下文全长，index_block_size=128
  池化在建模上并入打分宽）。实现：ctxBuilder `F2(score) + F8(topk)`
  （`index.js:407-418`；`extractor.js:706-709`：sparseIndexHeads/sparseIndexDim）。
- **vector / sfu**：F2 段 `3·Nh_i·T·S` + `2·Nh_i·T·S`；topk 段 vector=`T·S`、
  sfu=`T·sparseTopkBlocks`（=16）。
- **bytes.weights**：0（index q/k 投影由相邻 qkv_index_proj linear 叶计）。
- **bytes.actIn / actOut**：F2 一阶读写 + topk actIn=`T·S·b`、actOut=`T·blocks·b`
  （块 id 写出）。
- **来源**：二等对照 models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py
  MiniMaxM3VLIndexer（:492：index_block_size=128 池化打分 + topk_blocks=16 选块；
  index-value 路 checkpoint 显式关闭）+ 三等分解（`index.js:409-412` ref 注释）。
- **全局假设**：A2、A4。
- **已知近似/登记**：init/local blocks 保留（sparseInitBlock+sparseLocalBlock）在
  attention 侧生效，indexer 不计；index-value 路显式关闭不建模。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### minimax_sparse_attention — MiniMax M3 Block-Sparse GQA

- **触发面**：2/59 模型——MiniMax-M3 / M3-MXFP8（1 节点×57）。
- **matrix**：`Nh·T·(blocks·blockSize)·(D+dv)`，blocks=sparseTopkBlocks+Init+Local、
  blockSize=sparseBlockSize。实现：extractor case
  （`extractor.js:468-499`；矩阵镜像 minimaxSparseCoreMacs `extractor.js:194-201`）。
- **vector / sfu**：精确零（融合核）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`(Nh·T·D + kvH·selected·(D+dv) + 2·Nh·T·selected)·b`
  （Q 读 + 选中 KV 读 + scores/probs 中间量）；actOut=`(2·Nh·T·selected + Nh·T·dv +
  kvH·T·(D+dv))·b`——**含 kvWrite**（M3 稀疏注意力为融合算子，cache 写回在本叶；
  dense 侧由 k/v_proj linear actOut 计，两侧账目自洽，`extractor.js:477-484` 注释）。
- **来源**：二等对照 modeling_minimax_m3_vl.py MiniMaxM3VLAttention（:408）+
  eager_attention_forward（:340）；transformers 库版入库 models/MiniMaxAI/MiniMax-M3/
  （HF 仓库无 modeling，二等降级口径见 evidence-manifest.json）。
- **全局假设**：A2（4·scores 记本节点）。
- **已知近似/登记**：选块 per query token、per KV 组（index_heads=kv_heads）
  （`extractor.js:484` 注释）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 3.5 DeepSeek V4 双 case（swa / compressed）

#### dsv4_swa_attention — DeepSeek V4 Sliding-Window MQA

- **触发面**：3/59 模型——DeepSeek-V4-Flash / -0731 / -Vision-Exp（1 节点×2，L0-1
  层组）。compress_ratio=0 层（Pro 系无此叶）。
- **matrix**：`Nh·T_q·visible·(D+D)`，visible=`min(S, slidingWindow)`（prefill）；
  **decode available=1 是 legacy 行为**（`extractor.js:140-147` 镜像，有意保留）。
  实现：extractor case（`extractor.js:500-533`）。
- **vector / sfu**：精确零。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`(Nh·T_q·D + kvH·K_w·D + 2·scores)·b`（Q 读 + KV
  窗口 latent 读**一份**（K/V 共享单一 headDim 宽 latent）+ scores/probs）；
  actOut=`(2·scores + Nh·T_q·dv + kvH·T_q·D)·b`（含 kvWrite，宽=D）。
- **单位与换算**：K_w=`min(S, slidingWindow)`（decode 时 S=上下文长，窗口语义自然
  成立）。
- **来源**：二等 modeling 对照（vLLM.DeepseekV4SWACache / MQALayer，
  ops/index.js:419-427 implementation 指针）+ **降级声明**：V4 modeling 全网 404
  （HF 直连 + hf-mirror + ModelScope 探针全 ✗），依据 = config 字段 + ops 模板
  shape + 权重 index 实证（无 V 扩展投影 → K/V 共享 512 宽 latent）+
  memory.js dsv4 分支；/tmp/m11-formulas/dsv4.md §(b)5。
- **全局假设**：A2（2·scores 读写）。
- **已知近似/登记**：若真实实现缓存 `[K|V]=2D`/token，则 KV 读与 kvWrite 各低估
  2×（绝对差 ≤0.26MB/层·窗口，prefill 占比 <0.1%——dsv4.md §(f) 中置信项）；
  matrix 的 decode available=1 与 bytes 的真实窗口不对称是有意保留的 legacy 行为
  （dsv4.md §(e)6）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### dsv4_compressed_attention — DeepSeek V4 Compressed MLA

- **触发面**：5/59 模型——V4-Flash 系 20 节点/模型（L3,5,…,41 ratio=128 层）、
  V4-Pro 系 30 节点×31（L0-1 组 ×2 + 奇数层）。全 120 节点/122 实例。
- **matrix**：`Nh·T_q·ceil(S/ratio)·(D+D)`（legacy 镜像，decode available=1 同上）。
  实现：extractor case（`extractor.js:534-562`）。
- **vector / sfu**：精确零。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`(Nh·T_q·D + 2·kvH·K_c·D + 2·scores)·b`（读压缩
  缓存：每压缩位 K/V 态各 D，共 2·D）；actOut=`(2·scores + Nh·T_q·dv)·b`——
  **无 kvWrite**（压缩态写入归 compressor=mla_kv_compress 叶 actOut，防双计）。
- **单位与换算**：K_c=`ceil(S/ratio)`（decode 时 S=上下文长）。
- **来源**：二等 modeling 对照（compress_ratio=128，compressor 输出宽 2·D 实证：
  ops/index.js:387 + memory.js (2·D)/ratio 摊销）+ 降级声明同上；
  /tmp/m11-formulas/dsv4.md §(b)6。
- **全局假设**：A2。
- **已知近似/登记**：压缩层是否同时读原始滑窗 KV 未定（memory.js 显示每层维护
  滑窗 KV，legacy matrix 未计 +W；若 vLLM 实为 hybrid 需补 `min(ctx,W)·D` 一项——
  dsv4.md §(f) 残余登记）；`attn_sink` 每层 ≈Nh 参数未计（量级可忽略）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 3.6 线性注意力 / KDA

#### linear_attention — Gated Linear Attention（generic 槽位）

- **触发面**：**0/59 模型**（探针无一叶）——generic plain 变体的通用槽位保留
  （`index.js:184-185` ref 注释明示「目录 0 模型发射此叶」）。现网 KDA 家族
  （Qwen3.5/3.6/3.8、Kimi-K3、GLM-5.3-Flash）全部经 `gated_delta_attention` 叶。
  运行时 case 按节点路径分派：`/short_conv|conv/` → 卷积分支、`/state|recurrent/` →
  state 分支（`extractor.js:632-650`），当前无可达叶。
- **matrix**：plain 递推 `2·T·Nh·dk·dv`（外积 + query）。实现：
  `linearAttentionStateCounts({delta:false})`（`counts.js:140-152`）。
- **vector / sfu**：vector=`T·Nh·dk·dv`（decay 乘）；sfu=`Nh·T`（exp decay）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`2·T·Nh·dk·dv·b`、actOut=`T·Nh·dk·dv·b`
  （**递推状态读+写主导**：每 token 状态读+写一遍）。
- **单位与换算**：dk/dv 为每头维度，Nh 显式（state=Nh·dk·dv）。
- **来源**：二等 modeling 对照（Gated DeltaNet arXiv 2412.06464 递推语义，generic
  plain 变体）+ 缺失声明（目录 0 模型）（`index.js:184-185`）。
- **全局假设**：A6（per-token 递推下界；chunked 总量等价）。
- **已知近似/登记**：槽位保留，无现网触发；若未来模型走 plain 变体需补探针复核。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### gated_delta_attention — Gated Delta Attention（KDA 统一叶）

- **触发面**：34/59 模型（431 节点/1274 实例）：Qwen3.5 全系 21 + Qwen3.6×4 +
  Qwen3.8×6（2.4T/27B/Flash-Next）+ Kimi-K3（24×69）+ GLM-5.3-Flash×2（12×34）。
  linearAttentionMode ∈ {qwen3_5, qwen4_exp, kimi, kimi_k3, glm5_next}（全 delta）。
- **matrix**：`3·T·vh·dv·dk`（外积 + delta matvec + query；delta matvec 属矩阵
  MACs——2026-09-07 数学修正）。实现：`stateUpdateCounts` →
  `linearStateUpdateMacs`（`extractor.js:281-306`；delta 镜像
  `counts.js:127-152`）。
- **vector / sfu**：vector=`2·T·vh·dv·dk`（decay 乘 + delta 残差）；sfu=`Nh·T·3`
  （exp decay + beta sigmoid，A5）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：stateBytes=`(convElements·kernel + vh·dv·dk)·b`（conv
  环形历史 + 递归矩阵，与 memory.js linearStateElementsPerLayer 同源同式，vLLM
  MambaStateShapeCalculator.kda_state_shape）；actIn=actOut=stateBytes
  （**状态驻留 HBM，每 forward 读+写各一遍**，chunk 内不逐 token 重读，
  `extractor.js:289-306` 注释）。
- **单位与换算**：vh=Nh（KDA 值头数=键头数）；beta/A_log/dt_bias 参数化是属性级，
  零计数影响。
- **来源**：二等对照 models/moonshotai/Kimi-K3/modeling_kimi_linear.py
  KimiDeltaAttention（:477：beta sigmoid、safe decay exp(g)、S·k matvec）；
  Qwen3.5/GLM-5 系同族（delta=true）（`index.js:203-214` ref 注释）。
- **全局假设**：A6；执行形态假设 per-token 递推。
- **已知近似/登记**：conv 历史与递归状态合并计状态访存（stateUpdateCounts，
  M11-P0-5）；layer0 multiplier=1 的 4× 排查记录见 identity_calibration.md。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### linear_attention_gate — Linear Attention Output Gate（generic 槽位）

- **触发面**：**0/59 模型**——通用槽位保留（`index.js:194-196`：「目录 0 模型发射
  此叶」）。KDA 输出门 z·y 路实际经 `gated_rmsnorm` 叶（FusedRMSNormGated 语义）。
- **matrix**：精确零。
- **vector / sfu**：`vector=T·W`、`sfu=2·T·W`。实现：`gateCounts`
  （`counts.js:84-95`；extractor 四门控共用 case `extractor.js:580-584`）。
- **bytes.weights / actIn / actOut**：weights=0；actIn=actOut=`T·W·b`。
- **来源**：二等对照 Kimi-K3 modeling_kimi_linear.py FusedRMSNormGated（门乘路）
  + 缺失声明（`index.js:194-196`）。
- **全局假设**：A5。
- **已知近似/登记**：槽位保留。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### gated_rmsnorm — Gated RMSNorm

- **触发面**：34/59 模型（431 节点/1274 实例），与 gated_delta_attention 同集合——
  KDA 递推输出的 gated 归一化（Qwen3.5 `output_gate_norm`、Kimi-K3、GLM-Flash）。
- **matrix**：精确零。
- **vector / sfu**：vector=`4·T·H_n`（F3）`+ T·H_n`（门乘）= 5·T·H_n；
  sfu=`T + 2·T·H_n`（rsqrt + sigmoid）。实现：`rmsnormCounts({gated:true})`
  （`counts.js:67-79`；`extractor.js:578-579`）。
- **bytes.weights**：`H_n·b`（norm weight）。
- **bytes.actIn / actOut**：actIn=`2·T·H_n·b`（o + gate 两路读）；actOut=`T·H_n·b`。
- **单位与换算**：H_n = input_shape 末维（per-head gated 时为头维×头数展开宽）。
- **来源**：二等对照 Kimi-K3 modeling_kimi_linear.py FusedRMSNormGated（:539，逐头
  门控）+ 三等分解 = F3(gated)（`index.js:215-224` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：phi 由模型配置决定（sigmoid/SiLU），sfu 统一按 sigmoid=2 计
  （配置差异不区分——登记）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### causal_conv1d — Causal Short Convolution

- **触发面**：34/59 模型（431 节点/1274 实例），与 KDA 同集合：每个线性注意力层的
  q/k/v 短卷积（Qwen3.5 系 `conv`、Kimi-K3、GLM-Flash）。
- **matrix**：`T·width·kernel`，width=`2·keyProj + valueProj`（q/k/v 卷积通道宽）。
  实现：extractor case（`extractor.js:620-631`）。
- **vector / sfu**：运行时精确零（见「已知近似」）。
- **bytes.weights**：运行时 0（registry 版为 `channels·kernel·b`，未在运行时生效）。
- **bytes.actIn / actOut**：各 `T·width·b`（读输入窗口宽、写同宽输出，M11-P0-5）。
- **单位与换算**：matrix 为 MACs；`aten::conv1d` 的 `C_out·C_in·k·T` FLOPs 含 2×
  已换算（`index.js:68-69`）。
- **来源**：一等 `aten::conv1d` + SiLU 2 SFU/元素（A5）（`index.js:66-70` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：**双轨口径差**——注册表 `causalConvCounts`（counts.js:118-125）
  含 SiLU 激活段（vector=T·width、sfu=2·T·width、weights=channels·kernel·b），运行时
  手搓 case 只计 matrix+actIn/actOut（vector/sfu/weights=0）。激活与 conv 权重流量
  目前未计——登记为对齐审查项（§3.1b 双轨，见 §11）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 3.7 Kimi-K3 residual bank

#### attention_residual — Attention Residual

- **触发面**：1/59 模型——moonshotai/Kimi-K3（47 节点×93：每层 attention 前 + MLP 前
  各一 + 层组乘数；config attn_res_block_size=12）。
- **matrix**：score 小投影 `T·H·1`（[1,H] 投影）。实现：ctxBuilder 组合
  `F3(norms) + F1(scoreProj) + F2(aggregate) + add(mix)`
  （`index.js:308-319`；`extractor.js:710-715`）。
- **vector / sfu**：F3×2 = `8·T·H` + `2T`；F2（流数维 softmax）= `3·T·H` + `2·T·H`；
  add = `T·H`。
- **bytes.weights**：`H·b`（score 投影）+ `2·H·b`（两个 norm weight）。
- **bytes.actIn / actOut**：各分量读写一次之和（norms `2·TH·b`、scoreProj `TH·b`、
  aggregate `TH·b`、mix `2·TH·b` → actIn ≈ `6·TH·b`；actOut ≈ `4·TH·b`；以
  sumCounts 实算为准）。
- **来源**：二等对照 Kimi-K3 modeling_kimi_linear.py（use_attn_residuals :907、
  _forward_attn_residual :931、attn_res_block_size=12）+ 三等分解
  （`index.js:310-313` ref 注释）。
- **全局假设**：A4、A5；snapshot bank 的存储流量不计（只计每次前向的读写动作）。
- **已知近似/登记**：block 写层的 snapshot 保存流量未单列（并入 mix/add 一阶口径）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 3.8 attention 模块容器（非 registry，零计费）

`type==='attention'` 的模块节点（gqa/sparse/qwen35_full/linear/qsa/mla/
dsa_sparse_mla/dsv4 八种 attention_kind，探针 59 模型共 95 个 kind×model 实例：
qwen35_full 29、linear 34、mla 10、dsa_sparse_mla 9、dsv4 5、gqa 4、sparse 2、
qsa 2）恒为**父容器**（children=算子叶），`computeNodeCosts` 对其计 0、actions=null。
extractor 仍保留 type==="attention" 分支的 legacy 镜像（`extractor.js:329-338`，
返回 attentionCoreMacs 等），但探针证实 **59 模型 0 个无子 attention 叶**——该分支
当前无可达叶，属 W5 随旧链清理的 legacy 镜像（cost_counts.md「M11 现状」）。

---

## 4. 线性与投影

### linear — Linear

- **触发面**：59/59 模型（9891 节点/28707 实例）——全部投影形态：q/k/v/o、MLP
  gate/up/down、fused QKV（qkv_gate_proj、qkv_index_proj、fused_wqa_wkv、qkvz）、
  KDA 的 ba/decay 投影、MoE shared experts、indexer 投影、**lm_head**
  （output head，如 Qwen3.5-0.8B `[−1,−1,1024]→[−1,−1,248320]`）、视觉 patch_embed
  （如 M3 `[−1,−1,3,196]→[−1,−1,1280]`，conv 语义以 GEMM 计）。单模型节点数
  12（GLM-4.7）～569（Kimi-K3）；MoE 专家主干 GEMM 不在本叶（见 swiglu routed）。
- **matrix**：`T·out·in·expertFraction`（routed 路径按 `k/E` 缩放，
  `expertFractionFor` extractor.js:50-57）。实现：extractor case "linear"
  （`extractor.js:362-379`）+ `linearCounts`（`counts.js:16-28`）。
- **vector / sfu**：vector=`T·out`（bias；**主链 bias=false** 与旧链一致）、sfu=0。
- **bytes.weights**：`out·in·b`（权重读一遍；packed qweight 无 logical_weight_shape
  时返回 null → unknownComputePaths，本仓 59 模型实测 0 例）。
- **bytes.actIn / actOut**：actIn=`T·in·b`、actOut=`T·out·b`（读写各一次）。
- **单位与换算**：`aten::mm` 的 `m·n·2k` FLOPs → MACs 已换算（`index.js:29-30`）。
- **来源**：一等 aten::mm；A7：融合实现按语义分解计数（`index.js:29-31` ref 注释）。
- **全局假设**：A5（bias 逐 flop）、A7。
- **已知近似/登记**：
  - **同名双轨**：extractor case "linear" 手搓（含 embed 路径排除 → embedding
    gather 分支 `extractor.js:365-374`）+ registry `linear.counts` 死引用（§11）。
  - **embed 排除判据**：无 operatorId 但 weight_shapes ≥2 维、路径命中
    `/(^|\.)(patch_)?embed/` 的节点按 embedding gather 计（结构化路径判断，
    W3 TODO 换 attributes 标记，`extractor.js:353-359` 注释）。
  - bias=false 全局（bias 流量未计，与旧链一致——登记）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

---

## 5. 归一化与激活

### rmsnorm — RMSNorm

- **触发面**：57/59 模型（1898 节点/8633 实例）：input/post-attention norm 全仓通用。
  仅 Qwen3.8-2.4T×2 纯用 gemma_rmsnorm（无本叶）。
- **matrix**：精确零。
- **vector / sfu**：vector=`4·T·H_n`（x²、mean-reduce、×rsqrt、×w）；sfu=`T`
  （rsqrt=1，A5）。实现：`rmsnormCounts`（`counts.js:67-79`；
  `extractor.js:575-577`）。
- **bytes.weights**：`H_n·b`。
- **bytes.actIn / actOut**：actIn=`T·H_n·b`、actOut=`T·H_n·b`。
- **单位与换算**：H_n = input_shape 末维。
- **来源**：三等分解声明 mul/reduce/rsqrt/mul（无单一 aten 对应）（`index.js:117-118`）。
- **全局假设**：A5。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### gemma_rmsnorm — Gemma RMSNorm

- **触发面**：31/59 模型（2179 节点/4287 实例）：Qwen3.5/3.6/3.8（除 Flash-Next）
  与 MiniMax-M3 系的 q/k attention norm（Gemma 风格 checkpoint：缩放前 (1+w)）。
- **matrix**：精确零。
- **vector / sfu**：rmsnorm 基础上 **+ `T·H_n`**（(1+w) 加法）；sfu 同 rmsnorm。
  实现：`rmsnormCounts({weightOne:true})`（`counts.js:67-79`；
  `extractor.js:575-577` 按 operatorId 置 weightOne）。
- **bytes.weights / actIn / actOut**：与 rmsnorm 相同（(1+w) 是逐元素加法，不加流量）。
- **来源**：三等分解声明（Qwen3.5 Gemma 风格 checkpoint 语义）（`index.js:127-129`）。
- **全局假设**：A5。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### swiglu — SwiGLU

- **触发面**：59/59 模型（2161 节点/5774 实例）两个形态：
  - **dense MLP**（多数节点）：每 MLP 层一叶，vector/sfu/bytes 生效；
  - **routed FFN 压缩叶**（`expert_mlp`，MoE 专家主干）：matrix 段在此
    （探针 matrix>0 节点 926 个即 routed 形态），激活段 tokens×k。
- **matrix**：dense=精确零；routed=`T·k·3·EH·EI`（gate/up/down GEMM 融合语义，
  EH=expertHidden、EI=expertIntermediate）。实现：extractor case（`extractor.js:597-619`，
  ROUTED_EXPERT_RE 结构化路径判据）。
- **vector / sfu**：`vector=2·T_eff·I`（silu+mul）、`sfu=2·T_eff·I`（sigmoid=2 SFU）；
  dense 时 T_eff=T、routed 时 T_eff=T·k。
- **bytes.weights**：0（专家 GEMM 权重经参数量/权重字节链计；激活叶不重复计）。
- **bytes.actIn / actOut**：actIn=`2·T_eff·I·b`（gate/up 两路输出读）、
  actOut=`T_eff·I·b`。
- **单位与换算**：silu = x·sigmoid(x)：2 SFU + 1 mul（A5）。
- **来源**：一等 `aten::silu` + `aten::mul`（`index.js:136-137` ref 注释）。
- **全局假设**：A7（fused gate+up 按语义分解，融合收益记 implementation）。
- **已知近似/登记**：routed 计数公式 `T·k·3·EH·EI`（按 k 而非 k/E——旧链 ·(k/E)
  少乘 E 是已知双链 bug，2026-09-07 修正，`extractor.js:605-612` 注释）；
  per-expert 展开树若未来出现需回改并依赖 walker 乘 E。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### rope — RoPE

- **触发面**：59/59 模型（1101 节点/2393 实例）：每注意力层 q/k 旋转（含 MLA 的
  rope 分量、V4 的 rope/inverse_rope、partial rotary 因子）。
- **matrix**：精确零。
- **vector / sfu**：vector=`3·T·D_rope`（每维对 4 乘 2 加 = 3 flop/元素）；
  sfu=0（**A3：sin/cos 查表**）。实现：`ropeCounts`（`counts.js:108-115`）；
  extractor case（`extractor.js:571-574`，partial_rotary_factor 属性优先）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：各 `2·T·D_rope·b`（q+k 两路，读写各一次）。
- **单位与换算**：D_rope=headDim×partial_rotary_factor。
- **来源**：三等分解声明（无单一 aten 对应）（`index.js:78-79`）。
- **全局假设**：A3、A5。
- **已知近似/登记**：GLM-5.3-Flash 的 qk_rope_head_dim=0 → ropeDims=0，探针 row
  存在但分量为 0（精确零语义）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

---

## 6. MoE 路由与专家

### topk — TopK Routing

- **触发面**：44/59 模型（916 节点/2565 实例）：全部 MoE 模型的普通路由层
  （V4 的 hash 层除外——见 dsv4_hash_route；单节点×层数形态如 M2.7 1×62、
  K2 系 1×60，Qwen 系每 MoE 层显式节点）。
- **matrix**：精确零。
- **vector / sfu**：vector=`T·E`（比较/选择诚实计）、sfu=`T·k`（norm_topk_prob
  除法；normTopkProb=false 时 0）。实现：`topkCounts`（`counts.js:159-166`）；
  extractor case（`extractor.js:653-654`）。
- **bytes.weights**：0（router 权重由 linear 叶计）。
- **bytes.actIn / actOut**：actIn=`T·E·b`（router logits 读）、actOut=`T·k·b`
  （expert weights/ids 写）。
- **来源**：一等 `aten::topk`（flop_counter 不数比较选择——vector 按 T·E 诚实计）
  （`index.js:146-147` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### moe_dispatch — MoE Dispatch

- **触发面**：44/59 模型（926 节点/2580 实例），与 topk 同集合。
- **matrix / vector / sfu**：全精确零（gather 纯搬运）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`T·H·b`（hidden 读）、actOut=`T·k·H·b`（k 份专家
  输入写）。实现：`moeDispatchCounts`（`counts.js:168-173`；`extractor.js:655-656`）。
- **来源**：一等 `aten::index_select`（gather 纯搬运，零计算）（`index.js:156-157`）。
- **全局假设**：无特别引用。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### moe_combine — MoE Combine

- **触发面**：44/59 模型（926 节点/2580 实例），与 topk 同集合。
- **matrix**：精确零。
- **vector / sfu**：vector=`2·T·k·H`（乘 + 累加，2026-09-07 规格修正的诚实数学）。
  实现：`moeCombineCounts`（`counts.js:175-182`；`extractor.js:657-658`）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`(T·k·H + T·k)·b`（专家输出 + 权重读）、
  actOut=`T·H·b`（scatter + 加权合并写）。
- **来源**：三等分解声明 scatter + 加权合并（`index.js:165-167` ref 注释）。
- **全局假设**：A5（逐 flop）。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### moe_add — MoE Branch Add

- **触发面**：41/59 模型（873 节点/2422 实例）：有 shared expert / 双分支合并的
  MoE（DeepSeek 系、Kimi 系、GLM 系、Qwen3.5/3.8 MoE、M3）；MiniMax-M2.7 等
  无 shared 分支的 MoE 无此叶。
- **matrix**：精确零。
- **vector / sfu**：vector=`T·H`。实现：`addCounts`（`counts.js:184-191`；
  `extractor.js:659-660`）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`2·T·H·b`（routed+shared 两路读）、actOut=`T·H·b`。
- **来源**：一等 `aten::add`（`index.js:175-176`）。
- **全局假设**：无特别引用。
- **已知近似/登记**：cost_counts.md「结构级缺口」：decoder 层普通残差加法（+x）
  无算子节点，2TH·b×2/层未计——量级备注（60 层 ≈6MB/token，相对权重流量可忽略）
  是量化后的暂缓决定，不是遗漏。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### shared_expert_gate — Shared Expert Gate

- **触发面**：16/59 模型（426 节点/844 实例）：Qwen3.5-122B/35B-A3B/397B、
  Qwen3.6-35B、Qwen3.8-2.4T/Flash-Next（每 MoE 层一叶）。
- **matrix**：精确零。
- **vector / sfu**：vector=`T·W`、sfu=`2·T·W`（sigmoid）。实现：`gateCounts`
  （`counts.js:84-95`；`extractor.js:580-584`）。
- **bytes.weights**：0（W_g 融合在 shared expert MLP 或独立 linear 叶）。
- **bytes.actIn / actOut**：各 `T·W·b`。
- **来源**：二等 modeling 对照（Qwen3.5/3.6 MoE shared expert sigmoid gate；
  Qwen modeling 未入库——normalize.js sharedExpertGate 字段驱动）+
  A5（`index.js:343-346` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：探针 16 模型与 ref 注释「目录 16 模型发射此叶」一致。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### dsv4_hash_route — DeepSeek V4 Hash MoE Routing

- **触发面**：5/59 模型（10 节点/15 实例）：V4-Flash 系 2 节点×3（L0-1 组 + L2）、
  V4-Pro 系 2×3——前 num_hash_layers 层按 input_ids 查表路由，不走普通
  router logits + top-k（同模型其余 MoE 层仍走 topk 叶）。
- **matrix / vector / sfu**：全精确零（纯查表）。
- **bytes.weights**：`tableRows·b`，tableRows=`vocabSize·expertsPerToken`
  （V4-Flash：129280×6 ≈ 775,680 条目/层 ≈1.48MB——M11-P2 C 路接线，
  `extractor.js:661-670`；此前传 tableRows:0 低估，权重 index 实证
  ffn.gate.tid2eid）。
- **bytes.actIn / actOut**：actIn=`T·b`（token id 读）、actOut=`T·k·b`
  （expert ids/weights 写）。
- **来源**：二等 modeling 对照（DeepSeek V4 hash MoE：input_ids 查表固定专家集合，
  vLLM tid2eid；actIn/actOut 已核实生效——/tmp/m11-formulas/dsv4.md §(d)）。
- **全局假设**：无特别引用。
- **已知近似/登记**：tid2eid 存储 dtype 未证实（int32 索引按 bpe=2 计——dsv4.md
  §(d) 登记项）；dtype 裁决后 weights 口径或需调整。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

---

## 7. 多流残差与复合（mHC / Hyper-Connection / PLE）

> 本组为三等分解声明（A4/A7）：counts = 已知 F 函数组合，ctx 由
> extractor.js:681-748 的 11 个 ctxBuilder 构建；组合公式的人类可读展开如下。

### mhc_pre — mHC Pre

- **触发面**：7/59 模型（292 节点/341 实例）：DeepSeek-V4×5（42-60 节点/模型，
  每 decoder 层一叶）+ GLM-5.3-Flash×2（23×45）。vLLM MHCPreOp。
- **matrix**：`= T·H·n`（streams→input 合成小矩阵，n=mhcNumResidualStreams）。
  实现：ctxBuilder 组合 `F4(mix) + F1([H,n]) + add(merge)`
  （`index.js:226-235`；`extractor.js:730-734`）。
- **vector / sfu**：vector=`2·T·H`（sigmoid post mix + 合成加法）、sfu=`2·T·H`。
- **bytes.weights**：`H·n·b`。
- **bytes.actIn / actOut**：actIn=`4·T·H·b`（mix 读 + 矩阵输入 + add 双读）；
  actOut=`(2·T·H + T·n)·b`。
- **公式**：`p=sigmoid(M_a·s_a+b_a)+eps; C=Sinkhorn(softmax(M_c·s_c+b_c)+eps);
  x=Σ p_i·H_i`（softmax/Sinkhorn 的 SFU 细节并入 gate 段口径）。
- **来源**：三等分解声明（vLLM MHCPreOp implementation 指针，layers/hybrid.js）
  （`index.js:228-230` ref 注释）。
- **全局假设**：A4、A5、A7。
- **已知近似/登记**：Sinkhorn 迭代次数未建模（按 softmax+归一一段计）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### mhc_post — mHC Post

- **触发面**：7/59 模型（7 节点/7 实例）：每模型**末层** 1 节点（与
  mhc_fused_post_pre 互补：末层用 post+contract，中间层用 fused）。
- **matrix**：`= T·H·n`（combine 小矩阵）。实现：`F1(combine) + add(inject)`
  （`index.js:246-255`；`extractor.js:735-738`）。
- **vector / sfu**：vector=`T·H`、sfu=0（post mix 在 pre 侧计）。
- **bytes.weights**：`H·n·b`。
- **bytes.actIn / actOut**：actIn=`3·T·H·b`、actOut=`(T·H + T·n)·b`。
- **公式**：`H'_j = post_j·x + Σ C_ij·H_i`。
- **来源**：三等分解声明（vLLM MHCPostOp）（`index.js:248-249`）。
- **全局假设**：A4。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### mhc_fused_post_pre — mHC Fused Post + Pre

- **触发面**：7/59 模型（292 节点/341 实例）：与 mhc_pre 同集合同节奏——中间层
  每层一叶（上一层 post 与当前层 pre 的层间融合）。
- **matrix**：`= T·H·n`。实现：`F4(post) + add(inject) + F4(pre) + F1([H,n])`
  （`index.js:236-245`；`extractor.js:739-744`）。
- **vector / sfu**：vector=`3·T·H`（两次 gate + inject 加法）、sfu=`4·T·H`。
- **bytes.weights**：`H·n·b`。
- **bytes.actIn / actOut**：actIn=`5·T·H·b`；actOut=`(3·T·H + T·n)·b`。
- **公式**：`(H',post',C',x') = MHCPre(MHCPost(x,H,post,C); F,scale,base)`。
- **来源**：三等分解声明 = mhc_post + mhc_pre 融合（vLLM MHCFusedPostPreOp）；
  A7：融合收益记 attributes.implementation，不折算流量（`index.js:238-240`）。
- **全局假设**：A4、A7。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### mhc_contract — mHC Contract

- **触发面**：7/59 模型（7 节点/7 实例）：每模型末层 1 节点。
- **matrix**：精确零。
- **vector / sfu**：vector=`T·H`（n 流平均）、sfu=0。实现：`add(contract)`
  （`index.js:256-264`；`extractor.js:745-747`）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`2·T·H·b`（n 流读按二路一阶近似）、actOut=`T·H·b`。
- **公式**：`h = (1/n)·Σ H_i`。
- **来源**：三等分解声明（GLM-5.3-Flash 末层 HCContract 语义）（`index.js:258-259`）。
- **全局假设**：A4。
- **已知近似/登记**：n 流读按 addCounts 的 2 路口径近似（n>2 时读放大未折算——
  一阶口径声明）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### hyper_connection — Hyper Connection

- **触发面**：2/59 模型——Qwen/Qwen3.8-Flash-Next / -FP8（53 节点×97）。
- **matrix**：`= T·H²`（W_down/W_up 小矩阵 [H,H]）。实现：ctxBuilder 组合
  `F3(grouped) + F5(mix) + F1(mixers) + F4(gate) + add(combine)`
  （`index.js:320-330`；`extractor.js:716-722`）。
- **vector / sfu**：vector≈`8·T·H`（norm 4H + silu 2H + gate H + combine H）；
  sfu≈`4·T·H`（silu 2H + gate 2H）+ T（rsqrt）。
- **bytes.weights**：`(H² + H)·b`（mixer 矩阵 + norm weight）。
- **bytes.actIn / actOut**：actIn≈`7·T·H·b`；actOut≈`5·T·H·b`（各分量读写一次之和，
  以 sumCounts 实算为准）。
- **公式**：`x_n=GroupedRMSNorm(H); l=SiLU(W_down·x_n); gate=W_up·l;
  block_input=GateMix(x_n,gate); H'=Combine(H,block_output,injection)`。
- **来源**：三等分解声明（Qwen4Exp delayed HyperConnection，layers/hybrid.js）。
  **ref 注释滞后提醒**：注释写「Qwen modeling 未入库，离线取证」，但
  models/Qwen/Qwen3.8-Flash-Next/modeling_qwen4_exp.py 已于 2026-09-08 入库
  （evidence-qsa-glm.md §1）——来源可升二等，ref 注释待更新（只读发现，未改代码）。
- **全局假设**：A4、A5、A7。
- **已知近似/登记**：ctx 的 n 流参数用 normalized 近似（`extractor.js:673-674`
  「精度为初版，T4 恒等式校准后复核」）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### ple — Position Learning Enhancement

- **触发面**：2/59 模型——Qwen3.8-Flash-Next / -FP8（1 节点×1，指定层）。
- **matrix**：`= T·2·E_p·H + T·E_p·ngram`（W_kv 投影 + dilated short conv，
  E_p=pleEmbedDim）。实现：ctxBuilder 组合 `hashRoute(embed) + F1(kv) + F3(norm)
  + F7a(conv) + add`（`index.js:331-340`；`extractor.js:723-729`）。
- **vector / sfu**：`T·H`（add）+ `4·T·E_p`（norm）+ `T·E_p`（conv）；
  sfu=conv 段 `2·T·E_p` + norm 段 `T`。
- **bytes.weights**：`(2·E_p·H + E_p + E_p·ngram)·b`。
- **bytes.actIn / actOut**：hash 查表 actIn=`T·b`、actOut=`T·b`（tableRows=0：ngram
  表不计 weights）；其余分量读写一次之和。
- **公式**：`e=HashNGram(input_ids,context); [k,v]=W_kv·e;
  y=ShortConv(GatedNorm(k,v,RMSNorm(H)))`。
- **来源**：三等分解声明（Qwen4Exp PLE，layers/hybrid.js；同 hyper_connection 的
  ref 滞后提醒——modeling_qwen4_exp.py 已入库）。
- **全局假设**：A4、A5。
- **已知近似/登记**：ngram embedding 表（HashNGram 词表）无参数计费（tableRows=0
  ——表规模未建模，登记）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

---

## 8. 视觉族

### vision_position — Vision Position Embedding

- **触发面**：38/59 模型（38 节点/38 实例，每视觉塔 1 叶）：全部多模态模型
  （Qwen-VL 系、MiniMax-M3 系、Kimi-K2.5/2.6/2.7/K3、GLM-5.3-Flash×2、
  V4-Vision-Exp）。
- **matrix**：精确零。
- **vector / sfu**：vector=`T_v·H_v`（逐元素加法，T_v=visionTokens）。
  实现：`addCounts`（`counts.js:184-191`；`extractor.js:587-588`）。
- **bytes.weights**：0（位置编码本体不经权重字节链）。
- **bytes.actIn / actOut**：actIn=`2·T_v·H_v·b`（patch tokens + position 读）、
  actOut=`T_v·H_v·b`。
- **来源**：三等分解声明（逐元素加法；bytes 读 2 写 1 一阶约定）
  （`index.js:88-89` ref 注释）。
- **全局假设**：无特别引用。
- **已知近似/登记**：具体位置编码实现（learned 2D / rope 变体）由视觉塔配置决定，
  统一按加法计。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### vision_merge — Vision Patch Merge

- **触发面**：31/59 模型（31 节点/31 实例，每视觉塔 1 叶）：Qwen-VL 系 +
  GLM-5.3-Flash（merger.patch_merge）。MiniMax-M3/Kimi/V4-Exp 的视觉塔无此叶
  （结构不同）。
- **matrix**：精确零。
- **vector / sfu**：精确零（纯重排）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`inElements·b`、actOut=`outElements·b`
  （**permute 是真拷贝**——A1 豁免不适用于本叶）。实现：
  `rearrangeCounts({copy:true})`（`counts.js:215-221`；`extractor.js:589-590`）。
- **来源**：三等分解声明（rearrange copy=true）（`index.js:97-99` ref 注释）。
- **全局假设**：A1 的反面特例（真拷贝）。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### vision_activation — Vision Activation

- **触发面**：36/59 模型（65 节点/974 实例）：Qwen 系 2 节点/模型（视觉块 MLP 激活
  ×层组 + merger 激活）；MiniMax-M3×2、Kimi×4、V4-Vision-Exp 1 节点/模型（视觉块）。
- **matrix**：精确零。
- **vector / sfu**：vector=`2·T_v·I_v`、sfu=`2·T_v·I_v`。实现：`swigluCounts`
  （`counts.js:98-105`；`extractor.js:585-586`，intermediate=output_shape 末维）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`2·T_v·I_v·b`、actOut=`T_v·I_v·b`。
- **来源**：一等 `aten::gelu` / `aten::silu` 家族（φ 由视觉配置 hidden_act 决定）
  （`index.js:107-109` ref 注释）。
- **全局假设**：A5、F5 同构。
- **已知近似/登记**：gelu 的 erf/exp 差异未区分，sfu 统一按 silu 口径 2/元素计
  （登记）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

> 视觉塔的注意力与投影不经本组算子：patch_embed / 视觉 qkv / o_proj 走 linear 叶、
> 视觉 scores/context/softmax 走 matmul/softmax 叶（modality=vision 属性决定
> tokens=visionTokens，`extractor.js:40-42,318-326`）。

---

## 9. embedding 与输出头

### embedding（结构节点，非 registry）

- **触发面**：57/59 模型（57 节点/57 实例，每模型 1 个 `embed_tokens`，
  `type==='embedding'`、无 operator_id）。**例外**：MiniMax-M3 / M3-MXFP8 的结构树
  **无 embedding 节点**（builder 未发射，探针实测 0 节点）——两模型的查表流量当前
  不可见，属登记缺口（见「已知近似」）。
- **matrix**：精确零（gather 无 MACs）。
- **vector / sfu**：精确零。
- **bytes.weights**：0——嵌入表本体由参数量/权重字节链（nodeWeightBytes）计，
  本叶只计 gather 动作。
- **bytes.actIn / actOut**：actIn=`T·H·b`（每 token 读一行权重）、
  actOut=`T·H·b`（写一行 hidden）。实现：extractor `type==='embedding'` 分支
  （`extractor.js:343-351`）+ linear case 的 embed 路径分支（`extractor.js:365-374`，
  双入口同式，M11-P0-5）。
- **单位与换算**：H = output_shape 末维 || hiddenSize。
- **来源**：三等分解声明（embedding gather 一阶访存；M11-P0-5 补齐——此前被
  「非算子零向量」规则计为零，bytes 完整性棘轮实测抓出，`extractor.js:341-343`
  注释 + refactor_plan M11 已落地清单）。
- **全局假设**：无特别引用。
- **已知近似/登记**：
  - MiniMax-M3×2 无 embed 结构节点 → gather 流量缺失（**登记缺口**，修复面在
    builder 侧，非本文件范围）；
  - 视觉 patch_embed 是 linear 算子（GEMM 投影），不属本节；
  - tied embeddings（如 Qwen3.5 小杯）的权重共享去重由权重字节链处理，counts 不感知。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 输出头（lm_head）

输出头无独立 operator_id：`lm_head.linear` 是 operator_id=linear 的普通叶
（探针：每模型 1 节点，output_shape=[−1,−1,vocab]），计费完全走 §4 linear。
softmax 归一化交叉熵损失不计（工具口径：forward-only）。

---

## 10. view 零流量组（split 家族）

> **豁免理由（零是精确陈述）**：四叶均为 fused projection 输出的 strided view 拆分，
> 不发生数据拷贝、不使用任何计算/访存单元——A1（2026-09-07 拍板）。显式登记为零
> 而非漏算（principles §3.3「0 与 null 区分」）；bytes 完整性棘轮
> `bytesCompleteness.test.js:22` 的 `VIEW_OPS` 集合即本组显式豁免清单
> （新增豁免必须在该集合登记并写明理由）。
> 实现统一为 `rearrangeCounts()`（`counts.js:215-221` 无 copy 分支）+ extractor
> 共用 case（`extractor.js:591-596`）。

### split — Fused Projection Split

- **触发面**：36/59 模型（605 节点/726 实例）三个发射点：
  ① Qwen3.5/3.6/3.8 full-attention 层的 `qkv_gate_split`（fused QKV+gate 拆
  q/gate/k/v，`ops/index.js:277-282`，每 full 层 1 节点、乘数 1）；
  ② DeepSeek-V4×5 的 `qkv_split`（fused_wqa_wkv 拆 q_lora/kv latent，
  `ops/index.js:362-368`，42-60 节点×层数）；
  ③ MiniMax-M3×2 的 `qkv_index_split`（fused QKV+index 拆 main/index 分支，
  `ops/index.js:520-527`，2 节点×57+3）。
- **matrix / vector / sfu / bytes**：全精确零（A1 view 豁免）。
- **来源**：一等 `aten::split`（视图语义，flop_counter 无成本条目）（`index.js:59-60`）。
- **全局假设**：A1。
- **已知近似/登记**：split_sizes 由节点属性给出，不同模型分支数/宽度不同
  （`index.js:61` explanation）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### attention_qkv_split — Attention QKV Split

- **触发面**：40/59 模型（41 节点/1147 实例）：通用 GQA/full 模板与全部视觉塔的
  fused QKV 拆分——MiniMax-M2.7 文本（×62）、GLM-4.7 文本（2×89+3）、各视觉塔
  （M3 ×32、Qwen ×12-28、Kimi ×27、GLM-Flash ×24、V4-Exp ×32）。
- **matrix / vector / sfu / bytes**：全精确零（A1）。
- **来源**：二等 modeling 对照 models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py
  （fused QKV 拆 q/k/v 语义分支）（`index.js:388-390`）。
- **全局假设**：A1。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### qwen_qkvz_split — Qwen GDN QKVZ Split

- **触发面**：31/59 模型（383 节点/1137 实例）：Qwen3.5/3.6/3.8 全系的 GDN 层
  （fused qkvz 投影拆 q/k/v/z，每 KDA 层 1 节点，层组乘数 ×3 形态）。
- **matrix / vector / sfu / bytes**：全精确零（A1）。
- **来源**：二等 modeling 对照（vLLM.Qwen3NextAttention.qkv_proj /
  SGLang.Qwen3_5Attention.qkv_proj，ops/index.js:275）（`index.js:378-380`）。
- **全局假设**：A1。
- **已知近似/登记**：z 旁路到输出归一化（gated_rmsnorm），不进 short conv。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### mla_kv_split — MLA KV Latent Split

- **触发面**：19/59 模型（221 节点/1124 实例）：MLA 文本塔的 latent/rope 拆分
  （DeepSeek-R1/V3.1/V3.2 2×61、Kimi 系 2×61 / K3 23×24、GLM-5 系 2-38×78、
  GLM-Flash 11×11）。
- **matrix / vector / sfu / bytes**：全精确零（A1）。
- **来源**：一等 `aten::split`（`index.js:290-291`）。
- **全局假设**：A1（latent/rope 拆分零流量）。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

---

## 11. 双轨现状与护栏（§3.1b）

**现状**：42 条 registry 条目与 extractor 手搓分支**同名双轨**。运行时分派顺序：
`extractor.countsForNode` 的 type 分支（attention/embedding）→ 31 个手搓 switch
case（提前 return）→ default 走 registry `entry.counts(ctxBuilder())`（11 个
ctxBuilder）。**手搓 31 条的注册表 counts 引用是护栏认证过的死引用**——两轨对
同一 operatorId 可能口径不同（已发现实例：causal_conv1d 的 vector/sfu/weights、
matmul 的 F2 融合 vs 分解、swiglu 的 routed 分支）。

**护栏**：`scripts/check_principles.sh:65-86`（§3.1b 运行时接线判据）保证每条
FORMULAS 运行时可达——**手搓 case / ctxBuilder / 显式豁免三选一**，实测
42 条可达（手搓 31 / ctxBuilder 11 / 豁免 0）（refactor_plan.md:850）。配套：
§3.1d ref 注释 42/42；§3.1c bytes 完整性棘轮（全 leaf 三访存分量不得全零，
view 豁免除外）。长期方向 = 手搓分支逐条搬入 counts.js、消双轨
（refactor_plan.md「护栏 §3.1 改运行时判据」；M11.5 的共享 bytes 助手抽取）。

**对齐审查时的判读规则**：本文各节「实现」字段给的是**运行时真实生效**的位置；
registry 条目行号（index.js）是规格与 ref 来源的权威锚点。两者冲突时以运行时为准
登记差异（如 causal_conv1d）。

---

## 12. 自检清单

- [x] 42 条 registry 条目逐一覆盖：探针现身 40 条 + 零触发槽位 2 条
      （linear_attention、linear_attention_gate，§3.6）= 42。
- [x] 结构节点覆盖：embedding（§9）、attention 模块容器（§3.8）、
      split 家族 4 条（§10，VIEW_OPS 豁免登记）。
- [x] 探针实证：59/59 模型 prefill+decode 两相位跑通，unknown 叶 = 0；
      leaf 算子键 41 种，全部能在本文找到对应节。
- [x] 每个公式给出实现位置：counts.js F1-F9（15 个函数）+ extractor 31 case +
      11 ctxBuilder，均带 file:line。
- [x] 触发面数字全部来自 computeNodeCosts 探针（§1 方法），未用代码推断代替；
      与 ref 注释中的在案数字（如 shared_expert_gate 16 模型、qsa 16 模型、
      mla_output_gate 仅 K3）交叉核对一致。
- [x] 来源三级标注逐条引用 index.js 的 ref: 注释与证据 file:line；
      降级声明（dsv4 双 case、MiniMax-M3 transformers 库版、V3.2/V4 离线取证）
      显式标注；两处 ref 注释滞后（hyper_connection/ple 的「Qwen modeling 未入库」）
      已标注待更新。
