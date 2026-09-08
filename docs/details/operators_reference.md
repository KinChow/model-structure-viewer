# 算子对照参考（Operators Reference）— 结构槽位骨架版

> **用途**：逐算子人工对齐审查手册。**按结构槽位组织**（模型级 → 层内 → attention kind →
> MLP/MoE → 层间），每个算子一节，固定模板：触发面（trigger-map 实证）· matrix/vector/sfu/
> bytes 三分量 · 单位换算 · 基础分解 · 来源三级 · A 假设 · 已知近似 · 运行时 actions 数值
> 实证 · 对齐勾选。
> **口径权威链**：公式规格以 [`cost_counts.md`](./cost_counts.md) 为准；本文是它的逐算子
> 展开与实证附录，两者冲突时以 cost_counts.md + 代码注释为准并回改本文。
> **基线**：commit `a4d709a`（main），2026-09-09。全部 `file:line` 锚定该 commit
> （行号 + 符号锚双写）。触发面数字全部来自探针实跑（§1），非代码推断。
> **修正依据**：四份双向验证报告 `/tmp/m11-formulas/verify-{linear,attention,moe-norm,
> vision-misc}.md` 的修正小节**全部采纳**（关键数字已内联，防 /tmp 丢失）；C1（moe
> payload 宽度走 `staticWidth(node.input_shape)`，a4d709a `extractor.js:668-671`）与
> C2（dsv4_sparse_mla kvWrite=0，`extractor.js:453-455`）已落码，本文按 HEAD 口径书写。

---

## 全局裁决草案（G1–G3，**待用户确认**）

> 三条为跨算子口径裁决的草案化；确认后各算子节的「已知近似」随之改判。

- **G1 · gather/copy 一阶访存 = 按实际拷贝元素计**。
  符合：`embedding`（actIn/actOut = `T·H·b`，按真实 gather 行数）、`moe_dispatch`
  （`T·W·b` 读 / `T·k·W·b` 写）、`moe_combine`（scatter 同理）。
  **现违 1 处**：`vision_merge` 的 actIn/actOut 是单 token 宽（`inElements/outElements`，
  `extractor.js:602-603` case "vision_merge"），未乘 T_v——Qwen3.5-0.8B 实测 actIn=1536 B
  = 768·2（T_v=576，实际拷贝 768·576 元素，**少乘 576×**；2026-09-09 验证发现，
  verify-vision-misc 登记建议 b）。裁决通过 → `rearrangeCounts({copy:true})` 调用补乘
  tokens 并更新 golden。
- **G2 · registry/runtime 双轨差显式登记**。42 条 registry 条目与 extractor 手搓分支同名
  双轨；**运行时为准、registry 为规格锚**。已登记双轨差 5 处：
  ① `matmul`（registry=attentionCounts F2 融合口径 vs 运行时 scores/context 分解三叶，
  `index.js:37-46` 死引用）；② `causal_conv1d`（registry 含 SiLU vector/sfu+weights，
  运行时只计 matrix+actIn/actOut，`counts.js:118-125` vs `extractor.js:633-644`）；
  ③ `gated_delta_attention`（registry F7b delta vector=`2·T·state`/sfu=`3·Nh·T` vs 运行时
  stateUpdateCounts 精确零，`counts.js:140-152` vs `extractor.js:294-306`）；④
  `linear_attention`（registry plain 公式 vs 运行时 `/state|recurrent/` 分支实调
  stateUpdateCounts，`extractor.js:645-663`）；⑤ `swiglu`（routed GEMM 分支只在运行时）。
- **G3 · 融合 ≡ Σ子级 断言拟进护栏**。矩阵单元的精确恒等已核验（见 §4.3 各 kind 的
  一致性断言）：dense F2 = matmul(scores)+matmul(context)；indexer = F2+F8（sumCounts
  构造恒等）；MLA 压缩 = F1+F3。带登记差的两类：dense bytes 的 K/V 读宽（模块 kvH vs
  分解链 `Nh·D` repeat_kv 物化）与 kvWrite（cache 写归属 k/v_proj linear）；KDA 模块镜像
  把 norm 段折进 matrix（`3·vp·T`），子级以 vector/sfu 计。裁决通过 → 护栏测试固化
  「融合 matrix ≡ Σ子级 matrix；bytes 差额 == 登记项集合」。

---

## 0. 总览表

`matrix/vector/sfu` 列：✓ = 有计费；0 = 精确零（该单元无事可做，principles §3.3）；
0\* = 运行时精确零、registry 存在非零规格（G2 双轨差）。
`bytes` 列：✓ = 三访存分量至少 actIn/actOut 非零；∅ = 全零（A1 view 豁免）。
`来源`：一 = 一等 aten 锚点；二 = 二等 modeling 对照；三 = 三等分解声明（可组合如 `二+三`）。
`触发模型`：探针发射该 operator_id 叶的模型数 / 59；括号内 节点数 / 乘数后实例数。
`对齐`：勾选框供人工逐条核销。

| 算子 | matrix | vector | sfu | bytes | 公式摘要 | 来源 | 触发模型（节点/实例） | 对齐 |
|---|---|---|---|---|---|---|---|---|
| linear | ✓ | 0\* | 0 | ✓ | `T·out·in·xf`；w `out·in·b`；a `T·in·b`/`T·out·b` | 一 | 59/59（9891/28707） | - [ ] |
| matmul | ✓ | 0 | 0 | ✓ | `Nh·T·S·(D+dv)` 两叶 | 一 | 48/59（878/4162） | - [ ] |
| softmax | 0 | ✓ | ✓ | ✓ | `3E`/`2E`，E=`Nh·T·S` | 一 | 48/59（439/2081） | - [ ] |
| attention_output_gate | 0 | ✓ | ✓ | ✓ | `T·W`/`2·T·W` | 二 | 29/59（355/355） | - [ ] |
| attention_qkv_split | 0 | 0 | 0 | ∅ | view 零流量 | 二 | 40/59（41/1147） | - [ ] |
| mla_query_compress | ✓ | ✓ | ✓ | ✓ | F1(q_a)+F3 | 三+二 | 19/59（221/1124） | - [ ] |
| mla_kv_compress | ✓ | 0 | 0 | ✓ | `T·outW·H` | 三+二 | 24/59（464/1369） | - [ ] |
| mla_kv_split | 0 | 0 | 0 | ∅ | view 零流量 | 一 | 19/59（221/1124） | - [ ] |
| mla_output_gate | 0 | ✓ | ✓ | ✓ | `T·W`/`2·T·W` | 二 | 1/59（23/24） | - [ ] |
| qsa_indexer | ✓ | ✓ | ✓ | ✓ | F2+F8 | 二+三 | 16/59（327/698） | - [ ] |
| qsa_attention | ✓ | 0 | 0 | ✓ | `Nh·T·S_sel·(D+dv)` | 二 | 16/59（327/698） | - [ ] |
| minimax_sparse_indexer | ✓ | ✓ | ✓ | ✓ | F2+F8 | 二+三 | 2/59（2/114） | - [ ] |
| minimax_sparse_attention | ✓ | 0 | 0 | ✓ | `Nh·T·(blocks·bs)·(D+dv)` | 二 | 2/59（2/114） | - [ ] |
| dsv4_swa_attention | ✓ | 0 | 0 | ✓ | `Nh·T·W_win·(D+D)` | 二（降级） | 3/59（3/6） | - [ ] |
| dsv4_compressed_attention | ✓ | 0 | 0 | ✓ | `Nh·T·⌈S/ratio⌉·(D+D)` | 二（降级） | 5/59（120/122） | - [ ] |
| linear_attention | ✓ | ✓ | ✓ | ✓ | F7b plain（规格） | 二 | **0/59**（槽位保留） | - [ ] |
| linear_attention_gate | 0 | ✓ | ✓ | ✓ | `T·W`/`2·T·W` | 二 | **0/59**（槽位保留） | - [ ] |
| gated_delta_attention | ✓ | 0\* | 0\* | ✓ | `3·T·vh·dv·dk` | 二 | 34/59（431/1274） | - [ ] |
| gated_rmsnorm | 0 | ✓ | ✓ | ✓ | `5·T·H_n`/`T+2·T·H_n` | 二+三 | 34/59（431/1274） | - [ ] |
| causal_conv1d | ✓ | 0\* | 0\* | ✓（weights=0\*） | `T·width·kernel` | 一 | 34/59（431/1274） | - [ ] |
| attention_residual | ✓ | ✓ | ✓ | ✓ | F3×2+F1+F2+add | 二+三 | 1/59（47/93） | - [ ] |
| split | 0 | 0 | 0 | ∅ | view 零流量 | 一 | 36/59（605/726） | - [ ] |
| qwen_qkvz_split | 0 | 0 | 0 | ∅ | view 零流量 | 二 | 31/59（383/1137） | - [ ] |
| rmsnorm | 0 | ✓ | ✓ | ✓ | `4·T·H_n`/`T` | 三 | 57/59（1898/8633） | - [ ] |
| gemma_rmsnorm | 0 | ✓ | ✓ | ✓ | `5·T·H_n`/`T` | 三 | 31/59（2179/4287） | - [ ] |
| swiglu | 0（routed ✓） | ✓ | ✓ | ✓ | `2·T_eff·I`；routed `T·k·3·EH·EI` | 一 | 59/59（2161/5774） | - [ ] |
| rope | 0 | ✓ | 0 | ✓ | `3·T·D_rope` | 三 | 59/59（1101/2393） | - [ ] |
| topk | 0 | ✓ | ✓ | ✓ | `T·E`/`T·k` | 一 | 44/59（916/2565） | - [ ] |
| moe_dispatch | 0 | 0 | 0 | ✓ | gather `T·W·b`→`T·k·W·b` | 一 | 44/59（926/2580） | - [ ] |
| moe_combine | 0 | ✓ | 0 | ✓ | `2·T·k·W` | 三 | 44/59（926/2580） | - [ ] |
| moe_add | 0 | ✓ | 0 | ✓ | `T·H` | 一 | 41/59（873/2422） | - [ ] |
| shared_expert_gate | 0 | ✓ | ✓ | ✓ | `T·W`/`2·T·W` | 二 | 16/59（426/844） | - [ ] |
| dsv4_hash_route | 0 | 0 | 0 | ✓ | 查表 w `tableRows·b` | 二 | 5/59（10/15） | - [ ] |
| mhc_pre | ✓ | ✓ | ✓ | ✓ | `T·H·n` | 三 | 7/59（292/341） | - [ ] |
| mhc_post | ✓ | ✓ | 0 | ✓ | `T·H·n` | 三 | 7/59（7/7） | - [ ] |
| mhc_fused_post_pre | ✓ | ✓ | ✓ | ✓ | `T·H·n` | 三 | 7/59（292/341） | - [ ] |
| mhc_contract | 0 | ✓ | 0 | ✓ | `T·H` | 三 | 7/59（7/7） | - [ ] |
| hyper_connection | ✓ | ✓ | ✓ | ✓ | `T·H²` | 三 | 2/59（106/194） | - [ ] |
| ple | ✓ | ✓ | ✓ | ✓ | F1+F3+F7a+hash | 三 | 2/59（2/2） | - [ ] |
| vision_position | 0 | ✓ | 0 | ✓ | `T_v·H_v` | 三 | 38/59（38/38） | - [ ] |
| vision_merge | 0 | 0 | 0 | ✓（G1 缺口\*\*） | copy 单 token 宽 | 三 | 31/59（31/31） | - [ ] |
| vision_activation | 0 | ✓ | ✓ | ✓ | `2·T_v·I_v` | 一 | 36/59（65/974） | - [ ] |
| — embedding(struct) 结构节点 | 0 | 0 | 0 | ✓ | gather `T·H·b` | 三（M11-P0-5） | 57/59（57/57） | - [ ] |
| — attention 模块容器（type=attention） | 计费 0 | 计费 0 | 计费 0 | 计费 0 | 容器不计费 | — | 59/59（容器） | - [ ] |

\* linear vector：主链精确零（`T·out` 是 bias=true 的条件容量，主链不传 bias）；
gated_delta_attention / causal_conv1d 的 vector/sfu（与 causal_conv1d weights）：运行时
精确零，registry 存在非零规格（G2 双轨差登记，见各节）。
\*\* vision_merge：copy 一阶访存按实际拷贝元素应乘 T_v（G1 草案违反项，已登记）。

探针口径结论：59 模型共发现 **41 种 leaf 键** = 40 条 registry operatorId（42 条中
`linear_attention`、`linear_attention_gate` 为零触发通用槽位）+ 1 个结构节点
`embedding`。prefill（T=128）与 decode（T=1、S=4096）两相位 unknown 叶均为 **0**。
2026-09-09 在 a4d709a 复跑触发探针（`/tmp/m11-formulas/probe-rewrite.mjs`）与冻结的
`trigger-map.json` **双向 0 差异**（41 键 × 59 模型）。

---

## 1. 探针方法与口径（可复现）

```bash
node --input-type=module /tmp/m11-formulas/probe-verify.mjs   # 触发面（节点/实例/路径）
#   → /tmp/m11-formulas/probe-verify-out.json（59 模型全量）
node --input-type=module /tmp/m11-formulas/probe-rewrite.mjs  # 触发面复核 + A100 五路时间
#   → /tmp/m11-formulas/probe-rewrite-out.json（本文 §5 占比数据源）
```

管线（与探针脚本同源）：`models/catalog.json` 全部 59 模型：config → `normalizeConfig`
→ `resolveArchitecture` → `buildNetwork` → `createStructureIr` → `materializeModelStructure`
→ `computeNodeCosts(root, normalized, { batch:1, sequence:128, phase:'prefill' })`
（decode 相位 `sequence:4096`），逐叶收集 `attributes.operator_id`
（`type==='embedding'` 记为结构节点 `embedding`）。

统计口径：

- **leaf** = 无 children 的节点；父节点（模块/容器）不携带动作向量
  （`frontend/src/cost/compute.js:43-48`，aggregate 链由子节点累加）。
- **节点数 vs 实例数**：节点数 = 结构树中 leaf 出现次数；实例数 = Σ multiplier
  （层组 repeat 倍乘，`frontend/src/cost/traverse.js:40-55`）。下文记作「节点/实例」。
- **相位**：prefill（T=seq）与 decode（T=1、S=上下文全长）均跑过，unknown 叶皆 0
  （与 `frontend/src/structure/__tests__/builtinModels.test.js:54-55` 的 `computeComplete`
  断言一致）。
- 数值快照存档：`/tmp/m11-formulas/probe-verify-out.json`、`probe-rewrite-out.json`
  （临时文件，本文已内联全部所需数字）。

---

## 2. 记号、单位与全局假设速查

- 记号：`T`=tokens（phase 决定）、`S`=可见 key tokens、`H`=hidden、`Nh`=query 头数、
  `kvH`=KV 头数、`D`=head_dim、`dv`=value head_dim、`I`=intermediate、`E`=专家数、
  `k`=topk、`n`=流数、`b`=每元素字节（主链固定 `bpe=2`，`compute.js:23`）、
  `xf`=expertFraction（routed 路径 = k/E）。
- 单位（principles §3.1）：`matrix` 存 **MACs**（aten FLOPs 公式含 2×，抄时换算）、
  `vector` 存 flop、`sfu` 存操作次数、`bytes` 为每次前向 compulsory traffic
  （权重读一遍 + 输入读 + 输出写，读写各一次口径，**无 phase 分支**）。
- 全局假设 A1–A7 全文见 [`cost_counts.md`](./cost_counts.md)「全局假设」表：
  A1 split/view 零流量；A2 softmax 融合单遍；A3 rope sin/cos 查表；
  A4 复合分解逐条标注；A5 SFU 口径（sigmoid=2、exp=1、rsqrt=1、div=1）；
  A6 线性注意力 per-token 递推下界；A7 融合算子按语义分解、流量不折算。
- 九个共享 counts 实现 F1–F9 全部在 `frontend/src/structure/formulas/counts.js`
  （下称 counts.js），本文每节「实现」字段给出 file:line + 符号锚。
- **基础分解词汇表**（§4 每算子「基础分解」字段使用）：`matmul` / `softmax` /
  `rmsnorm` / `elementwise`（含 gate·mul、silu、add）/ `rope` / `gather` / `copy` /
  `conv` / `add`。某算子的 counts **不能**由该词汇组合复现时，双轨差显式登记
  （当前登记：`gated_delta_attention` 的递推状态访存、`topk`/`dsv4_hash_route` 的
  选择/查表原语、`mhc_*` 的 Sinkhorn/softmax-on-streams 段并入 gate 口径）。

---

## 3. 双向表（互为反查；锚链接互引）

### 表 A · 算子 → 结构槽位

> 槽位锚 = 表 B 行 id。`kind` 列：该算子出现在哪种注意力形态的子块（或 `—`）。

| 算子 | 结构槽位（表 B 锚） | attention kind | 触发模型数 |
|---|---|---|---|
| embedding | [B-model-embed](#b-model-embed) | — | 57/59 |
| linear | [B-model-final](#b-model-final)（lm_head）· [B-model-vision](#b-model-vision)（patch_embed）· [B-layer-attn](#b-layer-attn)（各 kind 的 qkv/o/压缩/indexer/ba/decay 投影）· [B-layer-mlp](#b-layer-mlp) · [B-layer-moe](#b-layer-moe)（shared experts） | 全部 | 59/59 |
| matmul | [B-layer-attn](#b-layer-attn) → [B-attn-dense](#b-attn-dense) · [B-model-vision](#b-model-vision) | dense + 视觉塔 | 48/59 |
| softmax | 同 matmul | 同上 | 48/59 |
| attention_qkv_split | [B-layer-attn](#b-layer-attn) → [B-attn-dense](#b-attn-dense) · [B-model-vision](#b-model-vision) | dense（通用模板）+ 视觉塔 | 40/59 |
| split | [B-layer-attn](#b-layer-attn) → [B-attn-dense](#b-attn-dense)（qwen35_full）· [B-attn-dsv4](#b-attn-dsv4) · [B-attn-minimax](#b-attn-minimax) | dense(qwen35_full)/dsv4/minimax | 36/59 |
| rope | [B-layer-attn](#b-layer-attn)（各 kind 的 q/k 旋转、V4 rope/inverse_rope） | 全部 | 59/59 |
| rmsnorm | [B-model-final](#b-model-final)（final norm）· [B-layer-pre](#b-layer-pre) / [B-layer-post](#b-layer-post) / [B-layer-attn](#b-layer-attn)（MLA q_a 后 norm、V4 q_norm） | 全部 | 57/59 |
| gemma_rmsnorm | [B-layer-attn](#b-layer-attn) → [B-attn-dense](#b-attn-dense)/[B-attn-minimax](#b-attn-minimax)（q/k attention norm）+ 2.4T 全槽 norm | dense/minimax | 31/59 |
| mla_query_compress | [B-layer-attn](#b-layer-attn) → [B-attn-mla](#b-attn-mla) · [B-attn-dsa](#b-attn-dsa) | mla/dsa | 19/59 |
| mla_kv_compress | 同上 + [B-attn-dsv4](#b-attn-dsv4)（compressor） | mla/dsa/dsv4 | 24/59 |
| mla_kv_split | [B-layer-attn](#b-layer-attn) → [B-attn-mla](#b-attn-mla) · [B-attn-dsa](#b-attn-dsa) | mla/dsa | 19/59 |
| mla_output_gate | [B-layer-attn](#b-layer-attn) → [B-attn-mla](#b-attn-mla) | mla | 1/59 |
| qsa_indexer | [B-layer-attn](#b-layer-attn) → [B-attn-dsa](#b-attn-dsa) · [B-attn-qsa](#b-attn-qsa) · [B-attn-dsv4](#b-attn-dsv4) | dsa/qsa/dsv4 | 16/59 |
| qsa_attention | 同上 | 同上 | 16/59 |
| minimax_sparse_indexer | [B-layer-attn](#b-layer-attn) → [B-attn-minimax](#b-attn-minimax) | minimax | 2/59 |
| minimax_sparse_attention | 同上 | minimax | 2/59 |
| dsv4_swa_attention | [B-layer-attn](#b-layer-attn) → [B-attn-dsv4](#b-attn-dsv4) | dsv4（ratio=0） | 3/59 |
| dsv4_compressed_attention | 同上 | dsv4（ratio=128） | 5/59 |
| linear_attention | [B-layer-attn](#b-layer-attn) → [B-attn-kda](#b-attn-kda)（generic plain 槽位） | linear | **0/59** |
| linear_attention_gate | 同上（输出门槽位） | linear | **0/59** |
| gated_delta_attention | [B-layer-attn](#b-layer-attn) → [B-attn-kda](#b-attn-kda) | kda | 34/59 |
| gated_rmsnorm | 同上（KDA 输出归一化） | kda | 34/59 |
| causal_conv1d | 同上（q/k/v 短卷积） | kda | 34/59 |
| attention_residual | [B-layer-res](#b-layer-res) | kda（K3 hybrid 层） | 1/59 |
| qwen_qkvz_split | [B-layer-attn](#b-layer-attn) → [B-attn-kda](#b-attn-kda) | kda | 31/59 |
| attention_output_gate | [B-layer-attn](#b-layer-attn) → [B-attn-dense](#b-attn-dense) | dense（qwen35_full） | 29/59 |
| swiglu | [B-layer-mlp](#b-layer-mlp)（dense）· [B-layer-moe](#b-layer-moe)（routed expert_mlp 压缩叶） | — | 59/59 |
| topk | [B-layer-moe](#b-layer-moe) | — | 44/59 |
| moe_dispatch | 同上 | — | 44/59 |
| moe_combine | 同上 | — | 44/59 |
| moe_add | 同上（双分支合并） | — | 41/59 |
| shared_expert_gate | 同上（shared 分支门） | — | 16/59 |
| dsv4_hash_route | [B-layer-moe](#b-layer-moe)（V4 hash 层替代 topk） | — | 5/59 |
| mhc_pre / mhc_fused_post_pre | [B-layer-streams](#b-layer-streams) | — | 7/59 |
| mhc_post / mhc_contract | 同上（末层） | — | 7/59 |
| hyper_connection | [B-layer-streams](#b-layer-streams) | — | 2/59 |
| ple | 同上（指定层） | — | 2/59 |
| vision_position | [B-model-vision](#b-model-vision) | — | 38/59 |
| vision_merge | 同上（Qwen-VL 系 + GLM-Flash merger） | — | 31/59 |
| vision_activation | 同上（块 MLP + merger） | — | 36/59 |
| — attention 容器 | [B-layer-attn](#b-layer-attn)（父容器，恒有子叶） | 全部 8 kind | 59/59 |

### 表 B · 结构槽位 → 算子序列（数据流顺序）

> 列：槽位（含锚）→ 算子组成序列 → 融合展开标记 → 缺口标记。

**模型级槽位**

| 锚 | 槽位 | 算子序列（数据流顺序） | 融合/展开 | 缺口 |
|---|---|---|---|---|
| <a id="b-model-embed"></a>B-model-embed | embedding 层 | `embedding`(gather) | — | MiniMax-M3×2 **无 embed 结构节点**（builder 未发射，查表流量缺失——登记缺口，修复面在 builder 侧） |
| B-model-stack | 层栈 | N × 层（层内槽位 repeat；层组 multiplier 见 §1） | 层组 repeat 乘 | — |
| <a id="b-model-final"></a>B-model-final | final norm + 头 | `rmsnorm`(final) → `linear`(lm_head，out=vocab) | lm_head 无独立 operator_id | 损失交叉熵 softmax 不计（forward-only 口径） |
| <a id="b-model-vision"></a>B-model-vision | 视觉塔 | `linear`(patch_embed，GEMM) → `vision_position` → 块×L[ `attention_qkv_split` → `rope` → `matmul`(scores) → `softmax` → `matmul`(context) → `linear`(o) → MLP: `linear`→`vision_activation`→`linear` ] → `vision_merge` → `vision_activation`(merger)（后两叶仅 Qwen-VL 系 + GLM-Flash） | 视觉注意力走 dense 分解链（modality=vision，tokens=T_v） | M3/Kimi/V4-Exp 视觉塔无 vision_merge（结构不同） |

**层内槽位**

| 锚 | 槽位 | 算子序列 | 融合/展开 | 缺口 |
|---|---|---|---|---|
| <a id="b-layer-pre"></a>B-layer-pre | pre-norm | `rmsnorm` 或 `gemma_rmsnorm`（Qwen3.8-2.4T×2 纯 gemma） | — | — |
| <a id="b-layer-attn"></a>B-layer-attn | attention 子块 | 按 kind 分派（下列 7 行） | 见各行 | 容器 `type=attention` 不计费 |
| <a id="b-attn-dense"></a>B-attn-dense | attention·dense（通用 GQA/full 模板、qwen35_full） | `linear`(qkv fused) → `attention_qkv_split`（通用/视觉）或 `split`(qkv_gate_split，qwen35_full) → `rope` → `matmul`(scores) → `softmax` → `matmul`(context) → `attention_output_gate`(仅 qwen35_full) → `linear`(o) | 模块级融合公式 = F2（`attentionCounts`，`index.js:37-46`）；一致性断言见 §4.3.1 | 普通残差加（+x）无算子位 |
| <a id="b-attn-mla"></a>B-attn-mla | attention·MLA（R1/V3.1/K2 系/K3/GLM-5 系） | q 路：`linear`(q_a) → `mla_query_compress`(F1+F3) → `linear`(q_b) → `rope`；kv 路：`mla_kv_compress`(kv_a，latent cache 写) → `mla_kv_split` → `rope`；核心：`matmul`×2 + `softmax` → `mla_output_gate`(仅 K3) → `linear`(o) | 压缩叶=融合声明（q_a+norm）；kv_b 不在叶内（防双计） | — |
| <a id="b-attn-dsa"></a>B-attn-dsa | attention·DSA（V3.2、GLM-5 系×8） | MLA 压缩链（同上） + `linear`(indexer q/k) → `qsa_indexer` → `qsa_attention`(kvH=1，latent 读宽) → `linear`(o) | indexer=F2+F8 融合声明 | — |
| <a id="b-attn-qsa"></a>B-attn-qsa | attention·QSA（Qwen3.8-Flash-Next×2） | `linear`(qkv) → `rope` → `linear`(indexer) → `qsa_indexer` → `qsa_attention`(逐头 GQA/MHA，计 kvWrite) → `linear`(o)；同层伴 `hyper_connection`/`ple` | 同上 | — |
| <a id="b-attn-minimax"></a>B-attn-minimax | attention·M3 块稀疏（M3×2） | `linear`(qkv_index fused) → `split`(qkv_index_split) → `gemma_rmsnorm`(q/k norm) → `linear`(index q/k) → `minimax_sparse_indexer` → `minimax_sparse_attention`(含 kvWrite) → `linear`(o) | indexer=F2+F8 | — |
| <a id="b-attn-kda"></a>B-attn-kda | attention·KDA（Qwen3.5/3.6/3.8、K3、GLM-Flash） | `linear`(qkvz fused) → `qwen_qkvz_split` → `causal_conv1d`(q/k/v 短卷积) → `linear`(ba/decay 投影) → `gated_delta_attention` → `gated_rmsnorm` → `linear`(o)；z 旁路 gated_rmsnorm 不进 conv | 模块级镜像=per-mode legacy 公式（`extractor.js:203-271`）；一致性断言见 §4.3.5 | 递推状态访存非基础词汇可表达（登记） |
| <a id="b-attn-dsv4"></a>B-attn-dsv4 | attention·V4（V4×5） | `mhc_pre` → L0-1 组：`dsv4_swa_attention`；C4 压缩层：`mla_kv_compress`(compressor) → `split`(qkv_split) → `rope`/inverse → `linear`(indexer) → `qsa_indexer` → `qsa_attention`(dsv4_sparse_mla) → `linear`(o)；层尾 `dsv4_hash_route`(前 3 层) 或 `topk` → MoE 链 | c128a=压缩历史+原始滑窗混合读（fc99269） | — |
| <a id="b-layer-res"></a>B-layer-res | 残差槽位 | `attention_residual`（仅 K3，attn 前+MLP 前） | F3×2+F1+F2+add | **普通残差加（+x）无算子节点**：2TH·b×2/层未计（60 层 ≈6MB/token，量化暂缓，见 `moe_add` 节登记） |
| <a id="b-layer-post"></a>B-layer-post | post-norm | `rmsnorm` / `gemma_rmsnorm` | — | — |
| <a id="b-layer-mlp"></a>B-layer-mlp | MLP 子块 | `linear`(gate/up fused) → `swiglu` → `linear`(down) | gate/up 融合按语义分解（A7） | 残差加同上 |
| <a id="b-layer-moe"></a>B-layer-moe | MoE 子块 | `topk` → `moe_dispatch` → `swiglu`(routed expert_mlp 压缩叶) → `moe_combine` → [`moe_add` + `shared_expert_gate`]（有 shared 分支的模型）；V4 前 3 层 `dsv4_hash_route` 替代 `topk` | routed GEMM 记 swiglu matrix 段 | Flash-Next×2 有 gate 无 shared_experts 分支（normalize 接线缺口，见 `moe_add`） |
| <a id="b-layer-streams"></a>B-layer-streams | 多流残差（层间槽位） | V4/GLM-Flash：`mhc_pre` → …层… → `mhc_fused_post_pre`（中间层）→ 末层 `mhc_post`+`mhc_contract`；Flash-Next：`hyper_connection`(每层 attn/mlp 两叶) + `ple`(指定层) | 融合叶 = post+pre 层间融合（A7） | snapshot bank 存储流量不计（`attention_residual`） |

**双向自检**：表 A 每行槽位锚在表 B 存在、表 B 每行算子在表 A 回指——见 §7 自检清单
「双向表交叉完整性」。

---

## 4. 结构槽位骨架正文（逐算子）

> 每算子固定模板：**触发面**（trigger-map 实证，引用 = 模型数/59（节点/实例））·
> **matrix / vector / sfu / bytes 三分量** · **单位与换算** · **基础分解** · **实现** ·
> **来源三级** · **全局假设** · **已知近似/登记** · **运行时 actions 实证** ·
> **对齐状态**。行号锚定 commit `a4d709a`。

### 4.1 模型级槽位

#### embedding（结构节点，非 registry）— [B-model-embed](#b-model-embed)

- **触发面**：57/59 模型（57 节点/57 实例，每模型 1 个 `embed_tokens`，
  `type==='embedding'`、无 operator_id）。**例外**：MiniMax-M3 / M3-MXFP8 的结构树
  **无 embedding 节点**（builder 未发射，探针实测 0 节点；2026-09-09 复核维持：全树
  「一维 >100,000 的 weight_shapes + type=embedding」双条件扫描 0 命中，即无任何显式
  vocab 维权重形状；M3 的 lm_head 不受影响——root.3.0 `[-1,-1,200064]` vocab 宽 linear
  叶正常计费，M2.7 有 embedding 叶）——两模型的查表流量当前不可见，属登记缺口
  （修复面在 builder 侧）。
- **matrix / vector / sfu**：全精确零（gather 无 MACs）。
- **bytes.weights**：0——嵌入表本体由参数量/权重字节链（nodeWeightBytes）计，本叶只计
  gather 动作。
- **bytes.actIn / actOut**：actIn=`T·H·b`（每 token 读一行权重）、actOut=`T·H·b`
  （写一行 hidden）。实现：extractor `type==='embedding'` 分支
  （`extractor.js:343-351`，`countsForNode` embedding 分支）+ linear case 的 embed 路径
  分支（`extractor.js:365-374`，双入口同式，M11-P0-5；**第二入口为防御性保留、现网
  不可达**——结构 id 为位置式 `root.x.y`，`/(^|\.)(patch_)?embed/` 正则恒不命中；
  patch_embed 实测 op=linear + 位置 id，正确走 GEMM）。
- **单位与换算**：H = output_shape 末维 || hiddenSize。
- **基础分解**：`gather`（一阶访存按实际拷贝元素计，G1 符合项）。
- **来源**：三等分解声明（embedding gather 一阶访存；M11-P0-5 补齐——此前被
  「非算子零向量」规则计为零，bytes 完整性棘轮实测抓出，`extractor.js:340-343`
  注释 + refactor_plan M11 已落地清单）。
- **全局假设**：无特别引用。
- **已知近似/登记**：MiniMax-M3×2 无 embed 结构节点（上）；tied embeddings（如
  Qwen3.5 小杯）的权重共享去重由权重字节链处理，counts 不感知。
- **运行时 actions 实证**：Qwen3.5-0.8B（T=128、H=1024）actIn=actOut=262,144 B
  = 128·1024·2，逐位吻合（verify-vision-misc §3.3）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### 输出头（lm_head）— [B-model-final](#b-model-final)

输出头无独立 operator_id：`lm_head.linear` 是 operator_id=linear 的普通叶（探针：每模型
1 节点，output_shape=`[−1,−1,vocab]`，59/59 模型实测恰 1 个），计费完全走 §4.2.2
`linear`。softmax 归一化交叉熵损失不计（工具口径：forward-only，探针无任何 loss 相关节点）。
样例：Qwen3.5-0.8B root.4.0 `[-1,-1,248320]`、R1 root.3.0 `[-1,-1,129280]`、
M3 root.3.0 `[-1,-1,200064]`、K3 root.6.0 `[-1,-1,163840]`、GLM-Flash root.4.0
`[-1,-1,154880]`。

#### 视觉塔三算子 — [B-model-vision](#b-model-vision)

> 视觉塔的注意力与投影不经专用算子：patch_embed / 视觉 qkv / o_proj 走 `linear` 叶、
> 视觉 scores/context/softmax 走 `matmul`/`softmax` 叶（modality=vision 属性决定
> tokens=visionTokens，`extractor.js:40-42,318-326`，`tokensFor`）。

##### vision_position — Vision Position Embedding

- **触发面**：38/59 模型（38 节点/38 实例，每视觉塔 1 叶）：全部多模态模型
  （Qwen-VL 系、MiniMax-M3 系、Kimi-K2.5/2.6/2.7/K3、GLM-5.3-Flash×2、V4-Vision-Exp；
  Qwen3.8-2.4T×2 无视觉塔不在集合）。
- **matrix**：精确零。**vector / sfu**：vector=`T_v·H_v`（逐元素加法）、sfu=0。
  实现：`addCounts`（`counts.js:184-191`，`addCounts`）+ case
  （`extractor.js:600-601`，case "vision_position"）。
- **bytes.weights**：0（位置编码本体不经权重字节链）。
- **bytes.actIn / actOut**：actIn=`2·T_v·H_v·b`（patch tokens + position 读）、
  actOut=`T_v·H_v·b`。
- **单位与换算**：T_v=visionTokens；bpe=2。
- **基础分解**：`add`。
- **来源**：三等分解声明（逐元素加法；bytes 读 2 写 1 一阶约定）
  （`index.js:86-94`，FORMULAS.vision_position ref 注释）。
- **全局假设**：无特别引用。
- **已知近似/登记**：具体位置编码实现（learned 2D / rope 变体）由视觉塔配置决定，
  统一按加法计。
- **运行时 actions 实证**：Qwen3.5-0.8B（T_v=576、H_v=768）：vector=442,368=576·768、
  actIn=1,769,472=2·576·768·2、actOut=884,736，逐位吻合（verify-vision-misc §3.1）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### vision_merge — Vision Patch Merge

- **触发面**：31/59 模型（31 节点/31 实例，每视觉塔 1 叶）：Qwen-VL 系 +
  GLM-5.3-Flash（merger.patch_merge）。MiniMax-M3/Kimi/V4-Exp 的视觉塔无此叶
  （结构不同）。
- **matrix / vector / sfu**：全精确零（纯重排）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`inElements·b`、actOut=`outElements·b`
  （permute 是真拷贝——A1 豁免不适用于本叶）。实现：
  `rearrangeCounts({copy:true})`（`counts.js:215-221`，`rearrangeCounts`）+ case
  （`extractor.js:602-603`，case "vision_merge"）。
- **单位与换算**：in/out elements 为**单 token 宽度**（Qwen3.5-0.8B in=768、out=3072）。
- **基础分解**：`copy`——**G1 草案违反项**：一阶访存应按实际拷贝元素计（×T_v），
  现按单 token 宽计（Qwen3.5-0.8B T_v=576 → 少乘 576×；doc=code 一致、无对外矛盾，
  已登记待裁决）。
- **来源**：三等分解声明（rearrange copy=true）（`index.js:95-104`，FORMULAS.vision_merge）。
- **全局假设**：A1 的反面特例（真拷贝）+ G1（草案）。
- **已知近似/登记**：未乘 T_v（上）；裁决通过后修 `extractor.js:602-603` 调用并更新
  golden。
- **运行时 actions 实证**：Qwen3.5-0.8B actIn=1536=768·2、actOut=6144=3072·2
  （verify-vision-misc §3.2）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### vision_activation — Vision Activation

- **触发面**：36/59 模型（65 节点/974 实例）：Qwen 系 2 节点/模型（视觉块 MLP 激活
  ×层组 + merger 激活）；MiniMax-M3×2、Kimi×4、V4-Vision-Exp 1 节点/模型（视觉块）。
- **matrix**：精确零。
- **vector / sfu**：vector=`2·T_v·I_v`、sfu=`2·T_v·I_v`。实现：`swigluCounts`
  （`counts.js:98-105`，`swigluCounts`）+ case（`extractor.js:598-599`，
  case "vision_activation"，intermediate=`staticWidth(output_shape)`）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`2·T_v·I_v·b`、actOut=`T_v·I_v·b`。
- **单位与换算**：I_v 取 `staticWidth`（正有限维之积）；视觉激活叶 output_shape 为
  `[-1,-1,I]`（动态 -1 维剔除后恰为末维，vision.js:29,76,124——若未来出现全正维
  output_shape 语义将分叉，登记）。
- **基础分解**：`elementwise`（φ：silu/gelu 家族，2 SFU + mul）。
- **来源**：一等 `aten::gelu` / `aten::silu` 家族（φ 由视觉配置 hidden_act 决定）
  （`index.js:105-114`，FORMULAS.vision_activation ref 注释）。
- **全局假设**：A5、F5 同构。
- **已知近似/登记**：gelu 的 erf/exp 差异未区分，sfu 统一按 silu 口径 2/元素计（登记）。
- **运行时 actions 实证**：swigluCounts 数值与 F5 同式（verify-moe-norm §3.2 语义澄清项
  逐位核验）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 4.2 层内公共槽位

#### 4.2.1 rmsnorm — RMSNorm — [B-layer-pre](#b-layer-pre) / [B-layer-post](#b-layer-post)

- **触发面**：57/59 模型（1898 节点/8633 实例）：input/post-attention norm 全仓通用。
  仅 Qwen3.8-2.4T×2 纯用 gemma_rmsnorm（无本叶）。
- **matrix**：精确零。
- **vector / sfu**：vector=`4·T·H_n`（x²、mean-reduce、×rsqrt、×w）；sfu=`T`
  （rsqrt=1，A5）。实现：`rmsnormCounts`（`counts.js:67-79`，`rmsnormCounts`）；
  case（`extractor.js:588-590`，case "rmsnorm"/"gemma_rmsnorm"）。
- **bytes.weights**：`H_n·b`。
- **bytes.actIn / actOut**：actIn=`T·H_n·b`、actOut=`T·H_n·b`。
- **单位与换算**：H_n = `staticWidth(input_shape)`（正维之积）。
- **基础分解**：`rmsnorm` + `elementwise`(×w)。
- **来源**：三等分解声明 mul/reduce/rsqrt/mul（无单一 aten 对应）
  （`index.js:115-124`，FORMULAS.rmsnorm）。
- **全局假设**：A5。
- **已知近似/登记**：无。
- **运行时 actions 实证**：vector=4TH_n、sfu=T、bytes 三项逐位吻合
  （verify-moe-norm §3.1 表）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### gemma_rmsnorm — Gemma RMSNorm

- **触发面**：31/59 模型（2179 节点/4287 实例）：Qwen3.5/3.6/3.8（除 Flash-Next）
  与 MiniMax-M3 系的 q/k attention norm（Gemma 风格 checkpoint：缩放前 (1+w)）；
  Flash-Next 无 gemma、Qwen3.8-2.4T×2 纯 gemma（负例互证，verify-moe-norm §1）。
- **matrix**：精确零。
- **vector / sfu**：rmsnorm 基础上 **+ `T·H_n`**（(1+w) 加法）→ `5·T·H_n`；sfu 同
  rmsnorm（`T`）。实现：`rmsnormCounts({weightOne:true})`（`counts.js:67-79`）+
  case（`extractor.js:588-590`，按 operatorId 置 weightOne）。
- **bytes.weights / actIn / actOut**：与 rmsnorm 相同（(1+w) 是逐元素加法，不加流量）。
- **基础分解**：`rmsnorm` + `elementwise`((1+w)、×w)。
- **来源**：三等分解声明（Qwen3.5 Gemma 风格 checkpoint 语义）
  （`index.js:125-133`，FORMULAS.gemma_rmsnorm）。
- **全局假设**：A5。
- **已知近似/登记**：无。
- **运行时 actions 实证**：M3 q/k norm（H_n=2048）vector=5·T·2048 逐位吻合
  （verify-moe-norm 反向抽样表）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### 4.2.2 linear — Linear（跨槽位投影）

- **触发面**：59/59 模型（9891 节点/28707 实例）——全部投影形态：q/k/v/o、MLP
  gate/up/down、fused QKV（qkv_gate_proj、qkv_index_proj、fused_wqa_wkv、qkvz）、
  KDA 的 ba/decay 投影、MoE shared experts、indexer 投影、**lm_head**
  （output head，如 Qwen3.5-0.8B `[−1,−1,1024]→[−1,−1,248320]`）、视觉 patch_embed
  （如 M3 `[−1,−1,3,196]→[−1,−1,1280]`，conv 语义以 GEMM 计）。单模型节点数
  4（MiniMax-M2.7：融合输入投影/o_proj/融合 MLP ×62 + lm_head）～569（Kimi-K3）；
  MoE 专家主干 GEMM 不在本叶（见 swiglu routed）。
- **matrix**：`T·out·in·expertFraction`（routed 路径按 `k/E` 缩放，
  `expertFractionFor`，`extractor.js:50-57`）。实现：case "linear"
  （`extractor.js:362-379`，case "linear"）+ `linearCounts`（`counts.js:16-28`，
  `linearCounts`）。
- **vector / sfu**：**主链精确零**——`linearCounts` 唯一调用点不传 bias（默认 false），
  vector 恒 0；sfu 字面量 0。`T·out` 是 bias=true 的条件容量，主链为死分支。
- **bytes.weights**：`out·in·b`（权重读一遍；packed qweight 无 logical_weight_shape 时
  `linearLogicalShape` 返回 null，case 再以 `derivedLinearShape`（input/output 正维积）
  兜底，两者皆缺才返回 null → unknownComputePaths；本仓 59 模型实测 0 例）。
- **bytes.actIn / actOut**：actIn=`T·in·b`、actOut=`T·out·b`（读写各一次）。
- **单位与换算**：`aten::mm` 的 `m·n·2k` FLOPs → MACs 已换算
  （`index.js:27-36`，FORMULAS.linear ref 注释）。
- **基础分解**：`matmul`（×1）[+ bias `elementwise`：主链不启用]。
- **来源**：一等 aten::mm；A7：融合实现按语义分解计数（`index.js:29-30` ref 注释）。
- **全局假设**：A5（bias 逐 flop）、A7。
- **已知近似/登记**：
  - **同名双轨**：extractor case "linear" 手搓（含 embed 路径排除）+ registry
    `linear.counts` 死引用（§6）。
  - **embed 排除判据**：无 operatorId 但 weight_shapes ≥2 维、路径命中
    `/(^|\.)(patch_)?embed/` 的节点按 embedding gather 计（结构化路径判断，W3 TODO
    换 attributes 标记，`extractor.js:353-359`，isLinearNode 注释）。
  - bias=false 全局（bias 流量未计，与旧链一致——登记）。
- **运行时 actions 实证**：T=128，[3072,6144]：matrix=2,415,919,104、weights=37,748,736、
  actIn=1,572,864、actOut=786,432，全等；bias=true 对照组 vector=393,216=T·out
  （验证条件分支语义）；routed 合成 experts=160/topk=8：matrix×0.05=120,795,955.2、
  `shared_experts` 路径因 `(?<!shared_)` 不缩放（verify-linear §1.1）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### 4.2.3 rope — RoPE — [B-layer-attn](#b-layer-attn) 各 kind

- **触发面**：59/59 模型（1101 节点/2393 实例）：每注意力层 q/k 旋转（含 MLA 的
  rope 分量、V4 的 rope/inverse_rope、partial rotary 因子）。
- **matrix**：精确零。
- **vector / sfu**：vector=`3·T·D_rope`（每维对 4 乘 2 加 = 3 flop/元素）；
  sfu=0（**A3：sin/cos 查表**）。实现：`ropeCounts`（`counts.js:108-115`，
  `ropeCounts`）；case（`extractor.js:584-587`，case "rope"，partial_rotary_factor
  属性优先）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：各 `2·T·D_rope·b`（q+k 两路，读写各一次）。
- **单位与换算**：D_rope=headDim×partial_rotary_factor。
- **基础分解**：`rope`（词汇表原语）。
- **来源**：三等分解声明（无单一 aten 对应）（`index.js:76-85`，FORMULAS.rope）。
- **全局假设**：A3、A5。
- **已知近似/登记**：~~GLM-5.3-Flash 的 qk_rope_head_dim=0 → ropeDims=0~~
  **更正（2026-09-09 双向验证，verify-attention F4）：该声明不成立**。rope case 读
  `config.headDim`（`extractor.js:584-587`），GLM-5.3-Flash 归一化 headDim=256（来自
  `qk_head_dim`/`qk_nope_head_dim`=256；`head_dim:0` 与 `qk_rope_head_dim:0` 均不进入
  该计算），实测该模型 rope 叶 11 节点 vector=`3·T·256`、bytes 各 `2·T·256·b`
  **非零**——无「精确零 rope」语义。「MLA rope 分量宽（qk_rope_head_dim）」若需单列
  口径应新增字段，不改 ropeDims 来源。
- **运行时 actions 实证**：V4-Flash 首叶 vector=196,608/实例=3·128·512、bytes 262,144 B
  （verify-attention §2.13）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### 4.2.4 残差槽位 — attention_residual + 普通残差加缺口 — [B-layer-res](#b-layer-res)

##### attention_residual — Attention Residual（Kimi-K3 residual bank）

- **触发面**：1/59 模型——moonshotai/Kimi-K3（47 节点×93：每层 attention 前 + MLP 前
  各一 + 层组乘数；config attn_res_block_size=12）。
- **matrix**：score 小投影 `T·H·1`（[1,H] 投影）。实现：ctxBuilder 组合
  `F3(norms) + F1(scoreProj) + F2(aggregate) + add(mix)`
  （`index.js:308-319`，FORMULAS.attention_residual；`extractor.js:723-728`，
  ctxBuilders.attention_residual）。
- **vector / sfu**：F3×2 = `8·T·H` + `2T`；F2（流数维 softmax）= `3·T·H` + `2·T·H`；
  add = `T·H`。
- **bytes.weights**：`H·b`（score 投影）+ `2·H·b`（两个 norm weight）。
- **bytes.actIn / actOut**：各分量读写一次之和（norms `2·TH·b`、scoreProj `TH·b`、
  aggregate `TH·b`、mix `2·TH·b` → actIn ≈ `6·TH·b`；actOut ≈ `4·TH·b`；以
  sumCounts 实算为准）。
- **基础分解**：`rmsnorm`×2 + `matmul` + `softmax` + `add`。
- **来源**：二等对照 Kimi-K3 modeling_kimi_linear.py（use_attn_residuals :907、
  _forward_attn_residual :931、attn_res_block_size=12）+ 三等分解
  （`index.js:310-313` ref 注释）。
- **全局假设**：A4、A5；snapshot bank 的存储流量不计（只计每次前向的读写动作）。
- **已知近似/登记**：block 写层的 snapshot 保存流量未单列（并入 mix/add 一阶口径）。
- **运行时 actions 实证**：sumCounts 合成算术逐项复算一致（verify-moe-norm §3.5 方法
  同源；本组 ctxBuilder 在 2e83940/a4d709a 间无变更）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

**普通残差加（+x）缺口**：decoder 层普通残差加法无算子节点，2TH·b×2/层未计——
量级备注（60 层 ≈6MB/token，相对权重流量可忽略）是量化后的暂缓决定，不是遗漏
（cost_counts.md「结构级缺口」）。表 B 在 [B-layer-res](#b-layer-res) /
[B-layer-mlp](#b-layer-mlp) / [B-layer-moe](#b-layer-moe) 三处以「缺口」标记。

### 4.3 attention 子块 — 按 kind（模块↔算子两级并排）— [B-layer-attn](#b-layer-attn)

> 每节结构：**模块级融合公式**（registry/legacy 镜像）→ **可展开子算子清单**（逐个
> 公式+实现位置，全模板）→ **一致性断言**（融合 ≈ Σ子级；不等即登记）。
> `type==='attention'` 模块容器（8 种 kind，59 模型 95 个 kind×model 实例：qwen35_full
> 29、linear 34、mla 10、dsa_sparse_mla 9、dsv4 5、gqa 4、sparse 2、qsa 2）恒为父
> 容器，`computeNodeCosts` 对其计 0、actions=null（`compute.js:43-48`）；extractor 保留
> 的 type==="attention" legacy 镜像（`extractor.js:329-338`，attention kind 分派）当前
> **0 个无子 attention 叶**——无可达叶，属 W5 随旧链清理的 legacy 镜像。

#### 4.3.1 dense — 通用 GQA/full + qwen35_full — [B-attn-dense](#b-attn-dense)

**模块级融合公式**（registry `matmul.counts=attentionCounts`，F2，
`index.js:37-46`；`counts.js:43-60`，`attentionCounts`）：
matrix=`Nh·T·S·(D+dv)`；vector=`3·Nh·T·S`、sfu=`2·Nh·T·S`（A2 softmax 段）；
actIn=`(Nh·T·D + kvH·S·D + kvH·S·dv + 2·Nh·T·S)·b`；
actOut=`(2·Nh·T·S + Nh·T·dv + kvH·T·(D+dv))·b`。

**子算子清单**（分解链三叶 + qwen35_full 专属一叶）：

##### matmul — MatMul（scores/context 两叶）

- **触发面**：48/59 模型（878 节点/4162 实例）。凡注意力走**分解链**的模型每层发射
  scores + context 两叶：MiniMax-M2.7（GQA 文本塔 ×62）、GLM-4.7（GQA ×89+3）、
  Qwen3.5/3.6/3.8 全系 full-attention 层（attention_kind=qwen35_full）、
  MLA 家族的文本塔（DeepSeek-R1/V3.1、Kimi 全系 ×61（+MTP 组）、Kimi-K3 ×24）、
  以及**全部视觉塔**（MiniMax-M3 ×32、Qwen 系 ×12–27、Kimi ×27、GLM-5.3-Flash ×24、
  V4-Vision-Exp ×32）。缺 matmul 的 11 个模型 = 文本侧整体融合的稀疏家族且无视觉塔
  （DeepSeek-V3.2、V4×4 非 vision、GLM-5/5.1/5.2/5.3 系 6 个）。
- **matrix**：scores 叶 `Nh·T·S·D` + context 叶 `Nh·T·S·dv`（prefill S=T、decode T=1）。
  实现：case "matmul"（`extractor.js:380-421`，case "matmul"），scores/context 用
  output_shape 模式匹配区分（`extractor.js:88-101`，`attentionShapePatterns`；
  context 模式先判防吞）。
- **vector / sfu**：精确零（两叶都是纯 GEMM；softmax 归 softmax 叶）。
- **bytes.weights**：0——注意力无权重，投影由 linear 叶计费。
- **bytes.actIn / actOut**：scores 叶 actIn=`(Nh·T·D + S·Nh·D)·b`（读 Q、K）、
  actOut=`Nh·T·S·b`（写 scores）；context 叶 actIn=`(Nh·T·S + S·Nh·dv)·b`
  （读 scores、V）、actOut=`Nh·T·dv·b`（写 context）（`extractor.js:394-419`）。
- **单位与换算**：matrix 为 MACs；`aten.bmm` 的 `2·m·n·k` FLOPs 已 ÷2。
- **基础分解**：`matmul`×2。
- **来源**：一等 `aten::bmm` ×2（`index.js:39-41` ref 注释）。
- **全局假设**：无特别引用（一阶读写各一次为本仓默认口径）。
- **已知近似/登记**：
  - **同名双轨**：registry 条目 `matmul.counts=attentionCounts`（F2 融合口径）在运行时
    被手搓 case 遮蔽（switch 提前 return）——死引用（G2 ①，§6）。
  - K 读宽按 `Nh·D`（repeat_kv 物化口径）而非 `kvH·D`：分解链把 GQA 的 K/V 扩展
    视为叶间真实张量传递；融合口径（F2）按 kvH 缩。两口径并存是有意分工
    （`counts.js:34-36` 注释）。
  - decode 的 keyTokens=options.sequence（上下文全长），与 F2 decode 行为一致。
- **运行时 actions 实证**：GLM-5.3-Flash 视觉塔 softmax/vector 旁证逐位吻合
  （verify-attention §2.1/§2.2 probe）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### softmax — Softmax

- **触发面**：与 matmul 完全同集合（程序化验证 set 相等），48/59 模型
  （439 节点/2081 实例）——分解链每对 scores/context 之间一叶。
- **matrix**：精确零（归约+逐元素，不用矩阵单元）。
- **vector / sfu**：`vector=3·elements`（max/sum 归约 + 乘）、`sfu=2·elements`
  （exp + div），elements=`Nh·T·S`。实现：`softmaxCounts`（`counts.js:202-209`，
  `softmaxCounts`），case "softmax"（`extractor.js:576-583`，case "softmax"）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：各 `Nh·T·S·b`（读 logits 一遍、写 probs 一遍，A2 单遍）。
- **基础分解**：`softmax`。
- **来源**：一等 `aten::_softmax`；torch flop_counter 明确不数 softmax，本仓**有意
  超越**计 vector/sfu/bytes（`index.js:49-51` ref 注释）。
- **全局假设**：A2（融合单遍；多遍读放大不建模）、A5。
- **已知近似/登记**：flash kernel 在线 softmax 不落地全量 scores/probs 时，本叶与
  F2 系融合注意力内的 `2·scores` 项同属「理论口径」——保持一致优先。
- **运行时 actions 实证**：GLM-5.3-Flash 视觉 softmax vector=3,145,728=3·Nh_v·T_v·S_v
  逐位吻合（verify-attention §2.2）。**本叶是全语料 vector/sfu 时间贡献第一**（§5）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### attention_output_gate — Attention Output Gate（qwen35_full 专属）

- **触发面**：29/59 模型（355 节点/355 实例；每 full-attention 层一个显式节点，
  不进层组乘数）——Qwen3.5/3.6/3.8 的 qwen35_full 模板
  （`ops/index.js:287-291`，qkv_gate fused 投影里第 2 段 gate 经 sigmoid 调制输出；
  activation 指针 :289、implementation 指针 :290）。
- **matrix**：精确零。
- **vector / sfu**：`vector=T·W`、`sfu=2·T·W`（sigmoid = exp+rcp，A5）。实现：
  `gateCounts`（`counts.js:84-95`，`gateCounts`），extractor 四门控共用 case
  （`extractor.js:593-597`，gate 四 case）。
- **bytes.weights**：0（gate 向量来自 qkv_gate_proj 输出切片，无独立投影权重；
  gateCounts 的 `gateProjection` 分支在本仓四门控叶均未启用）。
- **bytes.actIn / actOut**：actIn=`T·W·b`（gate+context 拼接读，按宽 W 一阶计）、
  actOut=`T·W·b`。
- **单位与换算**：W = `staticWidth(output_shape)`（**全部正维之积**；qwen35_full 叶输出
  `[.., Nh, D]` → W = Nh·D，如 Qwen3.5-0.8B 8·256 = 2048）。
- **基础分解**：`elementwise`（sigmoid·mul）。
- **来源**：二等 modeling 对照——vLLM/SGLang `fused_sigmoid_mul`
  （`ops/index.js:290` implementation 指针；`index.js:396-406` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：`activation: sigmoid|none` 由 plan 决定（`ops/index.js:289`）；
  activation=none 的模型本叶仍存在但语义为恒等——探针未区分（登记）。
- **运行时 actions 实证**：T=128、W=6144：vector=786,432=T·W、sfu=1,572,864=2·T·W、
  actIn=actOut=1,572,864（verify-linear §3.2）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

**一致性断言（dense）**：融合 F2 ≡ Σ子级 在 matrix/vector/sfu 上**精确成立**
（matrix=`Nh·T·S·(D+dv)` = scores 叶 + context 叶；vector/sfu = softmax 叶）。
bytes 两处**登记差**（G3）：① K/V 读宽——融合按 kvH，分解链按 `Nh·D`（repeat_kv
物化，GQA 下 F2 < 链）；② kvWrite——融合 actOut 含 `kvH·T·(D+dv)`（cache 写），
分解链由 k/v_proj linear actOut 计、attention 叶不计。两差均为有意分工
（`counts.js:34-36`、`extractor.js:486-489` 注释）。

#### 4.3.2 MLA（DeepSeek / Kimi / GLM MLA）— [B-attn-mla](#b-attn-mla)

**模块级融合公式**：F2 以 MLA 参数实例化（headDim=kvLora+rope、valueDim=kvLora、
kvH=1）；压缩链以融合声明叶承载（q_a+norm → mla_query_compress；kv_a →
mla_kv_compress），核心 scores/context/softmax 仍走 dense 分解链三叶。

##### mla_query_compress — MLA Query Compression

- **触发面**：19/59 模型（221 节点/1124 实例）：DeepSeek-R1/V3.1/V3.2（2 节点×61）、
  Kimi-K2 全系/K2.5/2.6/2.7（2×61）、Kimi-K3（23×24）、GLM-5/5.1（2×78）、
  GLM-5.2/5.3 系（38×78）、GLM-5.3-Flash×2（11×11）——即全部 MLA/DSA 文本塔的
  q_a 投影 + q_a_layernorm（q_b 由独立 q_b_proj linear 叶计）。
- **matrix**：`= T·qLora·H`（q_a 投影 GEMM；q_b 不在本叶）。实现：ctxBuilder 组合
  `F1(qa) + F3(norm)`（`index.js:265-276`，FORMULAS.mla_query_compress；
  `extractor.js:695-700`，ctxBuilders.mla_query_compress）。
- **vector / sfu**：norm 段 vector=`4·T·qLora`；sfu=`T`（rsqrt 归 SFU，A5）。
- **bytes.weights**：`(qLora·H + qLora)·b`（q_a 权重 + norm 权重）。
- **bytes.actIn / actOut**：actIn=`(T·H + T·qLora)·b`（x 读 + norm 读）、
  actOut=`2·T·qLora·b`（投影写 + norm 写；下游 q_b 再读）。
- **基础分解**：`matmul` + `rmsnorm`。
- **来源**：三等分解声明 + 二等对照 models/deepseek-ai/DeepSeek-V3.1/
  modeling_deepseek.py（q_a_proj :661、q_a_layernorm :664、q_b_proj 调用点 :769）
  （`index.js:267-269` ref 注释）。
- **全局假设**：A4（分解声明）、A5。
- **已知近似/登记**：组合里**不含 q_b**——2026-09-07 审计发现含 qb 会与独立
  q_b_proj 叶双计（Kimi/GLM 各 +19M/+25M 参数/层）（`extractor.js:696-698` 注释）。
- **运行时 actions 实证**：GLM-5.3-Flash（qLora=1536、H=4096、T=128）：matrix=
  805,306,368=T·qLora·H、vector=786,432=4·T·qLora、sfu=128=T、
  weights=12,585,984=(qLora·H+qLora)·b、actIn=1,441,792=(T·H+T·qLora)·b、
  actOut=786,432=2·T·qLora·b——六元全逐位吻合（verify-attention §2.3）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### mla_kv_compress — MLA KV Compression

- **触发面**：24/59 模型（464 节点/1369 实例）：上列 19 个 MLA 模型 + DeepSeek-V4×5
  （V4 的 compressor 叶，Flash 系 41 节点/模型、Pro 系 60×61）。kv_a 投影 + latent
  拆分前的投影段；**latent/压缩态 cache 写回由本叶 actOut 计**。
- **matrix**：`= T·outW·H`；outW 以节点自身 output_shape 为权威
  （MLA latent = kvLora+rope；V4 压缩 = 2·D·k）。实现：ctxBuilder `F1(proj)`
  （`index.js:277-287`；`extractor.js:701-714`，ctxBuilders.mla_kv_compress）。
- **vector / sfu**：精确零（组合仅 F1）。
- **bytes.weights**：`outW·H·b`。
- **bytes.actIn / actOut**：actIn=`T·H·b`；actOut=`T·outW·b`（= latent/压缩态 cache 写）。
- **基础分解**：`matmul`。
- **来源**：三等分解声明 + 二等对照 DeepSeek-V3.1 modeling_deepseek.py
  kv_a_proj_with_mqa（:669）（`index.js:279-281`）；V4 compressor 语义
  /tmp/m11-formulas/dsv4.md §(b)6。
- **全局假设**：A4。
- **已知近似/登记**：V4 ctx 失配已修（2026-09-08）——旧式用
  `kvLoraRank+qkRopeHeadDim` 拼 out 维，V4 无 kvLoraRank 得 out=64，compressor macs
  差 16–32×；现以 `staticWidth(output_shape)` 为权威（`extractor.js:702-705` 注释）。
- **运行时 actions 实证**：V4-Flash ratio=4 compressor outW=2048=2·D·2、
  GLM-5.3-Flash outW=512：matrix=T·outW·H 逐位吻合（verify-attention §2.4）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### mla_output_gate — MLA Output Gate

- **触发面**：1/59 模型——moonshotai/Kimi-K3（23 节点×24，每 MLA 层一个）。ref 注释
  「目录仅 Kimi-K3 发射此叶」与探针一致。
- **matrix**：精确零。
- **vector / sfu**：`vector=T·W`、`sfu=2·T·W`（sigmoid 门乘，A5）。实现：`gateCounts`
  （`counts.js:84-95`；`extractor.js:593-597`）。
- **bytes.weights**：0（门来自融合投影输出，本叶无独立 W_g；registry 公式
  `O'=sigmoid(W_g x)·O` 中的 W_g 若独立成叶应由 linear 计）。
- **bytes.actIn / actOut**：各 `T·W·b`。
- **单位与换算**：W = `staticWidth(output_shape)`（全部正维之积）；Kimi-K3 输出
  `[.., 96, 128]` → W = 96·128 = **12288**（≠ 末维 128，末维口径会低估 96×）——
  本叶是「W=output_shape 末维」旧表述的实际反例（verify-linear 修正 3）。
- **基础分解**：`elementwise`（sigmoid·mul）。
- **来源**：二等对照 models/moonshotai/Kimi-K3/modeling_kimi_linear.py
  （mla_use_output_gate :398、门乘 :470）（`index.js:297-307`）。
- **全局假设**：A5。
- **已知近似/登记**：无。
- **运行时 actions 实证**：T=128、W=6144 对照组逐位吻合（verify-linear §3.2）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

**一致性断言（MLA）**：压缩叶组合（F1+F3）与子级构造恒等（sumCounts）；核心三叶
断言同 dense。

#### 4.3.3 DSA / QSA 稀疏注意力（三分支共用 qsa_attention case）— [B-attn-dsa](#b-attn-dsa) / [B-attn-qsa](#b-attn-qsa)

**模块级融合公式**：indexer（F2 打分 + F8 topk）+ 稀疏核心（F2 变体，
S=`min(S, indexerBudget)`，按 kind 分派读宽与 kvWrite）。

##### qsa_indexer — QSA Indexer

- **触发面**：16/59 模型（327 节点/698 实例），与 qsa_attention 完全同集合：
  Qwen3.8-Flash-Next×2（qsa，12×12）、DeepSeek-V3.2（dsa_sparse_mla，2×61）、
  DeepSeek-V4×5（dsv4_sparse_mla，Flash 系 21、Pro 系 30）、GLM-5/5.1（2×78）、
  GLM-5.2/5.3 系（38×78）、GLM-5.3-Flash×2（11×11）。
- **matrix**：indexer 打分 `Nh_i·T·S·D_i` ×2（F2 双 bmm，S=上下文全长）。实现：
  ctxBuilder 组合 `F2(score) + F8(topk)`
  （`index.js:352-363`，FORMULAS.qsa_indexer；`extractor.js:715-718`，
  ctxBuilders.qsa_indexer：indexerNHeads/indexerHeadDim 独立参数）。
- **vector / sfu**：F2 段 `3·Nh_i·T·S` + `2·Nh_i·T·S`；topk 段 vector=`T·S`
  （experts=S）、sfu=`T·budget`（normTopkProb 除法）。
- **bytes.weights**：0（indexer 投影 W_q/W_k 由相邻 linear 叶计）。
- **bytes.actIn / actOut**（F2 组合按 `attentionCounts` 原样带入，含 V 路三项）：
  actIn=`(Nh_i·T·D_i + Nh_i·S·D_i[K] + Nh_i·S·D_i[V] + 2·Nh_i·T·S)·b + T·S·b`
  （Q/K/V 全上下文读 + scores/probs 读写 + topk 全量打分读）；actOut=
  `(2·Nh_i·T·S + Nh_i·T·D_i[context] + Nh_i·T·2D_i[kvWrite])·b + T·budget·b`
  （scores/probs + context 写 + indexer K/V cache 写 + top-k 索引写出，供
  qsa_attention 读）。**登记**：V 读、context 写、kvWrite 三项是 F2 通用公式带入的
  「indexer 语义外项」——indexer 只产打分与索引，真实核不写 KV cache、不产 context；
  现按 F2 原样计为理论口径（GLM-5.3-Flash prefill S=128 单实例实测：actIn=5,275,648 B
  = F2 5,242,880 + topk 32,768，其中 V 读 1,048,576 B ≈19.9%；actOut=5,767,168 B
  = F2 5,242,880 + topk 131,072，三项合计 ≈3.0MB ≈54.5%；V4-Flash actIn
  10,518,528 B、actOut 10,616,832 B 逐位吻合——verify-attention F3）。若后续做
  indexer 专用 F2 变体，应消去这三项并同步 minimax_sparse_indexer。
- **单位与换算**：budget=indexerBudget（`index_topk`/`indexer_budget` 直读，
  normalize.js:180）。
- **基础分解**：`matmul`×2 + `softmax` + topk 原语（词汇表外，诚实计）。
- **来源**：二等（vLLM.SparseAttnIndexer / DeepseekV4Indexer，
  ops/index.js:393-405 implementation 指针；V3.2/V4 modeling 未入库——离线取证）+
  三等分解（F2+F8）；Qwen3.8-Flash-Next / GLM-5.3-Flash 的 indexer 源码已于
  2026-09-08 入库升级为二等：models/Qwen/Qwen3.8-Flash-Next/modeling_qwen4_exp.py
  Qwen4ExpTextQSAIndexer（:671-687）、models/zai-org/GLM-5.3-Flash/
  modeling_glm5_next.py Glm5NextTextIndexer（:739-880）。依据
  /tmp/m11-formulas/evidence-qsa-glm.md 裁决一/二。
- **全局假设**：A2（indexer 打分的 softmax 段）、A4。
- **已知近似/登记**：topk actOut 宽度实为 `budget+compress_ratio-1`（Qwen :722 /
  GLM :870-872 的尾块补选），现按 `T·budget` 上限计——登记为理论口径；F2 带入三项
  见 bytes 条目登记。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### qsa_attention — QSA Sparse Attention

- **触发面**：同 qsa_indexer，16/59 模型（327 节点/698 实例）。三种
  attention_kind 共用本 case：`qsa`（Qwen3.8-Flash-Next×2 逐头 GQA/MHA）、
  `dsa_sparse_mla`（DeepSeek-V3.2 + GLM-5 系×8，MLA latent）、
  `dsv4_sparse_mla`（V4×5，MQA 压缩态）。
- **matrix**：`Nh·T·S_sel·(D+dv)`，S_sel=`min(S, indexerBudget)`。实现：case
  "qsa_attention"（`extractor.js:422-476`，case "qsa_attention"；矩阵镜像
  qsaCoreMacs `extractor.js:186-193`）。
- **vector / sfu**：精确零（融合核；softmax 段以 bytes 中间量体现）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**（F2 整体口径，按 kind 分派读宽，`extractor.js:449-473`）：
  - `dsa_sparse_mla`：kvH=1，K 读宽 `kvLora+rope`、V 读宽 `kvLora`，kvWrite=0
    （latent cache 写由 kv_a_proj linear actOut 计）；
  - `dsv4_sparse_mla`：kvH=config.kvH（=1），读宽 D/dv；**kvWrite=0（C2 已落码，
    a4d709a `extractor.js:453-455`：`kvWrite = latentRead || kind === "dsv4_sparse_mla"
    ? 0 : kvHeads·T·(D+dv)`，注释「压缩态写入归 compressor 叶、窗口写入已单列
    （fc99269），防三重计费」）**——修复前实现为非零 `kvH·T·(D+dv)`（V4 无
    kvLoraRank → latentRead=false），与「压缩态写归 compressor」规则冲突疑双计
    （verify-attention F1，V4-Flash 单实例 ≈256KB/层）；
  - `qsa`：kvH=config.kvH，读宽 D/dv，计 kvWrite=`kvH·T·(D+dv)`（paged cache
    写回模板内无叶承担）；
  - **滑窗补记（fc99269，2026-09-08）**：`dsv4_sparse_mla` 层另有混合读窗口项
    `W_win = kvH·min(S, slidingWindow)·D`（`extractor.js:456-461`，dsv4Window），
    actIn/actOut 各计一次。
  公式（a4d709a 口径）：actIn=`(Nh·T·D + kvH·S_sel·(kW+vW) + T·S_sel + W_win +
  2·Nh·T·S_sel)·b`；actOut=`(2·Nh·T·S_sel + Nh·T·dv + kvWrite + W_win)·b`。
- **单位与换算**：`T·S_sel` 项 = top-k 索引读（int32 按 b 计，由 qsa_indexer topk
  actOut 写、本叶读）；`2·scores` ×2 = A2 的 scores/probs 写+读。
- **基础分解**：`matmul`×2 + `softmax`（融合核；非基础组合等价式——bytes 内嵌
  scores/probs 中间量，登记为理论口径）。
- **来源**：二等 modeling 对照（FlashMLA-sparse 吸收式核按 latent 读宽、逐头变体按
  kvH——变体矩阵见 cost_counts.md F2 表）；bytes 结论 /tmp/m11-formulas/qsa.md
  §2.3-2.4、§4.3（并集偏差见 §5 表行 2）；GLM-5.3-Flash 证据改判
  qsa→dsa_sparse_mla 见 /tmp/m11-formulas/evidence-qsa-glm.md 裁决二
  （2026-09-08）；W_win 依据 fc99269（vLLM c128a 压缩历史+原始滑窗混合读，原取证
  /tmp/m11-formulas/dsv4-sliding-window.md 已不在盘，按 commit message 登记）。
- **全局假设**：A2（4·scores 记本节点——稀疏模板无独立 softmax 叶）、A5。
- **已知近似/登记**：prefill 因果三角未折减（与 F2 全仓口径一致，保守）；
  选中集「读一遍 vs 并集」偏差见 qsa.md §5 表行 2；kvWrite 家族规则一句话版：
  「模板里已有叶写了 cache 条目的（latent=kv_a_proj、压缩态=compressor），融合核
  不再写；没有的（逐头 K/V）由融合核补记」（qsa.md §4.3）——dsv4_sparse_mla 分支
  的例外已于 a4d709a 按 C2 归零，规则与实现现在一致。
- **运行时 actions 实证（a4d709a）**：V4-Flash prefill S=128 单实例：
  actIn=13,008,896 B =（4,194,304[NhTD] + 131,072[kv·S_sel·(D+dv)] + 16,384[T·S_sel]
  + 65,536[W_win] + 2,097,152[2·scores]）·2；actOut=**12,713,984 B** =
  (2,097,152 + 4,194,304[context] + 65,536[W_win])·2——kvWrite=0 逐位吻合
  （/tmp/m11-formulas/probe-c12-rewrite.mjs 实测；修复前 actOut=12,976,128 含
  kvWrite 262,144 B）。dsa 分支 kvWrite=0 另有 GLM 实证（verify-attention §2.7）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

**一致性断言（DSA/QSA）**：indexer = F2+F8（sumCounts 构造恒等）；稀疏核心 matrix
= F2 以 S_sel 实例化（构造恒等）。

#### 4.3.4 MiniMax M3 块稀疏 — [B-attn-minimax](#b-attn-minimax)

##### minimax_sparse_indexer — MiniMax M3 Block Indexer

- **触发面**：2/59 模型——MiniMax-M3 / M3-MXFP8（1 节点×57，稀疏层组整体乘数；
  config sparse_attention_freq = 3 个 dense 层 + 57 个稀疏层）。
- **matrix**：index 头块打分 `Nh_i·T·S·D_i` ×2（F2，S=上下文全长，index_block_size=128
  池化在建模上并入打分宽）。实现：ctxBuilder `F2(score) + F8(topk)`
  （`index.js:407-418`；`extractor.js:719-722`，ctxBuilders.minimax_sparse_indexer）。
- **vector / sfu**：F2 段 `3·Nh_i·T·S` + `2·Nh_i·T·S`；topk 段 vector=`T·S`、
  sfu=`T·sparseTopkBlocks`（=16，config sparse_topk_blocks=16）。
- **bytes.weights**：0（index q/k 投影由相邻 qkv_index_proj linear 叶计）。
- **bytes.actIn / actOut**（F2 组合按 attentionCounts 原样带入，含 V 路三项）：
  actIn=`(Nh_i·T·D_i + Nh_i·S·D_i[K] + Nh_i·S·D_i[V] + 2·Nh_i·T·S)·b + T·S·b`；
  actOut=`(2·Nh_i·T·S + Nh_i·T·D_i[context] + Nh_i·T·2D_i[kvWrite])·b +
  T·sparseTopkBlocks·b`（块 id 写出）。**登记**：V 读、context 写、kvWrite 为
  F2 通用公式带入的「indexer 语义外项」（与 qsa_indexer 同一条登记，量级：
  Nh_i=4、D_i=128 下 V 读=512·S、kvWrite=1024·T 元/实例——verify-attention F3）。
- **基础分解**：同 qsa_indexer。
- **来源**：二等对照 models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py
  MiniMaxM3VLIndexer（:492：index_block_size=128 池化打分 + topk_blocks=16 选块；
  index-value 路 checkpoint 显式关闭，config sparse_disable_index_value 实查非零）
  + 三等分解（`index.js:409-412` ref 注释）。
- **全局假设**：A2、A4。
- **已知近似/登记**：init/local blocks 保留（sparseInitBlock=0+sparseLocalBlock=1）
  在 attention 侧生效，indexer 不计；index-value 路显式关闭不建模；F2 带入三项
  见 bytes 条目登记。
- **运行时 actions 实证**：matrix=268,435,456 量级、sfu=T·16 形态逐位吻合
  （verify-attention §2.8）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### minimax_sparse_attention — MiniMax M3 Block-Sparse GQA

- **触发面**：2/59 模型——MiniMax-M3 / M3-MXFP8（1 节点×57）。
- **matrix**：`Nh·T·(blocks·blockSize)·(D+dv)`，blocks=sparseTopkBlocks+Init+Local、
  blockSize=sparseBlockSize。实现：case
  （`extractor.js:477-508`，case "minimax_sparse_attention"；矩阵镜像
  minimaxSparseCoreMacs `extractor.js:194-201`）。
- **vector / sfu**：精确零（融合核）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`(Nh·T·D + kvH·selected·(D+dv) + 2·Nh·T·selected)·b`
  （Q 读 + 选中 KV 读 + scores/probs 中间量）；actOut=`(2·Nh·T·selected + Nh·T·dv +
  kvH·T·(D+dv))·b`——**含 kvWrite**（M3 稀疏注意力为融合算子，cache 写回在本叶；
  dense 侧由 k/v_proj linear actOut 计，两侧账目自洽，`extractor.js:486-489` 注释）。
- **基础分解**：`matmul`×2 + `softmax`（融合核）。
- **来源**：二等对照 modeling_minimax_m3_vl.py MiniMaxM3VLAttention（:408）+
  eager_attention_forward（:340）；transformers 库版入库 models/MiniMaxAI/MiniMax-M3/
  （HF 仓库无 modeling，二等降级口径见 evidence-manifest.json）。
- **全局假设**：A2（4·scores 记本节点）。
- **已知近似/登记**：选块 per query token、per KV 组（index_heads=kv_heads）
  （`extractor.js:490` 注释）。
- **运行时 actions 实证**：blocks=16+0+1=17、blockSize=128（config 实查）逐位吻合
  （verify-attention §2.9）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

**一致性断言（minimax）**：indexer = F2+F8 构造恒等；attention matrix = F2 以
S=blocks·blockSize 实例化（构造恒等）。

#### 4.3.5 KDA / 线性注意力 — [B-attn-kda](#b-attn-kda)

**模块级融合公式**（legacy 镜像，`extractor.js:203-271`：qwen35LinearStateMacs /
glm5NextLinearStateMacs / kimiK3LinearStateMacs / qwen4ExpLinearStateMacs）：
以 Qwen3.5 形态为例 `T·(qkvzProjection + baProjection + shortConvolution +
recurrentState + gatedNorm + outputProjection)`，其中 recurrentState=`3·vh·dv·dk`。

**一致性断言（KDA）**：模块镜像 ≡ Σ子级 matrix（linear qkvz + linear ba×2 + conv1d
+ stateUpdate）**除 norm 段**——镜像把 gatedNorm 折为 `3·vp·T` 计入 matrix，子级
gated_rmsnorm 以 vector `5·T·H_n` + sfu 计、matrix=0。数值核对（Qwen3.5-0.8B 单实例，
T=128、kp=vp=2048）：镜像 = 128·(8,388,608+32,768+24,576+786,432+6,144+2,097,152) =
1,450,967,040；Σ子级 matrix = 1,073,741,824(qkvz) + 4,194,304(ba×2) + 3,145,728(conv)
+ 100,663,296(gda) + 268,435,456(out) = 1,450,180,608；差额 786,432 = T·3·vp（=norm
段折叠项）——**登记差**（G3），非 bug：镜像属 legacy 计费路径（0 可达叶）。

##### gated_delta_attention — Gated Delta Attention（KDA 统一叶）

- **触发面**：34/59 模型（431 节点/1274 实例）：Qwen3.5 全系 21 + Qwen3.6×4 +
  Qwen3.8×6（2.4T/27B/Flash-Next）+ Kimi-K3（24×69）+ GLM-5.3-Flash×2（12×34）。
  linearAttentionMode 实测承载 ∈ {qwen3_5, qwen4_exp, kimi_k3, glm5_next}（全 delta；
  plan 另支持 'kimi' 模式，现网无载体——K2 系为 MLA）。
- **matrix**：`3·T·vh·dv·dk`（外积 + delta matvec + query；delta matvec 属矩阵
  MACs——2026-09-07 数学修正）。实现：`stateUpdateCounts` →
  `linearStateUpdateMacs`（`extractor.js:281-306`，linearStateUpdateMacs/
  stateUpdateCounts；registry 规格 `index.js:203-214`，FORMULAS.gated_delta_attention，
  F7b delta 镜像 `counts.js:127-152`）。
- **vector / sfu**：**运行时精确零**（`stateUpdateCounts` 仅接 matrix+bytes，
  `vector:0、sfu:0`——decay exp 与 beta sigmoid 的 SFU、逐元素乘当前不计）。
  registry 口径（F7b delta=true：vector=`2·T·vh·dv·dk`、sfu=`Nh·T·3`）与运行时构成
  **双轨差**，与 causal_conv1d 同性质，按 §6 判读规则以运行时为准登记（G2 ③；
  2026-09-09 双向验证抓出，verify-vision-misc 修正 1）。消除路径：
  `stateUpdateCounts` 接入 F7b 的 vector/sfu 两单元。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：stateBytes=`(convElements·kernel + vh·dv·dk)·b`
  （convElements=`kh·dk·2+vh·dv`、kernel=`linearConvKernelSize−1`；与 memory.js
  `linearStateElementsPerLayer`（memory.js:59）同源同式，vLLM
  MambaStateShapeCalculator.kda_state_shape）；actIn=actOut=stateBytes
  （**状态驻留 HBM，每 forward 读+写各一遍**，chunk 内不逐 token 重读）。
- **单位与换算**：vh=Nh（KDA 值头数=键头数）；beta/A_log/dt_bias 参数化是属性级，
  零计数影响。
- **基础分解**：matrix = 3×per-token 批量 `matmul`（外积/delta matvec/query）；
  **状态访存（bytes）非基础词汇可表达**（递推状态驻留，非一阶 act 流）——登记。
- **来源**：二等对照 models/moonshotai/Kimi-K3/modeling_kimi_linear.py
  KimiDeltaAttention（:477：beta sigmoid、safe decay exp(g)、S·k matvec）；
  Qwen3.5/GLM-5 系同族（delta=true）（`index.js:203-214` ref 注释）。
- **全局假设**：A6；执行形态假设 per-token 递推。
- **已知近似/登记**：vector/sfu 双轨差（本节登记，2026-09-09）；conv 历史与递归状态
  合并计状态访存（stateUpdateCounts，M11-P0-5）；layer0 multiplier=1 的 4× 排查记录
  见 identity_calibration.md。
- **运行时 actions 实证**：Qwen3.5-0.8B 单实例 matrix=100,663,296=3·128·16·128·128
  （×3 层组乘=301,989,888 实测）、bytes=1,683,456；Kimi-K3 matrix=603,979,776、
  bytes=3,366,912；GLM-Flash ×3 组乘 1,207,959,552、bytes=6,733,824（全部实测吻合，
  verify-vision-misc §3.4）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### gated_rmsnorm — Gated RMSNorm

- **触发面**：34/59 模型（431 节点/1274 实例），与 gated_delta_attention 同集合——
  KDA 递推输出的 gated 归一化（Qwen3.5 `output_gate_norm`、Kimi-K3、GLM-Flash）。
- **matrix**：精确零。
- **vector / sfu**：vector=`4·T·H_n`（F3）`+ T·H_n`（门乘）= 5·T·H_n；
  sfu=`T + 2·T·H_n`（rsqrt + sigmoid）。实现：`rmsnormCounts({gated:true})`
  （`counts.js:67-79`；`extractor.js:591-592`，case "gated_rmsnorm"）。
- **bytes.weights**：`H_n·b`（norm weight）。
- **bytes.actIn / actOut**：actIn=`2·T·H_n·b`（o + gate 两路读）；actOut=`T·H_n·b`。
- **单位与换算**：H_n = `staticWidth(input_shape)`（per-head gated 时为头维×头数
  展开宽，Kimi-K3 实测 96·128=12288）。
- **基础分解**：`rmsnorm` + `elementwise`(sigmoid·mul)。
- **来源**：二等对照 Kimi-K3 modeling_kimi_linear.py FusedRMSNormGated（:539，逐头
  门控）+ 三等分解 = F3(gated)（`index.js:215-225` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：phi 由模型配置决定（sigmoid/SiLU），sfu 统一按 sigmoid=2 计
  （配置差异不区分——登记）。
- **运行时 actions 实证**：Qwen3.5-0.8B（H_n=2048、乘数 3）：vector=3,932,160、
  sfu=1,573,248、weights=12,288、actIn=3,145,728 全部实测吻合
  （verify-vision-misc §3.5）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### causal_conv1d — Causal Short Convolution

- **触发面**：34/59 模型（431 节点/1274 实例），与 KDA 同集合：每个线性注意力层的
  q/k/v 短卷积（Qwen3.5 系 `conv`、Kimi-K3、GLM-Flash）。
- **matrix**：`T·width·kernel`，width=`2·keyProj + valueProj`（q/k/v 卷积通道宽）。
  实现：case（`extractor.js:633-644`，case "causal_conv1d"）。
- **vector / sfu**：运行时精确零（见「已知近似」）。
- **bytes.weights**：运行时 0（registry 版为 `channels·kernel·b`，未在运行时生效）。
- **bytes.actIn / actOut**：各 `T·width·b`（读输入窗口宽、写同宽输出，M11-P0-5）。
- **单位与换算**：matrix 为 MACs；`aten::conv1d` 的 `C_out·C_in·k·T` FLOPs 含 2×
  已换算（`index.js:68-69`）。
- **基础分解**：`conv`。
- **来源**：一等 `aten::conv1d` + SiLU 2 SFU/元素（A5）（`index.js:66-75` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：**双轨口径差（G2 ②）**——注册表 `causalConvCounts`
  （`counts.js:118-125`）含 SiLU 激活段（vector=T·width、sfu=2·T·width、
  weights=channels·kernel·b），运行时手搓 case 只计 matrix+actIn/actOut
  （vector/sfu/weights=0）。激活与 conv 权重流量目前未计——登记为对齐审查项（§6）。
  本登记是双轨差登记的**正确范本**。
- **运行时 actions 实证**：Qwen3.5-0.8B（width=6144、kernel=4、乘数 3）
  matrix=9,437,184=128·6144·4·3、actIn=actOut=4,718,592；Kimi-K3（width=36,864）
  matrix=18,874,368、actIn=9,437,184（verify-vision-misc §3.6）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### linear_attention — Gated Linear Attention（generic 槽位）

- **触发面**：**0/59 模型**（探针无一叶；trigger-map 无键 + 全 59 模型全树 operator
  扫描 0 叶）——generic plain 变体的通用槽位保留
  （`index.js:182-191` ref 注释明示「目录 0 模型发射此叶」）。现网 KDA 家族
  （Qwen3.5/3.6/3.8、Kimi-K3、GLM-5.3-Flash）全部经 `gated_delta_attention` 叶。
  运行时 case 按节点路径分派：`/short_conv|conv/` → 卷积分支、`/state|recurrent/` →
  state 分支（`extractor.js:645-663`，case "linear_attention"），当前无可达叶。
- **matrix**（registry 规格）：plain 递推 `2·T·Nh·dk·dv`（外积 + query）。规格实现：
  `linearAttentionStateCounts({delta:false})`（`counts.js:140-152`）。
- **vector / sfu**（registry 规格）：vector=`T·Nh·dk·dv`（decay 乘）；sfu=`Nh·T`
  （exp decay）。
- **bytes**（registry 规格）：weights=0；actIn=`2·T·Nh·dk·dv·b`、actOut=`T·Nh·dk·dv·b`
  （递推状态读+写主导）。
- **基础分解**：matrix = 2×per-token 批量 `matmul`；状态访存登记同 gda。
- **来源**：二等 modeling 对照（Gated DeltaNet arXiv 2412.06464 递推语义，generic
  plain 变体）+ 缺失声明（目录 0 模型）（`index.js:184-185`）。
- **全局假设**：A6（per-token 递推下界；chunked 总量等价）。
- **已知近似/登记**：槽位保留，无现网触发。**运行时口径差（G2 ④）**：运行时
  `/state|recurrent/` 分支实际调用 `stateUpdateCounts`（`extractor.js:659-661`）——
  generic 模式 matrix=`1·T·kh·vh·dk·dv`、vector=0、sfu=0、bytes=stateBytes，**与上方
  F7b plain 公式不同**。当前 0/59 不可达、无现网影响；若未来 plain 变体接入，须先
  裁决口径（建议：手搓分支改调 `linearAttentionStateCounts` 使两轨收敛；或本节按
  运行时改写）。（2026-09-09 双向验证登记，verify-vision-misc 修正 2。）
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### linear_attention_gate — Linear Attention Output Gate（generic 槽位）

- **触发面**：**0/59 模型**——通用槽位保留（`index.js:192-202`：「目录 0 模型发射
  此叶」）。KDA 输出门 z·y 路实际经 `gated_rmsnorm` 叶（FusedRMSNormGated 语义）。
- **matrix**：精确零。
- **vector / sfu**：`vector=T·W`、`sfu=2·T·W`。实现：`gateCounts`
  （`counts.js:84-95`；extractor 四门控共用 case `extractor.js:593-597`）。
- **bytes.weights / actIn / actOut**：weights=0；actIn=actOut=`T·W·b`。
- **基础分解**：`elementwise`。
- **来源**：二等对照 Kimi-K3 modeling_kimi_linear.py FusedRMSNormGated（门乘路）
  + 缺失声明（`index.js:194-196`）。
- **全局假设**：A5。
- **已知近似/登记**：槽位保留；运行时若触发走 default→registry（无手搓 case），
  口径无冲突。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### qwen_qkvz_split — Qwen GDN QKVZ Split

- **触发面**：31/59 模型（383 节点/1137 实例）：Qwen3.5/3.6/3.8 全系的 GDN 层
  （fused qkvz 投影拆 q/k/v/z，每 KDA 层组 1 节点、主流乘数 ×3（实例≈3×节点；
  Flash-Next 例外 14/36 = 11×3+3×1））。
- **matrix / vector / sfu / bytes**：全精确零（A1）。
- **基础分解**：`copy`（view 豁免，零流量）。
- **来源**：二等 modeling 对照（vLLM.Qwen3NextAttention.qkv_proj /
  SGLang.Qwen3_5Attention.qkv_proj，ops/index.js:275）（`index.js:376-385`）。
- **全局假设**：A1。
- **已知近似/登记**：z 旁路到输出归一化（gated_rmsnorm），不进 short conv
  （ops/index.js:174/193-202）。
- **运行时 actions 实证**：六元全 0 实测（verify-vision-misc §3.8）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### 4.3.6 DeepSeek V4 双 case — [B-attn-dsv4](#b-attn-dsv4)

##### dsv4_swa_attention — DeepSeek V4 Sliding-Window MQA

- **触发面**：3/59 模型——DeepSeek-V4-Flash / -0731 / -Vision-Exp（1 节点×2，L0-1
  层组）。compress_ratio=0 层（Pro 系无此叶）。
- **matrix**：`Nh·T_q·visible·(D+D)`，visible=`min(S, slidingWindow)`（prefill）；
  **decode available=1 是 legacy 行为**（`extractor.js:135-147`，
  legacyDeepseekV4AttentionMacs 镜像 :140-145，有意保留）。
  实现：case（`extractor.js:509-542`，case "dsv4_swa_attention"）。
- **vector / sfu**：精确零。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`(Nh·T_q·D + kvH·K_w·D + 2·scores)·b`（Q 读 + KV
  窗口 latent 读**一份**（K/V 共享单一 headDim 宽 latent）+ scores/probs）；
  actOut=`(2·scores + Nh·T_q·dv + kvH·T_q·D)·b`（含 kvWrite，宽=D）。
- **单位与换算**：K_w=`min(S, slidingWindow)`（decode 时 S=上下文长，窗口语义自然
  成立）。
- **基础分解**：`matmul`×2 + `softmax`（融合核）。
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
- **运行时 actions 实证**：V4-Flash prefill S=128、×2 组：matrix=1,073,741,824
  （visible=min(128,128)）、actIn=actOut=12,713,984 B，逐位吻合
  （verify-attention §2.10）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

##### dsv4_compressed_attention — DeepSeek V4 Compressed MLA

- **触发面**：5/59 模型——V4-Flash 系 20 节点/模型（L3,5,…,41 ratio=128 层）、
  V4-Pro 系 30 节点×31（L0-1 组 ×2 + 奇数层）。全 120 节点/122 实例。
- **matrix**：`Nh·T_q·ceil(S/ratio)·(D+D)`（legacy 镜像，decode available=1 同
  swa 层，`extractor.js:140-145` 有意保留）。实现：case
  （`extractor.js:543-575`，case "dsv4_compressed_attention"）。
- **vector / sfu**：精确零。
- **bytes.weights**：0。
- **bytes.actIn / actOut**（a4d709a 口径，fc99269 滑窗补记后）：
  actIn=`(Nh·T_q·D + 2·kvH·K_c·D + kvH·W·D + 2·scores)·b`（读压缩缓存：每压缩位
  K/V 态各 D，共 2·D；**另读一份未压缩滑窗 KV**，W=`min(S, slidingWindow)`）；
  actOut=`(2·scores + Nh·T_q·dv + kvH·T_q·D)·b`——**含滑窗 kvWrite**（新 token
  窗口写入与 swa 层同口径）；压缩态写入仍归 compressor=mla_kv_compress 叶
  actOut（防双计）。
- **单位与换算**：K_c=`ceil(S/ratio)`（decode 时 S=上下文长）；W=`min(S,128)`。
- **基础分解**：`matmul`×2 + `softmax`（融合核）。
- **来源**：二等 modeling 对照（compress_ratio=128，compressor 输出宽 2·D 实证：
  ops/index.js:387 + memory.js (2·D)/ratio 摊销）+ 降级声明同 dsv4_swa_attention；
  滑窗混合读依据 fc99269（vLLM c128a = 压缩历史 + 原始滑窗 [t-128,t]，官方博客
  实证；原取证 /tmp/m11-formulas/dsv4-sliding-window.md 已不在盘，按 commit
  message 登记）；/tmp/m11-formulas/dsv4.md §(b)6。
- **全局假设**：A2。
- **已知近似/登记**：~~压缩层是否同时读原始滑窗 KV 未定~~ **已裁决并落码
  （2026-09-08，fc99269）：hybrid 读成立**，actIn 补 `kvH·min(S,W)·D`、actOut 补
  `kvH·T_q·D`；`attn_sink` 每层 ≈Nh 参数未计（量级可忽略）；关联登记（memory.js
  侧，fc99269 P1）：dsv4 分支滑窗容量按 per-token 无界计、应封顶
  `headDim·min(T,128)`——属 rate 模型形态修改，另波处理。
- **运行时 actions 实证**：V4-Flash prefill S=128（K_c=1、W=128）：actIn 8,554,496 B、
  actOut 8,552,448 B，与公式逐位吻合（verify-attention §2.11）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

**一致性断言（dsv4）**：两 case matrix = legacy 镜像（构造恒等）；C4 层 bytes 三处
cache 写账目自洽（压缩态=compressor 叶 actOut、滑窗写=attention 叶 actOut 末项、
qsa_attention dsv4 分支 kvWrite=0——C2 修复后无三重计费）。

### 4.4 MLP / MoE 子块 — [B-layer-mlp](#b-layer-mlp) / [B-layer-moe](#b-layer-moe)

#### swiglu — SwiGLU

- **触发面**：59/59 模型（2161 节点/5774 实例）两个形态：
  - **dense MLP**（多数节点）：每 MLP 层一叶，vector/sfu/bytes 生效；
  - **routed FFN 压缩叶**（`expert_mlp`，MoE 专家主干）：matrix 段在此
    （探针 matrix>0 节点 926 个即 routed 形态），激活段 tokens×k。
- **matrix**：dense=精确零；routed=`T·k·3·EH·EI`（gate/up/down GEMM 融合语义，
  EH=expertHidden、EI=expertIntermediate）。实现：case（`extractor.js:610-632`，
  case "swiglu"，ROUTED_EXPERT_RE 结构化路径判据 `extractor.js:37`）。
- **vector / sfu**：`vector=2·T_eff·I`（silu+mul）、`sfu=2·T_eff·I`（sigmoid=2 SFU）；
  dense 时 T_eff=T、routed 时 T_eff=T·k。
- **bytes.weights**：0（专家 GEMM 权重经参数量/权重字节链计；激活叶不重复计）。
- **bytes.actIn / actOut**：actIn=`2·T_eff·I·b`（gate/up 两路输出读）、
  actOut=`T_eff·I·b`。
- **单位与换算**：silu = x·sigmoid(x)：2 SFU + 1 mul（A5）。EH=`latent_size ||
  routedExpertHiddenSize || hiddenSize`（K3 latent MoE 经 attributes.latent_size=3584
  正确取 latent，`extractor.js:616`）；EI=`moeIntermediateSize || intermediateSize`。
- **基础分解**：`elementwise`（silu·mul）+ routed 的 `matmul`×3。
- **来源**：一等 `aten::silu` + `aten::mul`（`index.js:134-143` ref 注释）。
- **全局假设**：A7（fused gate+up 按语义分解，融合收益记 implementation）。
- **已知近似/登记**：routed 计数公式 `T·k·3·EH·EI`（按 k 而非 k/E——旧链 ·(k/E)
  少乘 E 是已知双链 bug，2026-09-07 修正，`extractor.js:618-620` 注释）；
  per-expert 展开树若未来出现需回改并依赖 walker 乘 E。
- **运行时 actions 实证**：vector/sfu/bytes 与 F5 逐位吻合（verify-moe-norm §3.2）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### topk — TopK Routing

- **触发面**：44/59 模型（916 节点/2565 实例）：全部 MoE 模型的普通路由层
  （V4 的 hash 层除外——见 dsv4_hash_route；单节点×层数形态如 M2.7 1×62、
  K2 系 1×60，Qwen 系每 MoE 层显式节点）。
- **matrix**：精确零。
- **vector / sfu**：vector=`T·E`（比较/选择诚实计）、sfu=`T·k`（norm_topk_prob
  除法；normTopkProb=false 时 0）。实现：`topkCounts`（`counts.js:159-166`，
  `topkCounts`）；case（`extractor.js:666-667`，case "topk"）。
- **bytes.weights**：0（router 权重由 linear 叶计）。
- **bytes.actIn / actOut**：actIn=`T·E·b`（router logits 读）、actOut=`T·k·b`
  （expert weights/ids 写）。
- **基础分解**：**topk 原语**（词汇表外——比较/选择无法由 gather/add 组合忠实复现，
  诚实计登记）。
- **来源**：一等 `aten::topk`（flop_counter 不数比较选择——vector 按 T·E 诚实计）
  （`index.js:144-153` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：无。
- **运行时 actions 实证**：normTopkProb 接线逐字核验（verify-moe-norm §3.3）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### moe_dispatch — MoE Dispatch

- **触发面**：44/59 模型（926 节点/2580 实例），与 topk 同集合。
- **matrix / vector / sfu**：全精确零（gather 纯搬运）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`T·W·b`（payload 读）、actOut=`T·k·W·b`（k 份专家
  输入写）。**W = dispatch payload 宽，按节点 IO 形状取（C1 已落码，a4d709a
  `extractor.js:668-669`：`hidden: staticWidth(node?.input_shape) || config?.hiddenSize
  || 0`）**：标准 MoE 与 V4 = H（`dims.expertInput=[-1,H]`）；Kimi-K3 latent MoE =
  latent（`routed_expert_hidden_size=3584`，节点 input_shape=[-1,latent]，
  `ops/index.js:841-846`，链路 `routed_expert_down_proj → dispatch`）。
- **单位与换算**：dims 按**每专家 payload** 声明、未摊 k（不会双计 topk）。
- **基础分解**：`gather`（G1 符合项）。
- **来源**：一等 `aten::index_select`（gather 纯搬运，零计算）（`index.js:154-162`）。
- **全局假设**：无特别引用。
- **已知近似/登记**：~~payload 宽暂取 hiddenSize 的 latent-MoE 偏差~~ **已修复
  （C1，a4d709a）**：修复前传 `config.hiddenSize`，对 K3 各放大 H/latent=2×
  （92 层实例 ≈ +21.6MB/token @decode，bpe=2；2026-09-09 双向验证发现，
  verify-moe-norm F1）；修法 W=staticWidth(node.input_shape)||H，golden/恒等式基线
  已同步重生（`cost-memory-actions.golden.json` 42 行翻新）。
- **运行时 actions 实证（a4d709a）**：Kimi-K3 单实例（T=128、latent=3584、k=16）：
  actIn=917,504 B=128·3584·2、actOut=14,680,064 B=128·16·3584·2——latent 宽逐位
  吻合（/tmp/m11-formulas/probe-c12-rewrite.mjs 实测；修复前 actIn=1,835,008）。
  V4-Flash：actIn=1,048,576=128·4096·2（标准宽不受影响）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### moe_combine — MoE Combine

- **触发面**：44/59 模型（926 节点/2580 实例），与 topk 同集合。
- **matrix**：精确零。
- **vector / sfu**：vector=`2·T·k·W`（乘 + 累加，2026-09-07 规格修正的诚实数学），
  sfu=0。W = 专家输出 payload 宽（同 moe_dispatch 的 W：标准/V4=H、Kimi-K3
  latent=3584；C1 已落码，a4d709a `extractor.js:670-671`）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`(T·k·W + T·k)·b`（专家输出 + 权重读）、
  actOut=`T·W·b`（scatter + 加权合并写）。实现：`moeCombineCounts`
  （`counts.js:175-182`，`moeCombineCounts`）。
- **基础分解**：`gather`/scatter + `elementwise`(加权累加)。
- **来源**：三等分解声明 scatter + 加权合并（`index.js:163-172` ref 注释）。
- **全局假设**：A5（逐 flop）。
- **已知近似/登记**：payload 宽 latent-MoE 偏差**已修复（C1，同 moe_dispatch）**；
  修复前 vector 与 bytes 各放大 2×（≈ +10.6Mflop、+21.6MB 每 token @decode）。
- **运行时 actions 实证（a4d709a）**：Kimi-K3 单实例：vector=14,680,064=
  2·128·16·3584、actIn=14,684,160=(14,680,064+2,048)、actOut=917,504——latent 宽
  逐位吻合（probe-c12-rewrite 实测）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### moe_add — MoE Branch Add

- **触发面**：41/59 模型（873 节点/2422 实例）：结构中物化了 shared expert /
  双分支合并的 MoE——DeepSeek 系 8、Kimi 系 8、GLM 系 9、Qwen3.5/3.8 MoE 14、
  M3 系 2。无此叶的 3 个模型：MiniMax-M2.7（`shared_intermediate_size=0`，
  真无 shared 分支）；Qwen3.8-Flash-Next×2（modeling 有 shared 分支——
  `modeling_qwen4_exp.py:984-996`，config 有 `shared_expert_intermediate_size=640`
  但无 `n_shared_experts` 键 → `normalize.js:12`（SHARED_EXPERT_KEYS）取空 →
  `moeModule`（`layers/moe.js:20-25,32`）不物化 shared_experts 与本叶、仅物化
  shared_expert_gate；合并加法欠覆盖 + gate/分支不对称，均登记待修——
  2026-09-09 双向验证发现，verify-moe-norm F2）。
- **matrix**：精确零。
- **vector / sfu**：vector=`T·H`。实现：`addCounts`（`counts.js:184-191`；
  `extractor.js:672-673`，case "moe_add"）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`2·T·H·b`（routed+shared 两路读）、actOut=`T·H·b`。
- **基础分解**：`add`。
- **来源**：一等 `aten::add`（`index.js:173-181`）。
- **全局假设**：无特别引用。
- **已知近似/登记**：cost_counts.md「结构级缺口」：decoder 层普通残差加法（+x）
  无算子节点，2TH·b×2/层未计——量化暂缓决定（§4.2.4）；Flash-Next 欠覆盖（上）。
- **运行时 actions 实证（a4d709a）**：Kimi-K3 单实例（H=7168）：vector=917,504=
  128·7168、actIn=3,673,216=2·128·7168·2、actOut=1,836,544（hidden 宽合并，
  `ops/index.js:867-870` dims.hidden，probe-c12-rewrite 实测）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### shared_expert_gate — Shared Expert Gate

- **触发面**：16/59 模型（426 节点/844 实例）：Qwen3.5-122B/35B-A3B/397B、
  Qwen3.6-35B、Qwen3.8-2.4T/Flash-Next（每 MoE 层一叶；Flash-Next 节点 26=22×2+4×1）。
- **matrix**：精确零。
- **vector / sfu**：vector=`T·W`、sfu=`2·T·W`（sigmoid）。实现：`gateCounts`
  （`counts.js:84-95`；`extractor.js:593-597`）。
- **bytes.weights**：0（W_g 融合在 shared expert MLP 或独立 linear 叶）。
- **bytes.actIn / actOut**：各 `T·W·b`。W 来自 2D 输出 `[−1,−1,3072]`，
  `staticWidth`=正维积=3072。
- **基础分解**：`elementwise`。
- **来源**：二等 modeling 对照（Qwen3.5/3.6 MoE shared expert sigmoid gate；
  Qwen modeling 未入库——normalize.js sharedExpertGate 字段驱动）+
  A5（`index.js:341-351` ref 注释）。
- **全局假设**：A5。
- **已知近似/登记**：Flash-Next×2「有 gate 无被门控分支」结构自洽性问题
  （同 moe_add 登记）。
- **运行时 actions 实证**：T=128、W=6144 对照组逐位吻合（verify-linear §1.8/§3.2）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### dsv4_hash_route — DeepSeek V4 Hash MoE Routing

- **触发面**：5/59 模型（10 节点/15 实例）：V4-Flash 系 2 节点×3（L0-1 组 + L2）、
  V4-Pro 系 2×3——前 num_hash_layers 层按 input_ids 查表路由，不走普通
  router logits + top-k（同模型其余 MoE 层仍走 topk 叶）。
- **matrix / vector / sfu**：全精确零（纯查表）。
- **bytes.weights**：`tableRows·b`，tableRows=`vocabSize·expertsPerToken`
  （V4-Flash：129280×6 ≈ 775,680 条目/层 ≈1.48MB——M11-P2 C 路接线，
  `extractor.js:674-683`，case "dsv4_hash_route"；权重 index 实证
  ffn.gate.tid2eid 恰 3 层 = num_hash_layers）。
- **bytes.actIn / actOut**：actIn=`T·b`（token id 读）、actOut=`T·k·b`
  （expert ids/weights 写）。实现：`hashRouteCounts`（`counts.js:194-199`）。
- **基础分解**：`gather`（查表）。
- **来源**：二等 modeling 对照（DeepSeek V4 hash MoE：input_ids 查表固定专家集合，
  vLLM tid2eid；actIn/actOut 已核实生效——/tmp/m11-formulas/dsv4.md §(d)）。
- **全局假设**：无特别引用。
- **已知近似/登记**：tid2eid 存储 dtype 未证实（int32 索引按 bpe=2 计——dsv4.md
  §(d) 登记项）；registry ref 注释仍写「weights 表条目接线待办」滞后（F8，非本文
  任务，登记）。
- **运行时 actions 实证**：weights/实例=1,551,360 B=775,680×2B ✓「≈1.48MB」✓；
  actIn/actOut 256/1,536 B（×2 组）逐位吻合（verify-attention §2.12）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 4.5 层间 / 多流残差槽位 — [B-layer-streams](#b-layer-streams)

> 本组为三等分解声明（A4/A7）：counts = 已知 F 函数组合，ctx 由
> `extractor.js:694-761` 的 11 个 ctxBuilder 构建；组合公式的人类可读展开如下。

#### mhc_pre — mHC Pre

- **触发面**：7/59 模型（292 节点/341 实例）：DeepSeek-V4×5（42-60 节点/模型，
  每 decoder 层一叶）+ GLM-5.3-Flash×2（23×45）。vLLM MHCPreOp。
- **matrix**：`= T·H·n`（streams→input 合成小矩阵，n=mhcNumResidualStreams）。
  实现：ctxBuilder 组合 `F4(mix) + F1([H,n]) + add(merge)`
  （`index.js:226-235`；`extractor.js:743-747`，ctxBuilders.mhc_pre）。
- **vector / sfu**：vector=`2·T·H`（sigmoid post mix + 合成加法）、sfu=`2·T·H`。
- **bytes.weights**：`H·n·b`。
- **bytes.actIn / actOut**：actIn=`4·T·H·b`（mix 读 + 矩阵输入 + add 双读）；
  actOut=`(2·T·H + T·n)·b`。
- **公式**：`p=sigmoid(M_a·s_a+b_a)+eps; C=Sinkhorn(softmax(M_c·s_c+b_c)+eps);
  x=Σ p_i·H_i`（softmax/Sinkhorn 的 SFU 细节并入 gate 段口径）。
- **基础分解**：`elementwise`(gate) + `matmul` + `add`；Sinkhorn 段并入 gate 口径
  （词汇表外登记）。
- **来源**：三等分解声明（vLLM MHCPreOp implementation 指针，layers/hybrid.js）
  （`index.js:228-230` ref 注释）。
- **全局假设**：A4、A5、A7。
- **已知近似/登记**：Sinkhorn 迭代次数未建模（按 softmax+归一一段计）。
- **运行时 actions 实证**：合成算术逐项复算一致（verify-moe-norm §3.5）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### mhc_post — mHC Post

- **触发面**：7/59 模型（7 节点/7 实例）：每模型**末层** 1 节点（与
  mhc_fused_post_pre 互补：末层用 post+contract，中间层用 fused）。
- **matrix**：`= T·H·n`（combine 小矩阵）。实现：`F1(combine) + add(inject)`
  （`index.js:246-255`；`extractor.js:748-751`）。
- **vector / sfu**：vector=`T·H`、sfu=0（post mix 在 pre 侧计）。
- **bytes.weights**：`H·n·b`。
- **bytes.actIn / actOut**：actIn=`3·T·H·b`、actOut=`(T·H + T·n)·b`。
- **公式**：`H'_j = post_j·x + Σ C_ij·H_i`。
- **基础分解**：`matmul` + `add`。
- **来源**：三等分解声明（vLLM MHCPostOp）（`index.js:248-249`）。
- **全局假设**：A4。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### mhc_fused_post_pre — mHC Fused Post + Pre

- **触发面**：7/59 模型（292 节点/341 实例）：与 mhc_pre 同集合同节奏——中间层
  每层一叶（上一层 post 与当前层 pre 的层间融合）。
- **matrix**：`= T·H·n`。实现：`F4(post) + add(inject) + F4(pre) + F1([H,n])`
  （`index.js:236-245`；`extractor.js:752-757`）。
- **vector / sfu**：vector=`3·T·H`（两次 gate + inject 加法）、sfu=`4·T·H`。
- **bytes.weights**：`H·n·b`。
- **bytes.actIn / actOut**：actIn=`5·T·H·b`；actOut=`(3·T·H + T·n)·b`。
- **公式**：`(H',post',C',x') = MHCPre(MHCPost(x,H,post,C); F,scale,base)`。
- **基础分解**：`elementwise`×2 + `matmul` + `add`。
- **来源**：三等分解声明 = mhc_post + mhc_pre 融合（vLLM MHCFusedPostPreOp）；
  A7：融合收益记 attributes.implementation，不折算流量（`index.js:238-240`）。
- **全局假设**：A4、A7。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### mhc_contract — mHC Contract

- **触发面**：7/59 模型（7 节点/7 实例）：每模型末层 1 节点。
- **matrix**：精确零。
- **vector / sfu**：vector=`T·H`（n 流平均）、sfu=0。实现：`add(contract)`
  （`index.js:256-264`；`extractor.js:758-760`）。
- **bytes.weights**：0。
- **bytes.actIn / actOut**：actIn=`2·T·H·b`（n 流读按二路一阶近似）、actOut=`T·H·b`。
- **公式**：`h = (1/n)·Σ H_i`。
- **基础分解**：`add`。
- **来源**：三等分解声明（GLM-5.3-Flash 末层 HCContract 语义）（`index.js:258-259`）。
- **全局假设**：A4。
- **已知近似/登记**：n 流读按 addCounts 的 2 路口径近似（n>2 时读放大未折算——
  一阶口径声明）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### hyper_connection — Hyper Connection

- **触发面**：2/59 模型——Qwen/Qwen3.8-Flash-Next / -FP8（53 节点×97）。
- **matrix**：`= T·H²`（W_down/W_up 小矩阵 [H,H]）。实现：ctxBuilder 组合
  `F3(grouped) + F5(mix) + F1(mixers) + F4(gate) + add(combine)`
  （`index.js:320-330`；`extractor.js:729-735`）。
- **vector / sfu**：vector≈`8·T·H`（norm 4H + silu 2H + gate H + combine H）；
  sfu≈`4·T·H`（silu 2H + gate 2H）+ T（rsqrt）。
- **bytes.weights**：`(H² + H)·b`（mixer 矩阵 + norm weight）。
- **bytes.actIn / actOut**：actIn≈`7·T·H·b`；actOut≈`5·T·H·b`（各分量读写一次之和，
  以 sumCounts 实算为准）。
- **公式**：`x_n=GroupedRMSNorm(H); l=SiLU(W_down·x_n); gate=W_up·l;
  block_input=GateMix(x_n,gate); H'=Combine(H,block_output,injection)`。
- **基础分解**：`rmsnorm` + `elementwise`(silu) + `matmul` + `elementwise`(gate) +
  `add`。
- **来源**：三等分解声明（Qwen4Exp delayed HyperConnection，layers/hybrid.js）。
  **ref 注释滞后提醒**：注释写「Qwen modeling 未入库，离线取证」，但
  models/Qwen/Qwen3.8-Flash-Next/modeling_qwen4_exp.py 已于 2026-09-08 入库
  （evidence-qsa-glm.md §1）——来源可升二等，ref 注释待更新（只读发现，未改代码）。
- **全局假设**：A4、A5、A7。
- **已知近似/登记**：ctx 的 n 流参数用 normalized 近似（`extractor.js:686-687`
  「精度为初版，T4 恒等式校准后复核」）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### ple — Position Learning Enhancement

- **触发面**：2/59 模型——Qwen3.8-Flash-Next / -FP8（1 节点×1，指定层）。
- **matrix**：`= T·2·E_p·H + T·E_p·ngram`（W_kv 投影 + dilated short conv，
  E_p=pleEmbedDim）。实现：ctxBuilder 组合 `hashRoute(embed) + F1(kv) + F3(norm)
  + F7a(conv) + add`（`index.js:331-340`；`extractor.js:736-742`）。
- **vector / sfu**：`T·H`（add）+ `4·T·E_p`（norm）+ `T·E_p`（conv）；
  sfu=conv 段 `2·T·E_p` + norm 段 `T`。
- **bytes.weights**：`(2·E_p·H + E_p + E_p·ngram)·b`。
- **bytes.actIn / actOut**：hash 查表 actIn=`T·b`、actOut=`T·b`（tableRows=0：ngram
  表不计 weights）；其余分量读写一次之和。
- **公式**：`e=HashNGram(input_ids,context); [k,v]=W_kv·e;
  y=ShortConv(GatedNorm(k,v,RMSNorm(H)))`。
- **基础分解**：`gather`(hash) + `matmul` + `rmsnorm` + `conv` + `add`。
- **来源**：三等分解声明（Qwen4Exp PLE，layers/hybrid.js；同 hyper_connection 的
  ref 滞后提醒——modeling_qwen4_exp.py 已入库）。
- **全局假设**：A4、A5。
- **已知近似/登记**：ngram embedding 表（HashNGram 词表）无参数计费（tableRows=0
  ——表规模未建模，登记）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

### 4.6 view 零流量组（split 家族）——跨槽位

> **豁免理由（零是精确陈述）**：四叶均为 fused projection 输出的 strided view 拆分，
> 不发生数据拷贝、不使用任何计算/访存单元——A1（2026-09-07 拍板）。显式登记为零
> 而非漏算（principles §3.3「0 与 null 区分」）；bytes 完整性棘轮
> `frontend/src/cost/__tests__/bytesCompleteness.test.js:22` 的 `VIEW_OPS` 集合即本组
> 显式豁免清单（新增豁免必须在该集合登记并写明理由）。
> 实现统一为 `rearrangeCounts()`（`counts.js:215-221` 无 copy 分支）+ extractor
> 共用 case（`extractor.js:604-609`，view 四 case）。
> **基础分解**：`copy`（view 语义——零流量豁免登记）。
> 运行时实测：四算子各抽叶 matrix/vector/sfu/bytes 六元全 0
> （verify-vision-misc §3.8）。

#### split — Fused Projection Split

- **触发面**：36/59 模型（605 节点/726 实例）三个发射点：
  ① **qwen35_full 层**的 `qkv_gate_split`（fused QKV+gate 拆 q/gate/k/v，
  `ops/index.js:277-280`，每 full 层 1 节点、乘数 1）——载体为 Qwen3.5 系 21 +
  Qwen3.6×4 + Qwen3.8 的 qwen3_5 系变体 4（2.4T×2、27B×2；实测 2.4T=69 linear+
  23 qwen35_full → 23n/23i）。**注意 Qwen3.8-Flash-Next（qwen4_exp）的 full 槽位是
  qsa_attention，不发射 split**（实测无此叶，2026-09-09 修正 3）；
  ② DeepSeek-V4×5 的 `qkv_split`（fused_wqa_wkv 拆 q_lora/kv latent，
  `ops/index.js:362-366`，42–60 节点、实例=节点+1（每模型恰一处 2 层组 ×2：
  Flash 系实测 42/43、Pro 系 60/61））；
  ③ MiniMax-M3×2 的 `qkv_index_split`（fused QKV+index 拆 main/index 分支，
  `ops/index.js:520-523`，2 节点×57+3）。
- **matrix / vector / sfu / bytes**：全精确零（A1 view 豁免）。
- **来源**：一等 `aten::split`（视图语义，flop_counter 无成本条目）
  （`index.js:57-65`）。
- **全局假设**：A1。
- **已知近似/登记**：split_sizes 由节点属性给出，不同模型分支数/宽度不同
  （`index.js:61` explanation）。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### attention_qkv_split — Attention QKV Split

- **触发面**：40/59 模型（41 节点/1147 实例）：通用 GQA/full 模板与全部视觉塔的
  fused QKV 拆分——MiniMax-M2.7 文本（×62）、GLM-4.7 文本（2×89+3）、各视觉塔
  （M3 ×32、Qwen ×12–27（0.8B 12、2B/4B 24、9B 及以上 27）、Kimi ×27、
  GLM-Flash ×24、V4-Exp ×32）。
- **matrix / vector / sfu / bytes**：全精确零（A1）。
- **来源**：二等 modeling 对照 models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py
  （fused QKV 拆 q/k/v 语义分支）（`index.js:386-395`）。
- **全局假设**：A1。
- **已知近似/登记**：无。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

#### qwen_qkvz_split — 见 §4.3.5（KDA 节内全模板）。

#### mla_kv_split — MLA KV Latent Split

- **触发面**：19/59 模型（221 节点/1124 实例）：MLA 文本塔的 latent/rope 拆分
  （DeepSeek-R1/V3.1/V3.2 2×61、Kimi 系 2×61 / K3 23×24、GLM-5 系 2-38×78、
  GLM-Flash 11×11）；与 mla_query_compress 集合恒等（程序化验证）。
- **matrix / vector / sfu / bytes**：全精确零（A1）。
- **来源**：一等 `aten::split`（`index.js:288-296`）。
- **全局假设**：A1（latent/rope 拆分零流量）。
- **已知近似/登记**：形状语义实证：K3 叶 in=576（=kv_lora 512+rope 64）out=512。
- **对齐状态**：- [ ] 用户已对齐（2026-09-__）

---

## 5. 占比统计（A100 五路时间，探针实测）

**方法**：`/tmp/m11-formulas/probe-rewrite.mjs`（只读仓库，a4d709a）——59 模型 ×
两相位（prefill T=128 / decode T=1、S=4096）跑 `computeNodeCosts` 聚合全叶动作向量，
过 `classifyRoofline`（`frontend/src/cost/roofline.js:50-113`）取五路时间
（matrix/vector/sfu/memory/comm）的 max。芯片 = `nvidia-a100-80gb-sxm`
（`frontend/src/cost/chips/public.js:17-46`）：bf16 312 Tflops → matrix
109.2 TMACs/s（η_flops=0.7、÷2）、vector 19.5 Tflop/s、sfu 4.875 Tops/s
（=vector/4）、HBM 2.039 TB/s → 1.835 TB/s（η_hbm=0.9）；comm=0（actions 无
commBytes，单机前向口径）。输出 `/tmp/m11-formulas/probe-rewrite-out.json`。

**总判定**（118 个模型×相位判定）：

| 相位 | matrix bound | memory bound | vector/sfu bound |
|---|---|---|---|
| prefill S=128 | 55 | 4（M3×2、Qwen3.5-0.8B×2） | **0** |
| decode S=4096 | 0 | 59 | **0** |

**五路时间占比**（share of dominant lane；prefill S=128；同 family 数字相同者为
量化/精度变体，行为一致；全 59 行明细见 probe-rewrite-out.json）：

| 模型（族） | bound(P) | memory% | matrix% | vector% | sfu% | bound(D) | D vec+sfu% |
|---|---|---|---|---|---|---|---|
| MiniMax-M2.7 | matrix | 43.8 | 100.0 | 0.48 | 0.49 | memory | 0.1 |
| MiniMax-M3 / -MXFP8 | memory | 100.0 | 92.1 | 2.68 | 6.84 | memory | 10.1 |
| Qwen3.5-0.8B / -Base | memory | 100.0 | 91.4 | 1.05 | 2.26 | memory | 2.8 |
| Qwen3.5-2B / -Base | matrix | 94.7 | 100.0 | 0.95 | 2.12 | memory | 3.0 |
| Qwen3.5-4B / -Base | matrix | 97.9 | 100.0 | 0.68 | 1.45 | memory | 1.7 |
| Qwen3.5-9B / -Base | matrix | 94.5 | 100.0 | 0.47 | 0.97 | memory | 1.2 |
| Qwen3.5-27B / 3.6-27B / 3.8-27B（各 3 变体） | matrix | 97.3 | 100.0 | 0.27 | 0.52 | memory | 0.4 |
| Qwen3.5-35B-A3B / 3.6-35B-A3B（各 4 变体） | matrix | 78.6 | 100.0 | 1.01 | 1.84 | memory | 2.9 |
| Qwen3.5-122B（3 量化变体） | matrix | 70.1 | 100.0 | 0.64 | 1.07 | memory | 1.5 |
| Qwen3.5-397B-A17B（3 变体） | matrix | 63.4 | 100.0 | 0.53 | 0.75 | memory | 1.0 |
| Qwen3.8-2.4T-A95B / -FP8 | matrix | 53.3 | 100.0 | 0.22 | 0.22 | memory | 0.0 |
| Qwen3.8-Flash-Next / -FP8 | matrix | 76.1 | 100.0 | 0.82 | 1.50 | memory | 2.0 |
| DeepSeek-R1 / V3.1 | matrix | 51.0 | 100.0 | 0.25 | 0.26 | memory | 0.1 |
| DeepSeek-V3.2 | matrix | 53.0 | 100.0 | 0.23 | 0.20 | memory | 0.0 |
| DeepSeek-V4-Flash 系 3 | matrix | 75.8–77.1 | 100.0 | 0.13 | 0.25 | memory | 0.0 |
| DeepSeek-V4-Pro 系 2 | matrix | 77.0 | 100.0 | 0.06 | 0.12 | memory | 0.0 |
| Kimi-K2 系 4 | matrix | 40.3 | 100.0 | 0.27 | 0.23 | memory | 0.1 |
| Kimi-K2.5/2.6/2.7 | matrix | 44.6 | 100.0 | 0.47 | 0.76 | memory | 1.6 |
| Kimi-K3 | matrix | 58.1 | 100.0 | 0.29 | 0.44 | memory | 0.4 |
| GLM-4.7 | matrix | 56.8 | 100.0 | 0.38 | 0.32 | memory | 0.1 |
| GLM-5 系 6 | matrix | 51.1–51.2 | 100.0 | 0.23 | 0.20 | memory | 0.0 |
| GLM-5.3-Flash / -BF16 | matrix | 65.6 | 100.0 | 0.30 | 0.56 | memory | 0.2 |

**统计口径汇总**：prefill memory lane share 均值 71.5%（40.3%–100%）、matrix share
均值 99.4%（91.4%–100%）、vector+sfu 合计 mean 1.55% / max 9.53%（M3）；decode
memory share 59/59 = 100%、vector+sfu mean 1.28% / max 10.08%（M3）。

**vector/sfu 贡献占比（回应「bound 是否会落 vector/sfu」）**：
**不会**——118 个判定中 0 个 vector/sfu bound；即便只看 vector/sfu 时间自身的
量级，最大单模型（M3，prefill）也仅占主导路的 9.5%（其 bound 为 memory）。
全语料 prefill 主导路总时 2.104 s 中，vector/sfu 贡献第一的是 **softmax**
（0.0193 s，0.92%——唯一 O(Nh·T·S) 平方增长的 vector/sfu 项，decode 长上下文下
进一步放大），其后 swiglu 0.169%、vision_activation 0.099%、moe_combine 0.070%、
rmsnorm 0.051%、gated_rmsnorm 0.043%；gda/causal_conv1d/matmul/rope 等贡献 ≈0
（rope 有 vector 但占比 0.0005%；matmul/split 家族/gda/moe_dispatch 精确零）。
**工程含义**：vector/sfu 计数偏差不影响 bound 分类与五路时间结论，但 softmax/swiglu
仍是 SFU-heavy 核（flop_counter 不数 softmax——本仓的有意超越），对齐时保持
A2/A5 口径优先。

---

## 6. 双轨现状与护栏（§3.1b）

**现状**：42 条 registry 条目与 extractor 手搓分支**同名双轨**。运行时分派顺序：
`countsForNode` 的 type 分支（attention/embedding，`extractor.js:329-351`）→ 手搓
switch case（提前 return）→ default 走 registry `entry.counts(ctxBuilder())`
（11 个 ctxBuilder，`extractor.js:694-761`）。**手搓 case 的注册表 counts 引用是
护栏认证过的死引用**——两轨对同一 operatorId 可能口径不同，已登记实例（G2）：
causal_conv1d 的 vector/sfu/weights、matmul 的 F2 融合 vs 分解、swiglu 的 routed
分支、gated_delta_attention 的 vector/sfu、linear_attention 的 state 分派。

**护栏**：`scripts/check_principles.sh:65-86`（§3.1b 运行时接线判据）保证每条
FORMULAS 运行时可达——**手搓 case / ctxBuilder / 显式豁免三选一**，实测 42 条可达
（手搓 31 / ctxBuilder 11 / 豁免 0）。配套：§3.1d ref 注释 42/42；§3.1c bytes
完整性棘轮（全 leaf 三访存分量不得全零，view 豁免除外）。长期方向 = 手搓分支逐条
搬入 counts.js、消双轨（refactor_plan.md「护栏 §3.1 改运行时判据」；M11.5 的共享
bytes 助手抽取）。

**对齐审查时的判读规则**：本文各节「实现」字段给的是**运行时真实生效**的位置
（a4d709a 行号）；registry 条目行号（index.js）是规格与 ref 来源的权威锚点。
两者冲突时以运行时为准登记差异（如 causal_conv1d、gda、linear_attention——G2）。
**C1/C2 落码状态**：moe payload 宽（`extractor.js:668-671`）与 dsv4_sparse_mla
kvWrite（`extractor.js:453-455`）已按验证结论修复，golden 基线同步重生
（a4d709a）。

---

## 7. 自检清单

- [x] **双向表交叉完整性**：表 A 42 行（40 registry + embedding + attention 容器）
      逐行核对——每行槽位锚在表 B 存在；表 B 18 行（模型级 4 + 层内 14）逐行核对——每行算子序列的
      每个算子在表 A 有行且槽位回指一致（split 家族 4 条与 linear 以跨槽位多锚
      标注）。表 B 缺口标记 3 类（普通残差加、M3 无 embed、Flash-Next shared 分支）
      均在对应算子节登记。
- [x] 42 条 registry 条目逐一覆盖：探针现身 40 条 + 零触发槽位 2 条
      （linear_attention、linear_attention_gate，§4.3.5）= 42。
- [x] 结构节点覆盖：embedding（§4.1）、attention 模块容器（§4.3 头注）、
      split 家族 4 条（§4.6，VIEW_OPS 豁免登记）。
- [x] 探针实证：59/59 模型 prefill+decode 两相位跑通，unknown 叶 = 0；
      leaf 算子键 41 种，全部能在本文找到对应节。a4d709a 复跑触发面与冻结
      trigger-map.json 0 差异（probe-rewrite.mjs）。
- [x] 每个公式给出实现位置：counts.js F1-F9（15 个函数）+ extractor 31 case +
      11 ctxBuilder，均带 file:line + 符号锚（a4d709a）。
- [x] 触发面数字全部来自 computeNodeCosts 探针（§1 方法），未用代码推断代替；
      与 ref 注释中的在案数字交叉核对一致。
- [x] 来源三级标注逐条引用 index.js 的 ref: 注释与证据 file:line；降级声明
      （dsv4 双 case、MiniMax-M3 transformers 库版、V3.2/V4 离线取证）显式标注；
      两处 ref 注释滞后（hyper_connection/ple 的「Qwen modeling 未入库」、
      dsv4_hash_route 的「weights 接线待办」）已标注待更新。
- [x] 四份验证报告的修正全部落文：linear 节点数下限 4（M2.7）、linear vector 主链
      精确零、weights 兜底链、attention_qkv_split Qwen ×12–27、W=staticWidth 定义
      （含 mla_output_gate K3 W=12288 反例）、split ② 实例=节点+1 与 ① Flash-Next
      缺席、qwen_qkvz_split 每层组措辞、qsa_indexer/minimax_sparse_indexer 的 F2
      带入三项、dsv4_compressed 滑窗已裁决落码、rope GLM-5.3-Flash 非零更正、
      matmul Kimi ×61、mla_query_compress vector/sfu 措辞、moe_add 负例归因
      （Flash-Next 接线缺口）、gda vector/sfu 运行时精确零、linear_attention 运行时
      口径差、vision_merge T_v 缺口、embedding 第二入口不可达、KDA 'kimi' 模式
      无现网载体；C1（moe payload=staticWidth）/C2（dsv4 kvWrite=0）按落码后口径
      书写并附 a4d709a 实测数值。
- [x] §5 占比统计回应「bound 是否会落 vector/sfu」：118 判定 0 个 vector/sfu
      bound（prefill 55 matrix + 4 memory；decode 59 memory）。
