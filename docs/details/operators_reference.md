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

探针**就是生成器本身**（不再有 `/tmp` 一次性脚本）：

```bash
node scripts/gen-operators-reference.mjs           # 重写本文机器段
node scripts/gen-operators-reference.mjs --check   # 只比对，diff 非空退出 1（= npm run docs:check）
node scripts/gen-operators-reference.mjs --json    # 探针原始数据打到 stdout（不落文件、不进 golden）
```

`--json` 的结构：`overview`（47 算子 × 59 模型的触发面 + 三分量非零面 + 占比原始值）·
`classes`（16 结构类 × 逐算子 × 两相位的三分量 / bytes 三分量 / AI / bound / 模块恒等式结果）·
`chip`（bound 判定用的参考芯片）· `overviewPhases` / `classPhases`（两段各自的工作点）。

管线（与生成器同源）：`models/catalog.json` 全部 59 模型：config → `normalizeConfig`
→ `resolveArchitecture` → `buildNetwork` → `createStructureIr` → `materializeModelStructure`
→ 逐叶 `countsForNode(node, { config, options, path, bytesPerElement: 2 })`，
逐叶收集 `attributes.operator_id`（`type==='embedding'` 记为结构节点 `embedding`）。

工作点分两套，因为用途不同：

- **总览表 / 占比表**：prefill `T=128`、decode `T=1, S=4096`。用于触发面与相对占比。
- **逐结构类明细表**：prefill `T=S=2048`、decode `T=1, S=4096`。bound 是 arithmetic
  intensity 与 ridge point 的比较结果，`T=128` 下 59 个模型全落 memory，表就失去查错价值；
  这一套与 `modelIdentities.test.js` 的 bound 断言同工作点，表与断言口径一致。

统计口径：

- **leaf** = 无 children 的节点；父节点（模块/容器）不携带动作向量
  （`frontend/src/cost/compute.js:43-48`，aggregate 链由子节点累加）。
- **节点数 vs 实例数**：节点数 = 结构树中 leaf 出现次数；实例数 = Σ multiplier
  （层组 repeat 倍乘，`frontend/src/cost/traverse.js:40-55`）。下文记作「节点/实例」。
- **相位**：两相位均跑过，unknown 叶皆 0（与
  `frontend/src/structure/__tests__/builtinModels.test.js:54-55` 的 `computeComplete`
  断言一致）。
- **历史取证引用**：文中还有 15 处指向 `/tmp/` 下早期会话临时取证文件的路径，
  多数已不在盘（相应位置已标注），结论都已内联到各节。护栏 §3.5b 对这个计数做了
  棘轮**只许下降**：新证据请落 `models/<org>/<id>/` 证据库
  （`scripts/fetch-evidence.mjs` + manifest），或者改写成「跑生成器就能复现」。

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

> **口径分工**：本节是**带注解的人工视图** —— 提供 §4 正文的锚链接、attention kind 归类和
> 阅读顺序。覆盖面（哪个算子出现在哪些槽位、槽位里的算子序列）以文末**生成段的双向表 A/B**
> 为准，那两张表由 `node scripts/gen-operators-reference.mjs` 从 59 模型实跑得出，
> `npm run docs:check` 会守住它不漂移。两处冲突时改本节，不改生成段。

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
- [x] **本节以下的机器段接管覆盖面对账**：总览表、占比表、注册表↔触发面对账、
      双向表 A/B、模块层分解台账全部由 `scripts/gen-operators-reference.mjs` 生成，
      `npm run docs:check`（node 测试 #189）守住不漂移。§3 与 §4 保留为带锚的人工视图。
- [x] **四条恒等式全部 error 模式、全部容差 0**（`npm test` 302 项）：
      - 融合分解：计算三分量逐位相等 + bytes 两侧夹逼，308 组 **0 不闭合**
      - 权重字节：32/32 行 **逐字节相等**，登记表空
      - 激活流**形状**连续性：33434 条声明边，末维匹配 29180，其余落在已登记的
        语义边类（control / slice / fused-in / concat / entry / regroup），登记表只许缩短
      - KV 读量：按层分桶的**逐层 cache 容量对账**。读全 cache 的层
        （gqa/mha/mla/qwen35_full）`kvRead` **逐字节等于**逐层容量×S（10/16 结构类有该桶）；
        选择性读的层（top-k / 块稀疏 / 压缩+滑窗）`kvRead ≤ 容量×S`；
        indexer 的 `indexRead ≤ 自己那份 index-k cache 容量×S`。无容差、无登记表。
- [x] **逐层归因工具**：`node scripts/diff-weight-identity.mjs --phase decode <modelId>`
      把权重字节差额摊到「第几层 / 哪个算子」，替代早期靠代数反解 + 单层变体二分的
      手工流程。权重字节这条从 1% 收到 0 的 20 余处修正全部由它定位。
- [x] **tid2eid 分类裁决落地**（2026-09-09 联网取证）：tid2eid 是 **buffer 不是参数**
      （NVIDIA Megatron-Bridge 文档明文 "Buffers are not parameters"；MaxText 同；
      出处 = Hash Layers, Roller et al. 2021）。哈希层的 gather 叶 `bytes.weights = 0`
      （与 embedding 表同待遇），表的常驻容量（vocab·k·4B int32）由
      `derivedBufferBytes` 计入显存（memory 面板新增 Buffers 行）。
      `dsv4_hash_route` 的分解 = 一个 gather 原子，DECOMPOSE_PENDING 13 → 12。
- [x] **fp32 参数登记表**（`formulas/paramDtypes.js`，2026-09-09 全量落地）：
      vLLM 显式声明 torch.float32 的参数**全部**按 4B 计，两侧同源 ——
      GDN/KDA 的 dt_bias + A_log 经 `gdnDecayElements(config, mode)`（四个家族 +
      generic）、mHC 的 fn/base/scale（大矩阵 hc_*_fn 经 linearCounts 的
      `weightBytesPerElement`），其余参数跟随 torch_dtype。checkpoint 证据在场时
      以 truth/skeleton 的逐 tensor weight_dtypes 为准（即 safetensors 头部机制）。
- [x] **量化容量缺口：swiglu 携带的专家 GEMM 未进量化枚举**（2026-09-09
      N2-3 显形；**探针实测爆炸半径远超初判**：不止 M2.7/M3——全部量化
      MoE 模型的路由专家权重都挂在 swiglu 叶，V4-Pro 1.55e12 /
      Qwen3.8-2.4T 2.37e12 / Kimi-K2 系 1.02e12 参数未进枚举，均按 bf16
      计 → 量化 MoE 模型容量普遍 ≈2× 偏高，专家块是主导项）。
      **2026-09-09 三波收官销案（完整方案与执行记录见
      [details/sharding_matrix.md](sharding_matrix.md)**——三层设计 +
      三消费者接线 + 四验收锚）：
      ① **W-A 拆 operator id**：专家融合叶独立 id `fused_moe_mlp`（对标
        vLLM `FusedMoE`，counts 逐位迁移，registry 49 条）+ weightMatrices
        声明层 1（统一助手 weightMatrixDecl；专家叶 ep 组 + attention/dense
        linear 叶 tp 组）；锚 1 全目录 5233 声明叶容差 0（棘轮表）；
      ② **W-B 三消费者接线**：量化枚举/逐卡投影/容量分桶读同一声明
        （声明优先、规则表回退）；计划轴 moe_tp/moe_ep + sharding.js
        组合语义纯函数（EP=TP×DP、无 EP 时 DP 切专家、混合 ETP）；
        锚 2 M2.7 三方一致（棘轮表）；锚 3 无声明叶逐位不变；
      ③ **W-C 覆盖核实 + 第四量化方案**：量化 MoE 家族声明逐模型核实
        （13 家族 / 30 模型，下表）；compressed-tensors 入枚举（w4a16 +
        mxfp4，Kimi K2 系/K3 派生路径此前按 0.5B 标量宽整模型计——
        attention/shared/embed 等排除矩阵被一起压到 0.5B，实测低估；
        另发现 VLM 家族的 quantization_config 嵌在 raw.text_config 下，
        quantizationConfigOf 整族漏检）；修正后 K2-Thinking 2.053e12 →
        5.940e11（bf16 封顶 → w4a16）、K2.5/K2.6/K2.7-Code 5.134e11 →
        5.948e11、K3 1.390e12 → 1.555e12；M2.7 4.767e11 → 2.521e11、
        V4-Pro 3.237e12 → 1.689e12、Qwen3.8-2.4T 4.851e12 → 2.480e12。

      **量化 MoE 家族声明核实表（2026-09-09，全部单 ep 组 matrices=3）**：

      | 家族 | 模型数 | EH（in） | EI（out） | E | 量化方案 |
      |---|---|---|---|---|---|
      | minimax_m2 | 1 | 3072 | 1536 | 256 | fp8 |
      | minimax_m3_vl | 1 | 6144 | 3072 | 128 | mxfp8 |
      | qwen3_5_moe | 7 | 2048 | 512 | 256 | fp8 / gptq |
      | qwen3_5_moe_text | 1 | 8192 | 2048 | 512 | fp8 |
      | qwen4_exp | 1 | 2560 | 640 | 512 | fp8 |
      | deepseek_v3 | 2 | 7168 | 2048 | 256 | fp8 |
      | deepseek_v32 | 1 | 7168 | 2048 | 256 | fp8 |
      | deepseek_v4 | 5 | 7168 | 3072 | 384 | fp8（hash 层同声明） |
      | kimi_k2 | 4 | 7168 | 2048 | 384 | fp8 / compressed-tensors |
      | kimi_k25 | 3 | 7168 | 2048 | 384 | compressed-tensors w4a16 |
      | kimi_k3 | 1 | 3584（潜空间 latent） | 3072 | 896 | compressed-tensors mxfp4 |
      | glm_moe_dsa | 2 | 6144 | 2048 | 256 | fp8 |
      | glm5_next | 1 | 4096 | 2048 | 288 | fp8 |

      登记事项：① shared expert 全部为独立 mlp 叶（tp 组声明）——设计里的
      「shared 融合形态同叶声明两组」是 schema 能力，当前无树触发；
      ② K2.5/K2.6/K2.7/K3 的 quantization_config 嵌在 raw.text_config 下
      （VLM 家族），quantizationConfigOf 曾整族漏检 → 落到 normalize 派生的
      0.5B 标量宽（整模型统一 0.5B，排除矩阵应 bf16 而被压低）——W-C 修复
      后走逐矩阵精确枚举；K2-Thinking 无标量派生路径，此前 bf16 封顶，
      是 w4a16 的直接受益者。
      机制：MoE 模板的专家 GEMM（gate/up/down 三矩阵 [moeI, EH]）融合在
      swiglu 叶的 counts 里（3·E·EH·EI），`QUANTIZABLE_OPS` 枚举只认 linear
      族叶 → 该块留在 bf16 桶。
      落法（2026-09-09 定稿，待执行；**完整方案见
      [details/sharding_matrix.md](sharding_matrix.md)**——三层设计 +
      三消费者接线 + 四验收锚 + 三波执行，下文为摘要）：
      ① **拆 operator id 而不是拆叶子**：MoE 专家融合叶现在与纯激活共用
        `swiglu` 这个 id——这正是 QSA/DSA/MSA 同类问题的翻版（一个 id 两种
        算法出处）。给融合专家 MLP 独立 id（对标 vLLM `FusedMoE` 模块），
        counts 原样迁移 + 增加机器可读的权重矩阵声明
        （attributes: {matrices: 3, out: EI, in: EH, expertFold: E}）；
      ② `quantizedMatrixBytes` 消费该声明枚举 3×E 矩阵，**不再自行推导
        E-folding**（派生侧第二套折叠已实测会错——模拟器手算出负容量的
        直接教训）；dense MLP 的 swiglu（纯激活无权重）保持现 id；
      ③ **验收锚** = 枚举元素数 ×2B 必须等于叶 counts 的 bytes.weights
        （该值已被权重字节恒等式锚定，单源）。
      成熟方案对照：vLLM/SGLang 量化按**模块**应用（FusedMoE 模块持有
      w13/w2 打包权重，ignore list 按模块名）——模块即单位，叶子自描述；
      HF/llm-analysis 类显存工具走 per-tensor checkpoint 元数据（msv 的
      checkpoint 路径 parameters_by_dtype 已是该方案且正确），缺口仅在
      无 checkpoint 的派生路径。

      **并行策略维度（2026-09-09 用户补刀后联网调研定稿）**：正确的模型不是
      「逐节点路径规则表」，而是**分片响应矩阵**——内存类 × 并行轴 → 除数/
      策略，叶子声明供权重分组，plan 供轴度数，投影为纯函数。

      维度 1（张量侧·叶子声明的分片亲和）：
      `weightMatrices: [{class, out, in, count}]`，class ∈
      {tp-affine（attention/dense/shared 的 GEMM）、ep-affine（路由专家）、
      vocab（lm_head/embed，视 vocabParallel）、replicated（norm）}。

      维度 2（计划侧·轴与策略旗标）：tp、ep（或 moe_tp×moe_ep 混合，TRT-LLM
      语义）、dp、pp、attnMode（dp attention）、vocabParallel。
      **联网核实的组合语义**：
      - vLLM：EP_SIZE = TP_SIZE × DP_SIZE（DP attention + EP 是 DeepSeek 系
        标准部署，路由专家每卡 = E/(tp×dp)）；**无 EP 时 DP 也把专家 ÷dp
        切**（专家 TP 切分语义）——「DP=复制」只对 attention 成立；
      - shared expert：通常 ÷tp；仅当 EP+TP+DP 全开且特定 all2all backend
        时复制（AMD vLLM playbook 实证）；
      - TRT-LLM：TP / EP / 混合 ETP（每卡持 E/moe_ep 个完整专家、专家权重
        再 ÷moe_tp）三模式——计划轴需要 moe_tp/moe_ep 分离才能表达混合；
      - MLA/MQA 的 KV：TP 无法切单 KV 头 → KV ×tp 复制（msv 已实现：
        kvBytesPerCard 的 isMla 分支）；DP attention 下复制（已实现）；
        KDA state 同（已实现）。

      维度 3（内存类侧·每类对轴的响应不同）：weights 按上表；KV/state 的
      MLA 与 attnMode 响应已实现；activations（TP/SP/CP）超出当前范围，
      登记不展开。

      **三个消费者读同一矩阵**：① 量化枚举（quantizedMatrixBytes）
      ② EP/TP/DP 逐卡投影（weightBytesPerCard + expertWeightRange 不均衡
      区间）③ 容量 base 分桶。现有路径规则表降级为无声明叶子的回退。
      已核实的家族差异：M2.7 无 shared expert（声明只含 ep 组）；DeepSeek/
      K2 系 shared 是独立 linear 叶（swiglu 声明只含 routed 组）；
      **DP 轴语义分家**：vLLM 推理 DP=复制（每 rank 全权重，msv 现状正确），
      ZeRO/FSDP 式 ÷dp 分片属训练/离线推理扩展（SiDP），登记为范围外。
      执行时逐模型核实声明，不得假设。
- [x] **量化容量 per-matrix 精确化**（`cost/quantBytes.js`，2026-09-09）：
      无 checkpoint 时不再用标量 `quantizationBytesPerParameter` 一刀切 ——
      枚举树上全部线性族叶子的 [out,in]（output/input 正维积），按方案精确计：
      fp8 块量化（权重 1B + scale，`scale_fmt: "ue8m0"` 为 1B/块否则 fp32 4B/块）、
      mxfp8（权重 1B + e8m0 1B/块）、gptq int4（0.5B 权重 + 每组 fp16 scale +
      int4 qzeros）。哪些矩阵被量化由 config 的 `dynamic`/`modules_to_not_convert`
      声明（`"-:"` 排除正则），checkpoint 命名与树 id 的差异（visual/vision_tower、
      model.language_model 前缀）用候选路径桥接。实测 V3.1（ue8m0）容量 +0.01%、
      Qwen3.5-27B-FP8 +0.02%、27B-GPTQ-Int4 +2.65%（scale 与 qzeros 此前整片缺失）。
      纯 bare config（无树）回退标量 —— 没有 [out,in] 就没有 scale 形状，
      这是信息极限而非建模缺口。

<!-- BEGIN GENERATED: operators -->

> **本节由 `node scripts/gen-operators-reference.mjs` 生成，请勿手改。**
> 触发面按 `models/catalog.json` 全量模型实跑（prefill T=128 / decode T=1,S=4096 两相位）；
> `matrix|vector|sfu|bytes` 列的 ✓/0 表示该分量在任一相位是否非零（0 = 精确零，principles §3.3）。
> `来源` = principles §3.5 的三级体系（一 aten 锚点 / 二 modeling 对照 / 三 分解声明）。

## 总览表（生成物：48 个算子 / 59 个模型）

| 算子 | matrix | vector | sfu | bytes | 来源 | 触发模型 | 节点 | 实例 | 出现槽位 |
|---|---|---|---|---|---|---|---|---|---|
| `linear` | ✓ | ✓ | 0 | ✓ | 一 | 59/59 | 10237 | 28643 | attn_residual · indexer · lm_head · merger 等 12 |
| `residual_add` | 0 | ✓ | 0 | ✓ | 一 | 59/59 | 2670 | 6604 | decoder · layer · text_decoder |
| `rope` | 0 | ✓ | 0 | ✓ | 三 | 59/59 | 1157 | 2393 | self_attn |
| `embedding` | 0 | 0 | 0 | ✓ | — | 57/59 | 57 | 57 | root |
| `rmsnorm` | 0 | ✓ | ✓ | ✓ | 三 | 57/59 | 2027 | 8637 | attn_residual · enorm · hnorm · indexer 等 14 |
| `swiglu` | 0 | ✓ | ✓ | ✓ | 一 | 56/59 | 1283 | 3194 | merger · mlp · shared_experts · vision_tower |
| `matmul` | ✓ | 0 | 0 | ✓ | 一 | 48/59 | 944 | 4162 | self_attn · vision_tower |
| `softmax` | 0 | ✓ | ✓ | ✓ | 一 | 48/59 | 472 | 2081 | self_attn · vision_tower |
| `fused_moe_mlp` | ✓ | ✓ | ✓ | ✓ | 一 | 44/59 | 962 | 2580 | moe |
| `moe_combine` | 0 | ✓ | 0 | ✓ | 三 | 44/59 | 962 | 2580 | moe |
| `moe_dispatch` | 0 | 0 | 0 | ✓ | 一 | 44/59 | 962 | 2580 | moe |
| `topk` | 0 | ✓ | ✓ | ✓ | 一 | 44/59 | 947 | 2565 | moe |
| `moe_add` | 0 | ✓ | 0 | ✓ | 一 | 41/59 | 906 | 2422 | moe |
| `attention_qkv_split` | 0 | 0 | 0 | 0 | 二 | 40/59 | 43 | 1147 | self_attn · vision_tower |
| `vision_position` | 0 | ✓ | 0 | ✓ | 三 | 38/59 | 38 | 38 | vision_tower |
| `split` | 0 | 0 | 0 | 0 | 一 | 36/59 | 641 | 726 | self_attn |
| `vision_activation` | 0 | ✓ | ✓ | ✓ | 一 | 36/59 | 69 | 978 | merger · projector · vision_tower |
| `causal_conv1d` | ✓ | 0 | 0 | ✓ | 一 | 34/59 | 433 | 1274 | self_attn |
| `gated_delta_attention` | ✓ | ✓ | ✓ | ✓ | 二 | 34/59 | 433 | 1274 | self_attn |
| `gated_rmsnorm` | 0 | ✓ | ✓ | ✓ | 二 | 34/59 | 433 | 1274 | self_attn |
| `gemma_rmsnorm` | 0 | ✓ | ✓ | ✓ | 三 | 31/59 | 2402 | 4289 | enorm · hnorm · input_layernorm · norm 等 7 |
| `qwen_qkvz_split` | 0 | 0 | 0 | 0 | 二 | 31/59 | 383 | 1137 | self_attn |
| `vision_merge` | 0 | 0 | 0 | ✓ | 三 | 31/59 | 31 | 31 | merger |
| `attention_output_gate` | 0 | ✓ | ✓ | ✓ | 二 | 29/59 | 384 | 355 | self_attn |
| `mla_kv_compress` | ✓ | 0 | 0 | ✓ | 三 | 24/59 | 475 | 1369 | self_attn |
| `mla_kv_split` | 0 | 0 | 0 | 0 | 一 | 19/59 | 230 | 1124 | self_attn |
| `mla_query_compress` | ✓ | 0 | 0 | ✓ | 三 | 19/59 | 230 | 1124 | self_attn |
| `shared_expert_gate` | 0 | ✓ | ✓ | ✓ | 二 | 16/59 | 442 | 844 | shared_expert_gate |
| `dsa_sparse_mla` | ✓ | 0 | 0 | ✓ | 二 | 9/59 | 187 | 551 | self_attn |
| `dsa_indexer` | ✓ | ✓ | 0 | ✓ | 二 | 7/59 | 165 | 529 | self_attn |
| `mhc_contract` | 0 | ✓ | 0 | ✓ | 三 | 7/59 | 7 | 7 | mhc_contract |
| `mhc_fused_post_pre` | ✓ | ✓ | ✓ | ✓ | 三 | 7/59 | 299 | 341 | mhc_ffn_pre |
| `mhc_post` | ✓ | ✓ | 0 | ✓ | 三 | 7/59 | 7 | 7 | mhc_final_post |
| `mhc_pre` | ✓ | ✓ | ✓ | ✓ | 三 | 7/59 | 299 | 341 | mhc_attn_pre |
| `dsv4_compressed_attention` | ✓ | 0 | 0 | ✓ | 二 | 5/59 | 122 | 122 | self_attn |
| `dsv4_hash_route` | 0 | 0 | 0 | ✓ | 二 | 5/59 | 15 | 15 | moe |
| `dsv4_indexer` | ✓ | ✓ | 0 | ✓ | 二 | 5/59 | 123 | 123 | self_attn |
| `dsv4_sparse_mla` | ✓ | 0 | 0 | ✓ | 二 | 5/59 | 123 | 123 | self_attn |
| `dsv4_swa_attention` | ✓ | 0 | 0 | ✓ | 二 | 3/59 | 6 | 6 | self_attn |
| `dsa_kpool_indexer` | ✓ | ✓ | 0 | ✓ | 二 | 2/59 | 22 | 22 | self_attn |
| `hyper_connection` | ✓ | ✓ | ✓ | ✓ | 二 | 2/59 | 110 | 194 | attn_hyper_connection · hyper_connection_mixer · mlp_hyper_connection |
| `minimax_sparse_attention` | ✓ | 0 | 0 | ✓ | 二 | 2/59 | 4 | 114 | self_attn |
| `minimax_sparse_indexer` | ✓ | ✓ | 0 | ✓ | 二 | 2/59 | 4 | 114 | self_attn |
| `ple` | ✓ | ✓ | ✓ | ✓ | 三 | 2/59 | 2 | 2 | ple |
| `qsa_indexer` | ✓ | ✓ | 0 | ✓ | 二 | 2/59 | 26 | 24 | self_attn |
| `qsa_sparse_attention` | ✓ | 0 | 0 | ✓ | 二 | 2/59 | 26 | 24 | self_attn |
| `attention_residual` | 0 | ✓ | ✓ | ✓ | 二 | 1/59 | 47 | 93 | attn_residual |
| `mla_output_gate` | 0 | ✓ | ✓ | ✓ | 二 | 1/59 | 23 | 24 | self_attn |

未识别叶子（无 operator_id 且非 embedding）：**0**

## 算力/访存占比（prefill，T=128，59 模型实例加权求和）

| 算子 | matrix (MACs) | matrix 占比 | bytes | bytes 占比 |
|---|---|---|---|---|
| `fused_moe_mlp` | 8.513e+13 | 37.99% | 5.751e+13 | 95.90% |
| `linear` | 1.327e+14 | 59.21% | 1.976e+12 | 3.29% |
| `matmul` | 2.815e+12 | 1.26% | 8.339e+10 | 0.14% |
| `mla_query_compress` | 1.665e+12 | 0.74% | 2.844e+10 | 0.05% |
| `mla_kv_compress` | 8.238e+11 | 0.37% | 1.541e+10 | 0.03% |
| `gated_delta_attention` | 4.641e+11 | 0.21% | 1.008e+10 | 0.02% |
| `hyper_connection` | 1.637e+11 | 0.07% | 7.707e+9 | 0.01% |
| `dsa_sparse_mla` | 1.268e+11 | 0.06% | 6.618e+9 | 0.01% |
| `dsv4_sparse_mla` | 9.901e+10 | 0.04% | 3.912e+9 | 0.01% |
| `softmax` | 0.000e+0 | 0.00% | 6.874e+10 | 0.11% |
| `rope` | 0.000e+0 | 0.00% | 4.421e+10 | 0.07% |
| `moe_combine` | 0.000e+0 | 0.00% | 3.248e+10 | 0.05% |
| `moe_dispatch` | 0.000e+0 | 0.00% | 3.247e+10 | 0.05% |
| `mhc_fused_post_pre` | 2.177e+10 | 0.01% | 6.903e+9 | 0.01% |
| `mhc_pre` | 2.177e+10 | 0.01% | 5.996e+9 | 0.01% |

合计：matrix 2.2407e+14 MACs · bytes 5.9969e+13（前 15 名之外的算子占比均 < 前列末位）

## 算力/访存占比（decode，T=1 S=4096，59 模型实例加权求和）

| 算子 | matrix (MACs) | matrix 占比 | bytes | bytes 占比 |
|---|---|---|---|---|
| `linear` | 1.666e+13 | 73.21% | 1.895e+12 | 52.61% |
| `matmul` | 5.347e+12 | 23.50% | 1.526e+11 | 4.24% |
| `fused_moe_mlp` | 6.651e+11 | 2.92% | 1.330e+12 | 36.93% |
| `softmax` | 0.000e+0 | 0.00% | 1.341e+11 | 3.72% |
| `mla_query_compress` | 1.301e+10 | 0.06% | 2.603e+10 | 0.72% |
| `dsa_sparse_mla` | 3.146e+10 | 0.14% | 1.969e+9 | 0.05% |
| `vision_activation` | 0.000e+0 | 0.00% | 2.463e+10 | 0.68% |
| `mla_kv_compress` | 6.436e+9 | 0.03% | 1.289e+10 | 0.36% |
| `dsa_indexer` | 9.899e+9 | 0.04% | 1.973e+9 | 0.05% |
| `dsv4_sparse_mla` | 1.017e+10 | 0.04% | 3.277e+8 | 0.01% |
| `rmsnorm` | 0.000e+0 | 0.00% | 8.900e+9 | 0.25% |
| `gated_delta_attention` | 3.625e+9 | 0.02% | 5.043e+9 | 0.14% |
| `dsv4_indexer` | 4.127e+9 | 0.02% | 7.158e+8 | 0.02% |
| `minimax_sparse_attention` | 4.064e+9 | 0.02% | 6.390e+8 | 0.02% |
| `hyper_connection` | 1.279e+9 | 0.01% | 2.603e+9 | 0.07% |

合计：matrix 2.2757e+13 MACs · bytes 3.6019e+12（前 15 名之外的算子占比均 < 前列末位）

## 注册表 ↔ 触发面对账（生成物）

- 注册表条目：**49**
- 实际被触发：**48**（含结构节点 `embedding`）
- 零触发条目：**2** —— `linear_attention` · `linear_attention_gate`

## 双向表 A（生成物）· 算子 → 结构槽位

> 槽位 = 叶子 id 去掉层号与叶名的路径（`decoder.0.self_attn.q_proj` → `decoder.self_attn`）。括号内为节点数。

| 算子 | 槽位（节点数） |
|---|---|
| `linear` | `decoder.self_attn`(4134) · `decoder.moe.shared_experts`(2613) · `decoder.mlp`(1068) · `decoder.moe`(1006) · `decoder.self_attn.indexer`(606) · `vision_tower`(192) · `mtp.layer.self_attn`(125) · `mtp.layer.moe.shared_experts`(99) · `decoder.attn_residual`(94) · `vision_tower.merger`(66) · `lm_head`(59) · `mtp`(51) · `mtp.layer.mlp`(45) · `mtp.layer.moe`(31) · `mtp.layer.self_attn.indexer`(14) · `projector`(11) · `text_decoder.self_attn`(8) · `text_decoder.mlp`(6) · `text_decoder.moe.shared_experts`(6) · `text_decoder.moe`(2) · `output_attn_residual`(1) |
| `residual_add` | `decoder`(2560) · `mtp.layer`(102) · `text_decoder`(8) |
| `rope` | `decoder.self_attn`(1095) · `mtp.layer.self_attn`(56) · `text_decoder.self_attn`(6) |
| `embedding` | `root`(57) |
| `rmsnorm` | `decoder.self_attn`(988) · `decoder.input_layernorm`(226) · `decoder.post_attention_layernorm`(226) · `decoder.self_attn.indexer`(180) · `decoder.attn_residual`(94) · `vision_tower`(76) · `decoder.moe`(46) · `mtp.layer.self_attn`(36) · `vision_tower.merger`(33) · `norm`(28) · `mtp.enorm`(20) · `mtp.hnorm`(20) · `mtp.shared_head_norm`(20) · `mtp.layer.input_layernorm`(11) · `mtp.layer.post_attention_layernorm`(11) · `mtp.layer.self_attn.indexer`(7) · `projector`(4) · `output_attn_residual`(1) |
| `swiglu` | `decoder.moe.shared_experts`(871) · `decoder.mlp`(356) · `mtp.layer.moe.shared_experts`(33) · `mtp.layer.mlp`(15) · `text_decoder.mlp`(2) · `text_decoder.moe.shared_experts`(2) · `vision_tower`(2) · `vision_tower.merger`(2) |
| `matmul` | `decoder.self_attn`(798) · `vision_tower`(76) · `mtp.layer.self_attn`(66) · `text_decoder.self_attn`(4) |
| `softmax` | `decoder.self_attn`(399) · `vision_tower`(38) · `mtp.layer.self_attn`(33) · `text_decoder.self_attn`(2) |
| `fused_moe_mlp` | `decoder.moe`(924) · `mtp.layer.moe`(36) · `text_decoder.moe`(2) |
| `moe_combine` | `decoder.moe`(924) · `mtp.layer.moe`(36) · `text_decoder.moe`(2) |
| `moe_dispatch` | `decoder.moe`(924) · `mtp.layer.moe`(36) · `text_decoder.moe`(2) |
| `topk` | `decoder.moe`(914) · `mtp.layer.moe`(31) · `text_decoder.moe`(2) |
| `moe_add` | `decoder.moe`(871) · `mtp.layer.moe`(33) · `text_decoder.moe`(2) |
| `attention_qkv_split` | `vision_tower`(38) · `decoder.self_attn`(3) · `mtp.layer.self_attn`(2) |
| `vision_position` | `vision_tower`(38) |
| `split` | `decoder.self_attn`(601) · `mtp.layer.self_attn`(36) · `text_decoder.self_attn`(4) |
| `vision_activation` | `vision_tower`(36) · `vision_tower.merger`(29) · `projector`(4) |
| `causal_conv1d` | `decoder.self_attn`(431) · `mtp.layer.self_attn`(2) |
| `gated_delta_attention` | `decoder.self_attn`(431) · `mtp.layer.self_attn`(2) |
| `gated_rmsnorm` | `decoder.self_attn`(431) · `mtp.layer.self_attn`(2) |
| `gemma_rmsnorm` | `decoder.input_layernorm`(710) · `decoder.post_attention_layernorm`(710) · `decoder.self_attn`(710) · `mtp.layer.self_attn`(66) · `mtp.enorm`(31) · `mtp.hnorm`(31) · `mtp.layer.input_layernorm`(31) · `mtp.layer.post_attention_layernorm`(31) · `mtp.shared_head_norm`(31) · `norm`(31) · `text_decoder.self_attn`(12) · `text_decoder.input_layernorm`(4) · `text_decoder.post_attention_layernorm`(4) |
| `qwen_qkvz_split` | `decoder.self_attn`(383) |
| `vision_merge` | `vision_tower.merger`(31) |
| `attention_output_gate` | `decoder.self_attn`(355) · `mtp.layer.self_attn`(29) |
| `mla_kv_compress` | `decoder.self_attn`(464) · `mtp.layer.self_attn`(11) |
| `mla_kv_split` | `decoder.self_attn`(221) · `mtp.layer.self_attn`(9) |
| `mla_query_compress` | `decoder.self_attn`(221) · `mtp.layer.self_attn`(9) |
| `shared_expert_gate` | `decoder.moe.shared_expert_gate`(426) · `mtp.layer.moe.shared_expert_gate`(16) |
| `dsa_sparse_mla` | `decoder.self_attn`(180) · `mtp.layer.self_attn`(7) |
| `dsa_indexer` | `decoder.self_attn`(158) · `mtp.layer.self_attn`(7) |
| `mhc_contract` | `decoder.mhc_contract`(7) |
| `mhc_fused_post_pre` | `decoder.mhc_ffn_pre`(292) · `mtp.layer.mhc_ffn_pre`(7) |
| `mhc_post` | `decoder.mhc_final_post`(7) |
| `mhc_pre` | `decoder.mhc_attn_pre`(292) · `mtp.layer.mhc_attn_pre`(7) |
| `dsv4_compressed_attention` | `decoder.self_attn`(120) · `mtp.layer.self_attn`(2) |
| `dsv4_hash_route` | `decoder.moe`(10) · `mtp.layer.moe`(5) |
| `dsv4_indexer` | `decoder.self_attn`(123) |
| `dsv4_sparse_mla` | `decoder.self_attn`(123) |
| `dsv4_swa_attention` | `decoder.self_attn`(3) · `mtp.layer.self_attn`(3) |
| `dsa_kpool_indexer` | `decoder.self_attn`(22) |
| `hyper_connection` | `decoder.attn_hyper_connection`(52) · `decoder.mlp_hyper_connection`(52) · `hyper_connection_mixer`(2) · `mtp.layer.attn_hyper_connection`(2) · `mtp.layer.mlp_hyper_connection`(2) |
| `minimax_sparse_attention` | `mtp.layer.self_attn`(2) · `text_decoder.self_attn`(2) |
| `minimax_sparse_indexer` | `mtp.layer.self_attn`(2) · `text_decoder.self_attn`(2) |
| `ple` | `decoder.ple`(2) |
| `qsa_indexer` | `decoder.self_attn`(24) · `mtp.layer.self_attn`(2) |
| `qsa_sparse_attention` | `decoder.self_attn`(24) · `mtp.layer.self_attn`(2) |
| `attention_residual` | `decoder.attn_residual`(47) |
| `mla_output_gate` | `decoder.self_attn`(23) |

## 双向表 B（生成物）· 结构槽位 → 算子序列（数据流顺序）

> 序列按子节点声明顺序取，跨 59 模型做「首次出现即追加」的并集 —— 同一槽位不同结构类的算子会依次排在后面。

| 结构槽位 | 算子序列 |
|---|---|
| `decoder.self_attn` | `linear` → `attention_qkv_split` → `rmsnorm` → `rope` → `matmul` → `softmax` → `qwen_qkvz_split` → `causal_conv1d` → `gated_delta_attention` → `gated_rmsnorm` → `split` → `gemma_rmsnorm` → `attention_output_gate` → `qsa_indexer` → `qsa_sparse_attention` → `mla_query_compress` → `mla_kv_compress` → `mla_kv_split` → `dsa_indexer` → `dsa_sparse_mla` → `dsv4_swa_attention` → `dsv4_indexer` → `dsv4_sparse_mla` → `dsv4_compressed_attention` → `mla_output_gate` → `dsa_kpool_indexer` |
| `mtp.layer.self_attn` | `linear` → `attention_qkv_split` → `rmsnorm` → `rope` → `matmul` → `softmax` → `split` → `gemma_rmsnorm` → `minimax_sparse_indexer` → `minimax_sparse_attention` → `attention_output_gate` → `qsa_indexer` → `qsa_sparse_attention` → `mla_query_compress` → `mla_kv_compress` → `mla_kv_split` → `dsa_indexer` → `dsa_sparse_mla` → `dsv4_swa_attention` → `dsv4_compressed_attention` → `causal_conv1d` → `gated_delta_attention` → `gated_rmsnorm` |
| `decoder.moe` | `linear` → `topk` → `moe_dispatch` → `fused_moe_mlp` → `moe_combine` → `moe_add` → `dsv4_hash_route` → `rmsnorm` |
| `text_decoder.self_attn` | `linear` → `split` → `gemma_rmsnorm` → `rope` → `matmul` → `softmax` → `minimax_sparse_indexer` → `minimax_sparse_attention` |
| `vision_tower` | `linear` → `vision_position` → `rmsnorm` → `attention_qkv_split` → `matmul` → `softmax` → `vision_activation` → `swiglu` |
| `mtp.layer.moe` | `linear` → `topk` → `moe_dispatch` → `fused_moe_mlp` → `moe_combine` → `moe_add` → `dsv4_hash_route` |
| `text_decoder.moe` | `linear` → `topk` → `moe_dispatch` → `fused_moe_mlp` → `moe_combine` → `moe_add` |
| `vision_tower.merger` | `vision_merge` → `rmsnorm` → `linear` → `vision_activation` → `swiglu` |
| `decoder.attn_residual` | `attention_residual` → `rmsnorm` → `linear` |
| `projector` | `linear` → `rmsnorm` → `vision_activation` |
| `decoder.input_layernorm` | `rmsnorm` → `gemma_rmsnorm` |
| `decoder.mlp` | `linear` → `swiglu` |
| `decoder.moe.shared_experts` | `linear` → `swiglu` |
| `decoder.post_attention_layernorm` | `rmsnorm` → `gemma_rmsnorm` |
| `decoder.self_attn.indexer` | `linear` → `rmsnorm` |
| `mtp.enorm` | `rmsnorm` → `gemma_rmsnorm` |
| `mtp.hnorm` | `rmsnorm` → `gemma_rmsnorm` |
| `mtp.layer.input_layernorm` | `rmsnorm` → `gemma_rmsnorm` |
| `mtp.layer.mlp` | `linear` → `swiglu` |
| `mtp.layer.moe.shared_experts` | `linear` → `swiglu` |
| `mtp.layer.post_attention_layernorm` | `rmsnorm` → `gemma_rmsnorm` |
| `mtp.layer.self_attn.indexer` | `linear` → `rmsnorm` |
| `mtp.shared_head_norm` | `rmsnorm` → `gemma_rmsnorm` |
| `norm` | `rmsnorm` → `gemma_rmsnorm` |
| `output_attn_residual` | `rmsnorm` → `linear` |
| `text_decoder.mlp` | `linear` → `swiglu` |
| `text_decoder.moe.shared_experts` | `linear` → `swiglu` |
| `decoder` | `residual_add` |
| `decoder.attn_hyper_connection` | `hyper_connection` |
| `decoder.mhc_attn_pre` | `mhc_pre` |
| `decoder.mhc_contract` | `mhc_contract` |
| `decoder.mhc_ffn_pre` | `mhc_fused_post_pre` |
| `decoder.mhc_final_post` | `mhc_post` |
| `decoder.mlp_hyper_connection` | `hyper_connection` |
| `decoder.moe.shared_expert_gate` | `shared_expert_gate` |
| `decoder.ple` | `ple` |
| `hyper_connection_mixer` | `hyper_connection` |
| `lm_head` | `linear` |
| `mtp` | `linear` |
| `mtp.layer` | `residual_add` |
| `mtp.layer.attn_hyper_connection` | `hyper_connection` |
| `mtp.layer.mhc_attn_pre` | `mhc_pre` |
| `mtp.layer.mhc_ffn_pre` | `mhc_fused_post_pre` |
| `mtp.layer.mlp_hyper_connection` | `hyper_connection` |
| `mtp.layer.moe.shared_expert_gate` | `shared_expert_gate` |
| `root` | `embedding` |
| `text_decoder` | `residual_add` |
| `text_decoder.input_layernorm` | `gemma_rmsnorm` |
| `text_decoder.post_attention_layernorm` | `gemma_rmsnorm` |

## 模块层分解台账（生成物）

- 已声明分解的模块：**26** —— `linear` · `rmsnorm` · `rope` · `swiglu` · `gate` · `softmax` · `topk_router` · `mla_query_compress` · `mla_kv_compress` · `dsv4_hash_route` · `vision_position` · `vision_merge` · `vision_activation` · `attention_residual` · `hyper_connection` · `ple` · `mhc_contract` · `mhc_pre` · `mhc_post` · `mhc_fused_post_pre` · `sdpa_attention` · `dsa_indexer` · `dsa_kpool_indexer` · `qsa_indexer` · `minimax_block_indexer` · `linear_attention_state`
- 尚未声明分解（`DECOMPOSE_PENDING`）：**0**

| 模块 | 待办原因 |
|---|---|

> 恒等式（融合分解 / 权重字节 / KV 读量 / 激活流形状连续性）的判定结果不在此生成，
> 由 `npm test` 的 `identities.test.js` 与 `modelIdentities.test.js` 断言并打印报表 —— 避免同一套数学写两遍。

## 逐结构类算子明细（生成物）

> 16 个结构类各取一个代表模型（plan §五），**每算子两行**：prefill T=S=2048 / decode T=1 S=4096。
> 数值 = 该模型内该算子**按实例加权求和**（层组 repeat 已乘）。视觉塔叶按 `visionTokens` 计。
> `AI` = matrix(MAC) / bytesMoved；`bound` = 五路 max（参考芯片 A100-80G，efficiency=1，仅判结构倾向）。
> `恒等式` 只对「算子即模块」的那些算（映射见 `formulas/moduleProbeParams.js` 的 `OPERATOR_TO_MODULE`）：
> `计算✓` = fused 三分量与原子分解逐位相等；`字节✓` = fused 落在 [compulsory 下界, Σ分解] 之间。
> `融合收益` = Σ 2·residentIntermediates·b（分解下会落 HBM、融合下留在寄存器/SRAM 的量）。
> `容差` 一列单列：结构性恒等式一律 0（整数相等 / 不等式夹逼），非 0 只可能是显式声明的近似执行形态。
> **bound 的粒度是叶级**。「attention prefill 算力瓶颈」说的是**模块级** —— MLA 的 `matmul` 叶
> 单看是访存侧（latent 读占主导，AI 未过 ridge），把 q/kv 压缩 + 投影 + scores/context 合起来才是算力侧。
> 模块级断言在 `modelIdentities.test.js`（error 模式），不在本表。

### S01 · Qwen/Qwen3.5-0.8B

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | prefill | 86/208 | 1.597e+12 | 0 | 0 | 1.701e+9 | 1.094e+9 | 2.756e+9 | 287.62 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 86/208 | 5.760e+10 | 0 | 0 | 1.701e+9 | 8.366e+7 | 1.025e+8 | 30.52 | memory | 计算✓ 字节✓ | 0 | 0 |
| `matmul` | prefill | 16/36 | 5.463e+10 | 0 | 0 | 0 | 3.566e+8 | 3.102e+8 | 81.92 | matrix | — | — | — |
| `matmul` | decode | 16/36 | 6.216e+9 | 0 | 0 | 0 | 1.782e+8 | 1.066e+8 | 21.83 | memory | — | — | — |
| `gated_delta_attention` | prefill | 6/18 | 2.899e+10 | 1.510e+8 | 2.765e+4 | 2.304e+3 | 3.232e+8 | 3.232e+8 | 44.85 | memory | 计算✓ 字节✓ | 0 | 0 |
| `gated_delta_attention` | decode | 6/18 | 1.416e+7 | 4.719e+6 | 8.640e+2 | 2.304e+3 | 1.010e+7 | 1.010e+7 | 0.70 | memory | 计算✓ 字节✓ | 0 | 0 |
| `causal_conv1d` | prefill | 6/18 | 9.060e+8 | 0 | 0 | 8.847e+5 | 4.530e+8 | 4.530e+8 | 1.00 | memory | — | — | — |
| `causal_conv1d` | decode | 6/18 | 4.424e+5 | 0 | 0 | 8.847e+5 | 8.847e+5 | 8.847e+5 | 0.17 | memory | — | — | — |
| `swiglu` | prefill | 13/24 | 0 | 3.523e+8 | 3.523e+8 | 0 | 7.046e+8 | 3.523e+8 | 0.00 | memory | 计算✓ 字节✓ | 2.936e+7 | 0 |
| `swiglu` | decode | 13/24 | 0 | 1.720e+5 | 1.720e+5 | 0 | 3.441e+5 | 1.720e+5 | 0.00 | memory | 计算✓ 字节✓ | 1.434e+4 | 0 |
| `residual_add` | prefill | 26/48 | 0 | 1.007e+8 | 0 | 0 | 4.027e+8 | 2.013e+8 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 26/48 | 0 | 4.915e+4 | 0 | 0 | 1.966e+5 | 9.830e+4 | 0.00 | memory | — | — | — |
| `gated_rmsnorm` | prefill | 6/18 | 0 | 3.775e+8 | 1.510e+8 | 4.608e+3 | 3.020e+8 | 1.510e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.679e+7 | 0 |
| `gated_rmsnorm` | decode | 6/18 | 0 | 1.843e+5 | 7.375e+4 | 4.608e+3 | 1.475e+5 | 7.373e+4 | 0.00 | memory | 计算✓ 字节✓ | 8.200e+3 | 0 |
| `gemma_rmsnorm` | prefill | 44/61 | 0 | 6.710e+8 | 1.249e+5 | 1.065e+5 | 2.684e+8 | 2.684e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.679e+7 | 0 |
| `gemma_rmsnorm` | decode | 44/61 | 0 | 3.276e+5 | 6.100e+1 | 1.065e+5 | 1.311e+5 | 1.311e+5 | 0.00 | memory | 计算✓ 字节✓ | 8.200e+3 | 0 |
| `softmax` | prefill | 8/18 | 0 | 3.739e+8 | 2.493e+8 | 0 | 2.493e+8 | 2.493e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 8/18 | 0 | 1.439e+8 | 9.594e+7 | 0 | 9.594e+7 | 9.594e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `vision_activation` | prefill | 2/13 | 0 | 4.601e+7 | 4.601e+7 | 0 | 9.201e+7 | 4.601e+7 | 0.00 | memory | — | — | — |
| `vision_activation` | decode | 2/13 | 0 | 4.601e+7 | 4.601e+7 | 0 | 9.201e+7 | 4.601e+7 | 0.00 | memory | — | — | — |
| `attention_output_gate` | prefill | 7/6 | 0 | 2.517e+7 | 5.033e+7 | 0 | 5.033e+7 | 5.033e+7 | 0.00 | memory | — | — | — |
| `attention_output_gate` | decode | 7/6 | 0 | 1.229e+4 | 2.458e+4 | 0 | 2.458e+4 | 2.458e+4 | 0.00 | memory | — | — | — |
| `rope` | prefill | 7/6 | 0 | 2.359e+7 | 0 | 0 | 3.146e+7 | 1.573e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 7/6 | 0 | 1.152e+4 | 0 | 0 | 1.536e+4 | 7.680e+3 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rmsnorm` | prefill | 3/25 | 0 | 4.953e+7 | 1.440e+4 | 4.301e+4 | 2.477e+7 | 2.477e+7 | 0.00 | memory | 计算✓ 字节✓ | 1.679e+7 | 0 |
| `rmsnorm` | decode | 3/25 | 0 | 4.953e+7 | 1.440e+4 | 4.301e+4 | 2.477e+7 | 2.477e+7 | 0.00 | memory | 计算✓ 字节✓ | 8.200e+3 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 4.194e+6 | 4.194e+6 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 2.048e+3 | 2.048e+3 | 0.00 | memory | — | — | — |
| `vision_position` | prefill | 1/1 | 0 | 4.424e+5 | 0 | 0 | 1.769e+6 | 8.847e+5 | 0.00 | memory | — | — | — |
| `vision_position` | decode | 1/1 | 0 | 4.424e+5 | 0 | 0 | 1.769e+6 | 8.847e+5 | 0.00 | memory | — | — | — |
| `vision_merge` | prefill | 1/1 | 0 | 0 | 0 | 0 | 8.847e+5 | 8.847e+5 | 0.00 | memory | — | — | — |
| `vision_merge` | decode | 1/1 | 0 | 0 | 0 | 0 | 8.847e+5 | 8.847e+5 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 1/12 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 1/12 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `qwen_qkvz_split` | prefill | 6/18 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `qwen_qkvz_split` | decode | 6/18 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | prefill | 7/6 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | decode | 7/6 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S02 · Qwen/Qwen3.5-35B-A3B

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | prefill | 155/412 | 4.226e+12 | 0 | 0 | 4.764e+9 | 2.863e+9 | 4.115e+9 | 359.87 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 155/412 | 2.574e+11 | 0 | 0 | 4.764e+9 | 2.550e+8 | 3.239e+8 | 48.17 | memory | 计算✓ 字节✓ | 0 | 0 |
| `fused_moe_mlp` | prefill | 21/40 | 2.062e+12 | 6.711e+8 | 6.711e+8 | 6.442e+10 | 1.342e+9 | 6.711e+8 | 31.03 | memory | — | — | — |
| `fused_moe_mlp` | decode | 21/40 | 1.007e+9 | 3.277e+5 | 3.277e+5 | 2.013e+9 | 6.554e+5 | 3.277e+5 | 0.50 | memory | — | — | — |
| `matmul` | prefill | 24/74 | 1.822e+11 | 0 | 0 | 0 | 1.132e+9 | 1.019e+9 | 84.72 | matrix | — | — | — |
| `matmul` | decode | 24/74 | 2.097e+10 | 0 | 0 | 0 | 4.794e+8 | 3.239e+8 | 26.11 | memory | — | — | — |
| `gated_delta_attention` | prefill | 10/30 | 9.664e+10 | 5.033e+8 | 9.216e+4 | 7.680e+3 | 1.054e+9 | 1.054e+9 | 45.85 | memory | 计算✓ 字节✓ | 0 | 0 |
| `gated_delta_attention` | decode | 10/30 | 4.719e+7 | 1.573e+7 | 2.880e+3 | 7.680e+3 | 3.293e+7 | 3.293e+7 | 0.72 | memory | 计算✓ 字节✓ | 0 | 0 |
| `causal_conv1d` | prefill | 10/30 | 2.013e+9 | 0 | 0 | 1.966e+6 | 1.007e+9 | 1.007e+9 | 1.00 | memory | — | — | — |
| `causal_conv1d` | decode | 10/30 | 9.830e+5 | 0 | 0 | 1.966e+6 | 1.966e+6 | 1.966e+6 | 0.17 | memory | — | — | — |
| `moe_combine` | prefill | 21/40 | 0 | 2.684e+9 | 0 | 0 | 2.686e+9 | 3.355e+8 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 21/40 | 0 | 1.311e+6 | 0 | 0 | 1.311e+6 | 1.638e+5 | 0.00 | memory | — | — | — |
| `residual_add` | prefill | 42/80 | 0 | 3.355e+8 | 0 | 0 | 1.342e+9 | 6.711e+8 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 42/80 | 0 | 1.638e+5 | 0 | 0 | 6.554e+5 | 3.277e+5 | 0.00 | memory | — | — | — |
| `gated_rmsnorm` | prefill | 10/30 | 0 | 1.258e+9 | 5.034e+8 | 7.680e+3 | 1.007e+9 | 5.033e+8 | 0.00 | memory | 计算✓ 字节✓ | 3.357e+7 | 0 |
| `gated_rmsnorm` | decode | 10/30 | 0 | 6.144e+5 | 2.458e+5 | 7.680e+3 | 4.915e+5 | 2.458e+5 | 0.00 | memory | 计算✓ 字节✓ | 1.639e+4 | 0 |
| `gemma_rmsnorm` | prefill | 68/101 | 0 | 2.170e+9 | 2.068e+5 | 3.420e+5 | 8.682e+8 | 8.682e+8 | 0.00 | memory | 计算✓ 字节✓ | 3.357e+7 | 0 |
| `gemma_rmsnorm` | decode | 68/101 | 0 | 1.060e+6 | 1.010e+2 | 3.420e+5 | 4.239e+5 | 4.239e+5 | 0.00 | memory | 计算✓ 字节✓ | 1.639e+4 | 0 |
| `softmax` | prefill | 12/37 | 0 | 1.222e+9 | 8.150e+8 | 0 | 8.150e+8 | 8.150e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 12/37 | 0 | 4.319e+8 | 2.880e+8 | 0 | 2.880e+8 | 2.880e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_add` | prefill | 21/40 | 0 | 1.678e+8 | 0 | 0 | 6.711e+8 | 3.355e+8 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 21/40 | 0 | 8.192e+4 | 0 | 0 | 3.277e+5 | 1.638e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 21/40 | 0 | 0 | 0 | 0 | 3.355e+8 | 2.684e+9 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 21/40 | 0 | 0 | 0 | 0 | 1.638e+5 | 1.311e+6 | 0.00 | memory | — | — | — |
| `shared_expert_gate` | prefill | 21/40 | 0 | 1.678e+8 | 3.355e+8 | 0 | 3.355e+8 | 3.355e+8 | 0.00 | memory | — | — | — |
| `shared_expert_gate` | decode | 21/40 | 0 | 8.192e+4 | 1.638e+5 | 0 | 1.638e+5 | 1.638e+5 | 0.00 | memory | — | — | — |
| `vision_activation` | prefill | 2/28 | 0 | 1.392e+8 | 1.392e+8 | 0 | 2.784e+8 | 1.392e+8 | 0.00 | memory | — | — | — |
| `vision_activation` | decode | 2/28 | 0 | 1.392e+8 | 1.392e+8 | 0 | 2.784e+8 | 1.392e+8 | 0.00 | memory | — | — | — |
| `attention_output_gate` | prefill | 11/10 | 0 | 8.389e+7 | 1.678e+8 | 0 | 1.678e+8 | 1.678e+8 | 0.00 | memory | — | — | — |
| `attention_output_gate` | decode | 11/10 | 0 | 4.096e+4 | 8.192e+4 | 0 | 8.192e+4 | 8.192e+4 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 21/40 | 0 | 8.389e+7 | 8.389e+7 | 0 | 1.678e+8 | 8.389e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `swiglu` | decode | 21/40 | 0 | 4.096e+4 | 4.096e+4 | 0 | 8.192e+4 | 4.096e+4 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | prefill | 11/10 | 0 | 7.078e+7 | 0 | 0 | 9.437e+7 | 4.719e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 11/10 | 0 | 3.456e+4 | 0 | 0 | 4.608e+4 | 2.304e+4 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rmsnorm` | prefill | 3/55 | 0 | 1.539e+8 | 3.168e+4 | 1.336e+5 | 7.697e+7 | 7.697e+7 | 0.00 | memory | 计算✓ 字节✓ | 3.357e+7 | 0 |
| `rmsnorm` | decode | 3/55 | 0 | 1.539e+8 | 3.168e+4 | 1.336e+5 | 7.697e+7 | 7.697e+7 | 0.00 | memory | 计算✓ 字节✓ | 1.639e+4 | 0 |
| `topk` | prefill | 21/40 | 0 | 2.154e+7 | 6.554e+5 | 0 | 4.194e+7 | 2.621e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 21/40 | 0 | 1.052e+4 | 3.200e+2 | 0 | 2.048e+4 | 1.280e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 8.389e+6 | 8.389e+6 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 4.096e+3 | 4.096e+3 | 0.00 | memory | — | — | — |
| `vision_position` | prefill | 1/1 | 0 | 6.636e+5 | 0 | 0 | 2.654e+6 | 1.327e+6 | 0.00 | memory | — | — | — |
| `vision_position` | decode | 1/1 | 0 | 6.636e+5 | 0 | 0 | 2.654e+6 | 1.327e+6 | 0.00 | memory | — | — | — |
| `vision_merge` | prefill | 1/1 | 0 | 0 | 0 | 0 | 1.327e+6 | 1.327e+6 | 0.00 | memory | — | — | — |
| `vision_merge` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.327e+6 | 1.327e+6 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 1/27 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 1/27 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `qwen_qkvz_split` | prefill | 10/30 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `qwen_qkvz_split` | decode | 10/30 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | prefill | 11/10 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | decode | 11/10 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S03 · zai-org/GLM-5

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_mlp` | prefill | 2/75 | 4.639e+13 | 5.033e+9 | 5.033e+9 | 1.450e+12 | 1.007e+10 | 5.033e+9 | 31.67 | memory | — | — | — |
| `fused_moe_mlp` | decode | 2/75 | 2.265e+10 | 2.458e+6 | 2.458e+6 | 4.530e+10 | 4.915e+6 | 2.458e+6 | 0.50 | memory | — | — | — |
| `linear` | prefill | 28/700 | 3.466e+13 | 0 | 0 | 3.385e+10 | 1.529e+10 | 2.195e+10 | 487.57 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 28/700 | 1.693e+10 | 0 | 0 | 3.385e+10 | 7.465e+6 | 1.072e+7 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `dsa_sparse_mla` | prefill | 3/78 | 3.352e+12 | 0 | 0 | 0 | 4.404e+10 | 4.713e+10 | 36.76 | memory | — | — | — |
| `dsa_sparse_mla` | decode | 3/78 | 3.272e+9 | 0 | 0 | 0 | 2.259e+8 | 4.345e+7 | 12.15 | memory | — | — | — |
| `mla_query_compress` | prefill | 3/78 | 2.010e+12 | 0 | 0 | 1.963e+9 | 1.963e+9 | 6.543e+8 | 438.86 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | decode | 3/78 | 9.815e+8 | 0 | 0 | 1.963e+9 | 9.585e+5 | 3.195e+5 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `dsa_indexer` | prefill | 3/78 | 6.703e+11 | 2.095e+10 | 0 | 0 | 6.550e+10 | 3.408e+10 | 6.73 | memory | 计算✓ 字节✓ | 1.074e+9 | 0 |
| `dsa_indexer` | decode | 3/78 | 1.309e+9 | 4.089e+7 | 0 | 0 | 2.064e+8 | 6.328e+7 | 4.85 | memory | 计算✓ 字节✓ | 2.097e+6 | 0 |
| `mla_kv_compress` | prefill | 3/78 | 5.653e+11 | 0 | 0 | 5.521e+8 | 1.963e+9 | 1.840e+8 | 209.45 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | decode | 3/78 | 2.760e+8 | 0 | 0 | 5.521e+8 | 9.585e+5 | 8.986e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 2/75 | 0 | 1.510e+10 | 0 | 0 | 1.510e+10 | 1.887e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 2/75 | 0 | 7.373e+6 | 0 | 0 | 7.374e+6 | 9.216e+5 | 0.00 | memory | — | — | — |
| `residual_add` | prefill | 6/156 | 0 | 1.963e+9 | 0 | 0 | 7.852e+9 | 3.926e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 6/156 | 0 | 9.585e+5 | 0 | 0 | 3.834e+6 | 1.917e+6 | 0.00 | memory | — | — | — |
| `rope` | prefill | 3/78 | 0 | 3.926e+9 | 0 | 0 | 5.234e+9 | 2.617e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 3/78 | 0 | 1.917e+6 | 0 | 0 | 2.556e+6 | 1.278e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rmsnorm` | prefill | 19/391 | 0 | 9.619e+9 | 8.008e+5 | 2.349e+6 | 4.810e+9 | 4.810e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.007e+8 | 0 |
| `rmsnorm` | decode | 19/391 | 0 | 4.697e+6 | 3.910e+2 | 2.349e+6 | 2.349e+6 | 2.349e+6 | 0.00 | memory | 计算✓ 字节✓ | 4.916e+4 | 0 |
| `moe_add` | prefill | 2/75 | 0 | 9.437e+8 | 0 | 0 | 3.775e+9 | 1.887e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 2/75 | 0 | 4.608e+5 | 0 | 0 | 1.843e+6 | 9.216e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 2/75 | 0 | 0 | 0 | 0 | 1.887e+9 | 1.510e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 2/75 | 0 | 0 | 0 | 0 | 9.216e+5 | 7.373e+6 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 3/78 | 0 | 7.801e+8 | 7.801e+8 | 0 | 1.560e+9 | 7.801e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.007e+8 | 0 |
| `swiglu` | decode | 3/78 | 0 | 3.809e+5 | 3.809e+5 | 0 | 7.619e+5 | 3.809e+5 | 0.00 | memory | 计算✓ 字节✓ | 4.915e+4 | 0 |
| `topk` | prefill | 2/75 | 0 | 4.040e+7 | 1.229e+6 | 0 | 7.864e+7 | 4.915e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 2/75 | 0 | 1.973e+4 | 6.000e+2 | 0 | 3.840e+4 | 2.400e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 2.517e+7 | 2.517e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.229e+4 | 1.229e+4 | 0.00 | memory | — | — | — |
| `mla_kv_split` | prefill | 3/78 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | decode | 3/78 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S04 · moonshotai/Kimi-K2-Instruct

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_mlp` | prefill | 1/60 | 4.329e+13 | 4.027e+9 | 4.027e+9 | 2.029e+12 | 8.053e+9 | 4.027e+9 | 21.21 | memory | — | — | — |
| `fused_moe_mlp` | decode | 1/60 | 2.114e+10 | 1.966e+6 | 1.966e+6 | 4.228e+10 | 3.932e+6 | 1.966e+6 | 0.50 | memory | — | — | — |
| `linear` | prefill | 14/427 | 1.971e+13 | 0 | 0 | 1.925e+10 | 8.510e+9 | 1.267e+10 | 487.52 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 14/427 | 9.623e+9 | 0 | 0 | 1.925e+10 | 4.155e+6 | 6.186e+6 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `matmul` | prefill | 4/122 | 2.621e+12 | 0 | 0 | 0 | 1.960e+10 | 1.843e+10 | 68.93 | memory | — | — | — |
| `matmul` | decode | 4/122 | 5.117e+9 | 0 | 0 | 0 | 3.213e+8 | 3.298e+7 | 14.44 | memory | — | — | — |
| `mla_query_compress` | prefill | 2/61 | 1.375e+12 | 0 | 0 | 1.343e+9 | 1.791e+9 | 3.838e+8 | 390.98 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | decode | 2/61 | 6.716e+8 | 0 | 0 | 1.343e+9 | 8.745e+5 | 1.874e+5 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | prefill | 2/61 | 5.158e+11 | 0 | 0 | 5.037e+8 | 1.791e+9 | 1.439e+8 | 211.51 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | decode | 2/61 | 2.519e+8 | 0 | 0 | 5.037e+8 | 8.745e+5 | 7.027e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | prefill | 2/61 | 0 | 2.457e+10 | 1.638e+10 | 0 | 1.638e+10 | 1.638e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 2/61 | 0 | 4.797e+7 | 3.198e+7 | 0 | 3.198e+7 | 3.198e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 1/60 | 0 | 1.409e+10 | 0 | 0 | 1.409e+10 | 1.762e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 1/60 | 0 | 6.881e+6 | 0 | 0 | 6.882e+6 | 8.602e+5 | 0.00 | memory | — | — | — |
| `rope` | prefill | 2/61 | 0 | 9.211e+9 | 0 | 0 | 1.228e+10 | 6.140e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 2/61 | 0 | 4.497e+6 | 0 | 0 | 5.997e+6 | 2.998e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `residual_add` | prefill | 4/122 | 0 | 1.791e+9 | 0 | 0 | 7.164e+9 | 3.582e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 4/122 | 0 | 8.745e+5 | 0 | 0 | 3.498e+6 | 1.749e+6 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 9/245 | 0 | 8.245e+9 | 5.018e+5 | 2.013e+6 | 4.123e+9 | 4.123e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.175e+8 | 0 |
| `rmsnorm` | decode | 9/245 | 0 | 4.026e+6 | 2.450e+2 | 2.013e+6 | 2.013e+6 | 2.013e+6 | 0.00 | memory | 计算✓ 字节✓ | 5.735e+4 | 0 |
| `moe_add` | prefill | 1/60 | 0 | 8.808e+8 | 0 | 0 | 3.523e+9 | 1.762e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 1/60 | 0 | 4.301e+5 | 0 | 0 | 1.720e+6 | 8.602e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 1/60 | 0 | 0 | 0 | 0 | 1.762e+9 | 1.409e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 1/60 | 0 | 0 | 0 | 0 | 8.602e+5 | 6.881e+6 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 2/61 | 0 | 5.788e+8 | 5.788e+8 | 0 | 1.158e+9 | 5.788e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.510e+8 | 0 |
| `swiglu` | decode | 2/61 | 0 | 2.826e+5 | 2.826e+5 | 0 | 5.652e+5 | 2.826e+5 | 0.00 | memory | 计算✓ 字节✓ | 7.373e+4 | 0 |
| `topk` | prefill | 1/60 | 0 | 4.805e+7 | 9.830e+5 | 0 | 9.437e+7 | 3.932e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 1/60 | 0 | 2.346e+4 | 4.800e+2 | 0 | 4.608e+4 | 1.920e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 2.936e+7 | 2.936e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.434e+4 | 1.434e+4 | 0.00 | memory | — | — | — |
| `mla_kv_split` | prefill | 2/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | decode | 2/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S05 · deepseek-ai/DeepSeek-V4-Pro

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | prefill | 547/546 | 1.745e+14 | 0 | 0 | 1.704e+11 | 2.979e+10 | 2.773e+10 | 765.56 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 547/546 | 8.521e+10 | 0 | 0 | 1.704e+11 | 1.455e+7 | 1.354e+7 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `fused_moe_mlp` | prefill | 61/61 | 4.952e+13 | 4.605e+9 | 4.605e+9 | 3.095e+12 | 9.211e+9 | 4.605e+9 | 15.93 | memory | — | — | — |
| `fused_moe_mlp` | decode | 61/61 | 2.418e+10 | 2.249e+6 | 2.249e+6 | 4.836e+10 | 4.497e+6 | 2.249e+6 | 0.50 | memory | — | — | — |
| `dsv4_sparse_mla` | prefill | 30/30 | 2.064e+12 | 0 | 0 | 0 | 1.631e+10 | 1.612e+10 | 63.64 | memory | — | — | — |
| `dsv4_sparse_mla` | decode | 30/30 | 4.027e+9 | 0 | 0 | 0 | 8.657e+7 | 2.359e+7 | 36.55 | memory | — | — | — |
| `mla_kv_compress` | prefill | 61/61 | 1.368e+12 | 0 | 0 | 1.336e+9 | 1.791e+9 | 3.817e+8 | 389.89 | matrix | — | — | — |
| `mla_kv_compress` | decode | 61/61 | 6.679e+8 | 0 | 0 | 1.336e+9 | 8.745e+5 | 1.864e+5 | 0.50 | memory | — | — | — |
| `dsv4_indexer` | prefill | 30/30 | 5.156e+11 | 1.611e+10 | 0 | 0 | 4.987e+10 | 2.494e+10 | 6.89 | memory | — | — | — |
| `dsv4_indexer` | decode | 30/30 | 1.007e+9 | 3.146e+7 | 0 | 0 | 1.268e+8 | 4.781e+7 | 5.76 | memory | — | — | — |
| `dsv4_compressed_attention` | prefill | 31/31 | 1.331e+11 | 0 | 0 | 0 | 8.847e+9 | 8.907e+9 | 7.50 | memory | — | — | — |
| `dsv4_compressed_attention` | decode | 31/31 | 4.063e+6 | 0 | 0 | 0 | 1.067e+7 | 4.603e+6 | 0.27 | memory | — | — | — |
| `mhc_fused_post_pre` | prefill | 61/61 | 8.597e+10 | 6.350e+9 | 3.662e+9 | 1.688e+8 | 1.673e+10 | 7.478e+9 | 3.53 | memory | — | — | — |
| `mhc_fused_post_pre` | decode | 61/61 | 4.198e+7 | 3.101e+6 | 1.788e+6 | 1.688e+8 | 8.169e+6 | 3.651e+6 | 0.23 | memory | — | — | — |
| `mhc_pre` | prefill | 61/61 | 8.597e+10 | 5.455e+9 | 1.871e+9 | 1.688e+8 | 1.494e+10 | 5.687e+9 | 4.13 | memory | — | — | — |
| `mhc_pre` | decode | 61/61 | 4.198e+7 | 2.663e+6 | 9.136e+5 | 1.688e+8 | 7.295e+6 | 2.777e+6 | 0.23 | memory | — | — | — |
| `rope` | prefill | 122/122 | 0 | 4.951e+10 | 0 | 0 | 6.601e+10 | 3.300e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 122/122 | 0 | 2.417e+7 | 0 | 0 | 3.223e+7 | 1.612e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 61/61 | 0 | 1.075e+10 | 0 | 0 | 1.075e+10 | 1.791e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 61/61 | 0 | 5.247e+6 | 0 | 0 | 5.248e+6 | 8.745e+5 | 0.00 | memory | — | — | — |
| `residual_add` | prefill | 122/122 | 0 | 1.791e+9 | 0 | 0 | 7.164e+9 | 3.582e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 122/122 | 0 | 8.745e+5 | 0 | 0 | 3.498e+6 | 1.749e+6 | 0.00 | memory | — | — | — |
| `moe_add` | prefill | 61/61 | 0 | 8.955e+8 | 0 | 0 | 3.582e+9 | 1.791e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 61/61 | 0 | 4.372e+5 | 0 | 0 | 1.749e+6 | 8.745e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 61/61 | 0 | 0 | 0 | 0 | 1.791e+9 | 1.075e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 61/61 | 0 | 0 | 0 | 0 | 8.745e+5 | 5.247e+6 | 0.00 | memory | — | — | — |
| `mhc_post` | prefill | 1/1 | 1.409e+9 | 1.468e+7 | 0 | 0 | 1.762e+8 | 2.946e+7 | 6.85 | memory | — | — | — |
| `mhc_post` | decode | 1/1 | 6.881e+5 | 7.168e+3 | 0 | 0 | 8.602e+4 | 1.438e+4 | 6.85 | memory | — | — | — |
| `swiglu` | prefill | 61/61 | 0 | 7.676e+8 | 7.676e+8 | 0 | 1.535e+9 | 7.676e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `swiglu` | decode | 61/61 | 0 | 3.748e+5 | 3.748e+5 | 0 | 7.496e+5 | 3.748e+5 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rmsnorm` | prefill | 126/123 | 0 | 1.082e+9 | 2.519e+5 | 2.642e+5 | 5.411e+8 | 5.411e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.175e+8 | 0 |
| `rmsnorm` | decode | 126/123 | 0 | 5.283e+5 | 1.230e+2 | 2.642e+5 | 2.642e+5 | 2.642e+5 | 0.00 | memory | 计算✓ 字节✓ | 5.735e+4 | 0 |
| `topk` | prefill | 58/58 | 0 | 4.621e+7 | 7.127e+5 | 0 | 9.123e+7 | 2.851e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 58/58 | 0 | 2.256e+4 | 3.480e+2 | 0 | 4.454e+4 | 1.392e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `mhc_contract` | prefill | 1/1 | 0 | 1.468e+7 | 0 | 0 | 5.872e+7 | 2.936e+7 | 0.00 | memory | — | — | — |
| `mhc_contract` | decode | 1/1 | 0 | 7.168e+3 | 0 | 0 | 2.867e+4 | 1.434e+4 | 0.00 | memory | — | — | — |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 2.936e+7 | 2.936e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.434e+4 | 1.434e+4 | 0.00 | memory | — | — | — |
| `dsv4_hash_route` | prefill | 3/3 | 0 | 0 | 0 | 0 | 7.373e+4 | 7.373e+4 | 0.00 | memory | — | — | — |
| `dsv4_hash_route` | decode | 3/3 | 0 | 0 | 0 | 0 | 3.600e+1 | 3.600e+1 | 0.00 | memory | — | — | — |
| `split` | prefill | 61/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | decode | 61/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S06 · moonshotai/Kimi-K2.5

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_mlp` | prefill | 1/60 | 4.329e+13 | 4.027e+9 | 4.027e+9 | 2.029e+12 | 8.053e+9 | 4.027e+9 | 21.21 | memory | — | — | — |
| `fused_moe_mlp` | decode | 1/60 | 2.114e+10 | 1.966e+6 | 1.966e+6 | 4.228e+10 | 3.932e+6 | 1.966e+6 | 0.50 | memory | — | — | — |
| `linear` | prefill | 21/538 | 2.019e+13 | 1.206e+7 | 0 | 2.018e+10 | 8.959e+9 | 1.325e+10 | 476.19 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 21/538 | 4.868e+11 | 1.206e+7 | 0 | 2.018e+10 | 4.533e+8 | 5.892e+8 | 22.94 | memory | 计算✓ 字节✓ | 0 | 0 |
| `matmul` | prefill | 6/176 | 2.654e+12 | 0 | 0 | 0 | 2.024e+10 | 1.895e+10 | 67.72 | memory | — | — | — |
| `matmul` | decode | 6/176 | 7.035e+10 | 0 | 0 | 0 | 1.418e+9 | 1.003e+9 | 29.06 | memory | — | — | — |
| `mla_query_compress` | prefill | 2/61 | 1.375e+12 | 0 | 0 | 1.343e+9 | 1.791e+9 | 3.838e+8 | 390.98 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | decode | 2/61 | 6.716e+8 | 0 | 0 | 1.343e+9 | 8.745e+5 | 1.874e+5 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | prefill | 2/61 | 5.158e+11 | 0 | 0 | 5.037e+8 | 1.791e+9 | 1.439e+8 | 211.51 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | decode | 2/61 | 2.519e+8 | 0 | 0 | 5.037e+8 | 8.745e+5 | 7.027e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | prefill | 3/88 | 0 | 2.525e+10 | 1.684e+10 | 0 | 1.684e+10 | 1.684e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 3/88 | 0 | 1.407e+9 | 9.380e+8 | 0 | 9.380e+8 | 9.380e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 1/60 | 0 | 1.409e+10 | 0 | 0 | 1.409e+10 | 1.762e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 1/60 | 0 | 6.881e+6 | 0 | 0 | 6.882e+6 | 8.602e+5 | 0.00 | memory | — | — | — |
| `rope` | prefill | 2/61 | 0 | 9.211e+9 | 0 | 0 | 1.228e+10 | 6.140e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 2/61 | 0 | 4.497e+6 | 0 | 0 | 5.997e+6 | 2.998e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `residual_add` | prefill | 4/122 | 0 | 1.791e+9 | 0 | 0 | 7.164e+9 | 3.582e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 4/122 | 0 | 8.745e+5 | 0 | 0 | 3.498e+6 | 1.749e+6 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 12/300 | 0 | 8.505e+9 | 5.581e+5 | 2.142e+6 | 4.253e+9 | 4.253e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.175e+8 | 0 |
| `rmsnorm` | decode | 12/300 | 0 | 2.635e+8 | 5.657e+4 | 2.142e+6 | 1.318e+8 | 1.318e+8 | 0.00 | memory | 计算✓ 字节✓ | 5.735e+4 | 0 |
| `moe_add` | prefill | 1/60 | 0 | 8.808e+8 | 0 | 0 | 3.523e+9 | 1.762e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 1/60 | 0 | 4.301e+5 | 0 | 0 | 1.720e+6 | 8.602e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 1/60 | 0 | 0 | 0 | 0 | 1.762e+9 | 1.409e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 1/60 | 0 | 0 | 0 | 0 | 8.602e+5 | 6.881e+6 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 2/61 | 0 | 5.788e+8 | 5.788e+8 | 0 | 1.158e+9 | 5.788e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.510e+8 | 0 |
| `swiglu` | decode | 2/61 | 0 | 2.826e+5 | 2.826e+5 | 0 | 5.652e+5 | 2.826e+5 | 0.00 | memory | 计算✓ 字节✓ | 7.373e+4 | 0 |
| `vision_activation` | prefill | 2/28 | 0 | 2.474e+8 | 2.474e+8 | 0 | 4.949e+8 | 2.474e+8 | 0.00 | memory | — | — | — |
| `vision_activation` | decode | 2/28 | 0 | 2.474e+8 | 2.474e+8 | 0 | 4.949e+8 | 2.474e+8 | 0.00 | memory | — | — | — |
| `topk` | prefill | 1/60 | 0 | 4.805e+7 | 9.830e+5 | 0 | 9.437e+7 | 3.932e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 1/60 | 0 | 2.346e+4 | 4.800e+2 | 0 | 4.608e+4 | 1.920e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 2.936e+7 | 2.936e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.434e+4 | 1.434e+4 | 0.00 | memory | — | — | — |
| `vision_position` | prefill | 1/1 | 0 | 1.180e+6 | 0 | 0 | 4.719e+6 | 2.359e+6 | 0.00 | memory | — | — | — |
| `vision_position` | decode | 1/1 | 0 | 1.180e+6 | 0 | 0 | 4.719e+6 | 2.359e+6 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 1/27 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 1/27 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | prefill | 2/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | decode | 2/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S07 · deepseek-ai/DeepSeek-V3.1

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_mlp` | prefill | 2/58 | 4.185e+13 | 3.892e+9 | 3.892e+9 | 1.308e+12 | 7.785e+9 | 3.892e+9 | 31.72 | memory | — | — | — |
| `fused_moe_mlp` | decode | 2/58 | 2.043e+10 | 1.901e+6 | 1.901e+6 | 4.087e+10 | 3.801e+6 | 1.901e+6 | 0.50 | memory | — | — | — |
| `linear` | prefill | 22/425 | 3.127e+13 | 0 | 0 | 3.053e+10 | 1.063e+10 | 1.993e+10 | 511.78 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 22/425 | 1.527e+10 | 0 | 0 | 3.053e+10 | 5.192e+6 | 9.730e+6 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `matmul` | prefill | 6/122 | 5.242e+12 | 0 | 0 | 0 | 3.905e+10 | 3.686e+10 | 69.06 | memory | — | — | — |
| `matmul` | decode | 6/122 | 1.023e+10 | 0 | 0 | 0 | 3.548e+8 | 6.596e+7 | 24.32 | memory | — | — | — |
| `mla_query_compress` | prefill | 3/61 | 1.375e+12 | 0 | 0 | 1.343e+9 | 1.791e+9 | 3.838e+8 | 390.98 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | decode | 3/61 | 6.716e+8 | 0 | 0 | 1.343e+9 | 8.745e+5 | 1.874e+5 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | prefill | 3/61 | 5.158e+11 | 0 | 0 | 5.037e+8 | 1.791e+9 | 1.439e+8 | 211.51 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | decode | 3/61 | 2.519e+8 | 0 | 0 | 5.037e+8 | 8.745e+5 | 7.027e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | prefill | 3/61 | 0 | 4.915e+10 | 3.277e+10 | 0 | 3.277e+10 | 3.277e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 3/61 | 0 | 9.594e+7 | 6.396e+7 | 0 | 6.396e+7 | 6.396e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | prefill | 3/61 | 0 | 1.842e+10 | 0 | 0 | 2.456e+10 | 1.228e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 3/61 | 0 | 8.995e+6 | 0 | 0 | 1.199e+7 | 5.997e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 2/58 | 0 | 1.362e+10 | 0 | 0 | 1.362e+10 | 1.703e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 2/58 | 0 | 6.652e+6 | 0 | 0 | 6.653e+6 | 8.315e+5 | 0.00 | memory | — | — | — |
| `residual_add` | prefill | 6/122 | 0 | 1.791e+9 | 0 | 0 | 7.164e+9 | 3.582e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 6/122 | 0 | 8.745e+5 | 0 | 0 | 3.498e+6 | 1.749e+6 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 16/245 | 0 | 8.245e+9 | 5.018e+5 | 2.013e+6 | 4.123e+9 | 4.123e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.175e+8 | 0 |
| `rmsnorm` | decode | 16/245 | 0 | 4.026e+6 | 2.450e+2 | 2.013e+6 | 2.013e+6 | 2.013e+6 | 0.00 | memory | 计算✓ 字节✓ | 5.735e+4 | 0 |
| `moe_add` | prefill | 2/58 | 0 | 8.514e+8 | 0 | 0 | 3.406e+9 | 1.703e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 2/58 | 0 | 4.157e+5 | 0 | 0 | 1.663e+6 | 8.315e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 2/58 | 0 | 0 | 0 | 0 | 1.703e+9 | 1.362e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 2/58 | 0 | 0 | 0 | 0 | 8.315e+5 | 6.652e+6 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 3/61 | 0 | 7.130e+8 | 7.130e+8 | 0 | 1.426e+9 | 7.130e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.510e+8 | 0 |
| `swiglu` | decode | 3/61 | 0 | 3.482e+5 | 3.482e+5 | 0 | 6.963e+5 | 3.482e+5 | 0.00 | memory | 计算✓ 字节✓ | 7.373e+4 | 0 |
| `topk` | prefill | 2/58 | 0 | 3.124e+7 | 9.503e+5 | 0 | 6.082e+7 | 3.801e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 2/58 | 0 | 1.525e+4 | 4.640e+2 | 0 | 2.970e+4 | 1.856e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 2.936e+7 | 2.936e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.434e+4 | 1.434e+4 | 0.00 | memory | — | — | — |
| `mla_kv_split` | prefill | 3/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | decode | 3/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S08 · zai-org/GLM-5.3-Flash

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_mlp` | prefill | 23/42 | 1.732e+13 | 2.819e+9 | 2.819e+9 | 6.088e+11 | 5.637e+9 | 2.819e+9 | 28.05 | memory | — | — | — |
| `fused_moe_mlp` | decode | 23/42 | 8.456e+9 | 1.376e+6 | 1.376e+6 | 1.691e+10 | 2.753e+6 | 1.376e+6 | 0.50 | memory | — | — | — |
| `linear` | prefill | 214/494 | 1.683e+13 | 0 | 0 | 1.739e+10 | 5.679e+9 | 1.153e+10 | 486.34 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 214/494 | 1.480e+11 | 0 | 0 | 1.739e+10 | 1.155e+8 | 1.843e+8 | 8.37 | memory | 计算✓ 字节✓ | 0 | 0 |
| `dsa_sparse_mla` | prefill | 11/11 | 7.563e+11 | 0 | 0 | 0 | 6.762e+9 | 6.647e+9 | 56.40 | memory | — | — | — |
| `dsa_sparse_mla` | decode | 11/11 | 7.382e+8 | 0 | 0 | 0 | 2.924e+7 | 6.128e+6 | 20.87 | memory | — | — | — |
| `gated_delta_attention` | prefill | 13/34 | 2.190e+11 | 1.141e+9 | 2.089e+5 | 1.123e+6 | 2.442e+9 | 2.442e+9 | 44.84 | memory | 计算✓ 字节✓ | 0 | 0 |
| `gated_delta_attention` | decode | 13/34 | 1.070e+8 | 3.565e+7 | 6.528e+3 | 1.123e+6 | 7.632e+7 | 7.632e+7 | 0.70 | memory | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | prefill | 11/11 | 1.417e+11 | 0 | 0 | 1.384e+8 | 1.845e+8 | 6.921e+7 | 361.41 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | decode | 11/11 | 6.921e+7 | 0 | 0 | 1.384e+8 | 9.011e+4 | 3.379e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | prefill | 11/11 | 4.724e+10 | 0 | 0 | 4.614e+7 | 1.845e+8 | 2.307e+7 | 186.18 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | decode | 11/11 | 2.307e+7 | 0 | 0 | 4.614e+7 | 9.011e+4 | 1.126e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `mhc_fused_post_pre` | prefill | 24/45 | 3.624e+10 | 2.703e+9 | 1.569e+9 | 7.115e+7 | 7.246e+9 | 3.251e+9 | 3.43 | memory | — | — | — |
| `mhc_fused_post_pre` | decode | 24/45 | 1.769e+7 | 1.320e+6 | 7.661e+5 | 7.115e+7 | 3.538e+6 | 1.588e+6 | 0.23 | memory | — | — | — |
| `mhc_pre` | prefill | 24/45 | 3.624e+10 | 2.325e+9 | 8.140e+8 | 7.115e+7 | 6.491e+9 | 2.496e+9 | 4.00 | memory | — | — | — |
| `mhc_pre` | decode | 24/45 | 1.769e+7 | 1.135e+6 | 3.975e+5 | 7.115e+7 | 3.169e+6 | 1.219e+6 | 0.23 | memory | — | — | — |
| `causal_conv1d` | prefill | 13/34 | 6.845e+9 | 0 | 0 | 6.685e+6 | 3.423e+9 | 3.423e+9 | 1.00 | memory | — | — | — |
| `causal_conv1d` | decode | 13/34 | 3.342e+6 | 0 | 0 | 6.685e+6 | 6.685e+6 | 6.685e+6 | 0.17 | memory | — | — | — |
| `dsa_kpool_indexer` | prefill | 11/11 | 5.917e+9 | 1.871e+8 | 0 | 0 | 7.912e+8 | 3.754e+8 | 5.07 | memory | — | — | — |
| `dsa_kpool_indexer` | decode | 11/11 | 4.614e+7 | 5.767e+6 | 0 | 0 | 1.599e+7 | 2.233e+6 | 2.53 | memory | — | — | — |
| `moe_combine` | prefill | 23/42 | 0 | 5.637e+9 | 0 | 0 | 5.639e+9 | 7.046e+8 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 23/42 | 0 | 2.753e+6 | 0 | 0 | 2.753e+6 | 3.441e+5 | 0.00 | memory | — | — | — |
| `residual_add` | prefill | 48/90 | 0 | 7.550e+8 | 0 | 0 | 3.020e+9 | 1.510e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 48/90 | 0 | 3.686e+5 | 0 | 0 | 1.475e+6 | 7.373e+5 | 0.00 | memory | — | — | — |
| `rope` | prefill | 11/11 | 0 | 2.215e+9 | 0 | 0 | 2.953e+9 | 1.476e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 11/11 | 0 | 1.081e+6 | 0 | 0 | 1.442e+6 | 7.209e+5 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `gated_rmsnorm` | prefill | 13/34 | 0 | 2.852e+9 | 1.141e+9 | 8.704e+3 | 2.282e+9 | 1.141e+9 | 0.00 | memory | 计算✓ 字节✓ | 6.713e+7 | 0 |
| `gated_rmsnorm` | decode | 13/34 | 0 | 1.393e+6 | 5.571e+5 | 8.704e+3 | 1.114e+6 | 5.571e+5 | 0.00 | memory | 计算✓ 字节✓ | 3.278e+4 | 0 |
| `matmul` | prefill | 2/48 | 1.617e+9 | 0 | 0 | 0 | 6.301e+7 | 3.785e+7 | 16.03 | memory | — | — | — |
| `matmul` | decode | 2/48 | 3.221e+9 | 0 | 0 | 0 | 8.808e+7 | 6.291e+7 | 21.33 | memory | — | — | — |
| `moe_add` | prefill | 23/42 | 0 | 3.523e+8 | 0 | 0 | 1.409e+9 | 7.046e+8 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 23/42 | 0 | 1.720e+5 | 0 | 0 | 6.881e+5 | 3.441e+5 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 26/70 | 0 | 5.589e+8 | 5.589e+8 | 0 | 1.118e+9 | 5.589e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.007e+8 | 0 |
| `swiglu` | decode | 26/70 | 0 | 5.582e+7 | 5.582e+7 | 0 | 1.116e+8 | 5.582e+7 | 0.00 | memory | 计算✓ 字节✓ | 4.915e+4 | 0 |
| `mhc_post` | prefill | 1/1 | 8.053e+8 | 8.389e+6 | 0 | 0 | 1.007e+8 | 1.688e+7 | 6.85 | memory | — | — | — |
| `mhc_post` | decode | 1/1 | 3.932e+5 | 4.096e+3 | 0 | 0 | 4.915e+4 | 8.240e+3 | 6.85 | memory | — | — | — |
| `moe_dispatch` | prefill | 23/42 | 0 | 0 | 0 | 0 | 7.046e+8 | 5.637e+9 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 23/42 | 0 | 0 | 0 | 0 | 3.441e+5 | 2.753e+6 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 41/84 | 0 | 2.883e+8 | 8.243e+4 | 1.708e+5 | 1.442e+8 | 1.442e+8 | 0.00 | memory | 计算✓ 字节✓ | 6.713e+7 | 0 |
| `rmsnorm` | decode | 41/84 | 0 | 5.882e+7 | 1.283e+4 | 1.708e+5 | 2.942e+7 | 2.942e+7 | 0.00 | memory | 计算✓ 字节✓ | 3.278e+4 | 0 |
| `topk` | prefill | 23/42 | 0 | 2.537e+7 | 6.881e+5 | 0 | 4.955e+7 | 2.753e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 23/42 | 0 | 1.239e+4 | 3.360e+2 | 0 | 2.419e+4 | 1.344e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `mhc_contract` | prefill | 1/1 | 0 | 8.389e+6 | 0 | 0 | 3.355e+7 | 1.678e+7 | 0.00 | memory | — | — | — |
| `mhc_contract` | decode | 1/1 | 0 | 4.096e+3 | 0 | 0 | 1.638e+4 | 8.192e+3 | 0.00 | memory | — | — | — |
| `softmax` | prefill | 1/24 | 0 | 3.790e+7 | 2.526e+7 | 0 | 2.526e+7 | 2.526e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 1/24 | 0 | 7.550e+7 | 5.033e+7 | 0 | 5.033e+7 | 5.033e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 1.678e+7 | 1.678e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 8.192e+3 | 8.192e+3 | 0.00 | memory | — | — | — |
| `vision_position` | prefill | 1/1 | 0 | 2.621e+5 | 0 | 0 | 1.049e+6 | 5.243e+5 | 0.00 | memory | — | — | — |
| `vision_position` | decode | 1/1 | 0 | 2.621e+5 | 0 | 0 | 1.049e+6 | 5.243e+5 | 0.00 | memory | — | — | — |
| `vision_merge` | prefill | 1/1 | 0 | 0 | 0 | 0 | 5.243e+5 | 5.243e+5 | 0.00 | memory | — | — | — |
| `vision_merge` | decode | 1/1 | 0 | 0 | 0 | 0 | 5.243e+5 | 5.243e+5 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 1/24 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 1/24 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | prefill | 11/11 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | decode | 11/11 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S09 · MiniMaxAI/MiniMax-M3

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | prefill | 25/488 | 2.647e+13 | 0 | 0 | 2.390e+10 | 1.175e+10 | 1.169e+10 | 559.21 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 25/488 | 3.317e+12 | 0 | 0 | 2.390e+10 | 2.996e+9 | 3.903e+9 | 107.71 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `fused_moe_mlp` | prefill | 2/57 | 2.644e+13 | 2.869e+9 | 2.869e+9 | 8.262e+11 | 5.738e+9 | 2.869e+9 | 31.67 | memory | — | — | — |
| `fused_moe_mlp` | decode | 2/57 | 1.291e+10 | 1.401e+6 | 1.401e+6 | 2.582e+10 | 2.802e+6 | 1.401e+6 | 0.50 | memory | — | — | — |
| `minimax_sparse_attention` | prefill | 2/57 | 1.959e+12 | 0 | 0 | 0 | 3.277e+10 | 3.277e+10 | 29.90 | memory | — | — | — |
| `minimax_sparse_attention` | decode | 2/57 | 2.032e+9 | 0 | 0 | 0 | 2.867e+8 | 3.280e+7 | 6.36 | memory | — | — | — |
| `matmul` | prefill | 4/70 | 1.204e+12 | 0 | 0 | 0 | 1.596e+10 | 1.509e+10 | 38.78 | memory | — | — | — |
| `matmul` | decode | 4/70 | 2.202e+12 | 0 | 0 | 0 | 2.882e+10 | 2.795e+10 | 38.79 | memory | — | — | — |
| `minimax_sparse_indexer` | prefill | 2/57 | 6.123e+10 | 1.435e+9 | 0 | 0 | 3.027e+9 | 1.958e+9 | 12.28 | memory | 计算✓ 字节✓ | 1.343e+8 | 0 |
| `minimax_sparse_indexer` | decode | 2/57 | 1.195e+8 | 2.802e+6 | 0 | 0 | 6.450e+7 | 2.827e+6 | 1.78 | memory | 计算✓ 字节✓ | 2.621e+5 | 0 |
| `softmax` | prefill | 2/35 | 0 | 2.185e+10 | 1.457e+10 | 0 | 1.457e+10 | 1.457e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 2/35 | 0 | 4.128e+10 | 2.752e+10 | 0 | 2.752e+10 | 2.752e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `residual_add` | prefill | 6/120 | 0 | 1.510e+9 | 0 | 0 | 6.040e+9 | 3.020e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 6/120 | 0 | 7.373e+5 | 0 | 0 | 2.949e+6 | 1.475e+6 | 0.00 | memory | — | — | — |
| `moe_combine` | prefill | 2/57 | 0 | 5.738e+9 | 0 | 0 | 5.739e+9 | 1.434e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 2/57 | 0 | 2.802e+6 | 0 | 0 | 2.802e+6 | 7.004e+5 | 0.00 | memory | — | — | — |
| `gemma_rmsnorm` | prefill | 20/355 | 0 | 1.333e+10 | 7.270e+5 | 1.547e+6 | 5.334e+9 | 5.334e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.007e+8 | 0 |
| `gemma_rmsnorm` | decode | 20/355 | 0 | 6.510e+6 | 3.550e+2 | 1.547e+6 | 2.604e+6 | 2.604e+6 | 0.00 | memory | 计算✓ 字节✓ | 4.916e+4 | 0 |
| `rope` | prefill | 5/117 | 0 | 3.128e+9 | 0 | 0 | 4.171e+9 | 2.086e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 5/117 | 0 | 1.528e+6 | 0 | 0 | 2.037e+6 | 1.018e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `vision_activation` | prefill | 1/32 | 0 | 1.699e+9 | 1.699e+9 | 0 | 3.397e+9 | 1.699e+9 | 0.00 | memory | — | — | — |
| `vision_activation` | decode | 1/32 | 0 | 1.699e+9 | 1.699e+9 | 0 | 3.397e+9 | 1.699e+9 | 0.00 | memory | — | — | — |
| `moe_add` | prefill | 2/57 | 0 | 7.172e+8 | 0 | 0 | 2.869e+9 | 1.434e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 2/57 | 0 | 3.502e+5 | 0 | 0 | 1.401e+6 | 7.004e+5 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 3/60 | 0 | 7.550e+8 | 7.550e+8 | 0 | 1.510e+9 | 7.550e+8 | 0.00 | memory | 计算✓ 字节✓ | 2.517e+7 | 0 |
| `swiglu` | decode | 3/60 | 0 | 3.686e+5 | 3.686e+5 | 0 | 7.373e+5 | 3.686e+5 | 0.00 | memory | 计算✓ 字节✓ | 1.229e+4 | 0 |
| `moe_dispatch` | prefill | 2/57 | 0 | 0 | 0 | 0 | 1.434e+9 | 5.738e+9 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 2/57 | 0 | 0 | 0 | 0 | 7.004e+5 | 2.802e+6 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 2/64 | 0 | 1.698e+9 | 3.318e+5 | 1.638e+5 | 8.493e+8 | 8.493e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.007e+8 | 0 |
| `rmsnorm` | decode | 2/64 | 0 | 1.698e+9 | 3.318e+5 | 1.638e+5 | 8.493e+8 | 8.493e+8 | 0.00 | memory | 计算✓ 字节✓ | 4.916e+4 | 0 |
| `topk` | prefill | 2/57 | 0 | 1.529e+7 | 4.669e+5 | 0 | 2.988e+7 | 1.868e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 2/57 | 0 | 7.467e+3 | 2.280e+2 | 0 | 1.459e+4 | 9.120e+2 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `vision_position` | prefill | 1/1 | 0 | 6.636e+6 | 0 | 0 | 2.654e+7 | 1.327e+7 | 0.00 | memory | — | — | — |
| `vision_position` | decode | 1/1 | 0 | 6.636e+6 | 0 | 0 | 2.654e+7 | 1.327e+7 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 1/32 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 1/32 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | prefill | 3/60 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | decode | 3/60 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S10 · Qwen/Qwen3.8-2.4T-A95B

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | prefill | 330/691 | 9.613e+13 | 0 | 0 | 9.388e+10 | 2.396e+10 | 2.270e+10 | 684.05 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 330/691 | 4.694e+10 | 0 | 0 | 9.388e+10 | 1.170e+7 | 1.108e+7 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `fused_moe_mlp` | prefill | 47/92 | 9.483e+13 | 7.718e+9 | 7.718e+9 | 4.742e+12 | 1.544e+10 | 7.718e+9 | 19.90 | memory | — | — | — |
| `fused_moe_mlp` | decode | 47/92 | 4.631e+10 | 3.768e+6 | 3.768e+6 | 9.261e+10 | 7.537e+6 | 3.768e+6 | 0.50 | memory | — | — | — |
| `matmul` | prefill | 48/46 | 1.581e+12 | 0 | 0 | 0 | 7.913e+9 | 7.721e+9 | 101.15 | matrix | — | — | — |
| `matmul` | decode | 48/46 | 3.087e+9 | 0 | 0 | 0 | 3.987e+8 | 1.281e+7 | 7.50 | memory | — | — | — |
| `gated_delta_attention` | prefill | 23/69 | 8.891e+11 | 4.631e+9 | 8.479e+5 | 7.066e+4 | 9.532e+9 | 9.532e+9 | 46.63 | memory | 计算✓ 字节✓ | 0 | 0 |
| `gated_delta_attention` | decode | 23/69 | 4.341e+8 | 1.447e+8 | 2.650e+4 | 7.066e+4 | 2.979e+8 | 2.979e+8 | 0.73 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 47/92 | 0 | 3.087e+10 | 0 | 0 | 3.087e+10 | 3.087e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 47/92 | 0 | 1.507e+7 | 0 | 0 | 1.508e+7 | 1.507e+6 | 0.00 | memory | — | — | — |
| `causal_conv1d` | prefill | 23/69 | 1.158e+10 | 0 | 0 | 1.130e+7 | 5.788e+9 | 5.788e+9 | 1.00 | memory | — | — | — |
| `causal_conv1d` | decode | 23/69 | 5.652e+6 | 0 | 0 | 1.130e+7 | 1.130e+7 | 1.130e+7 | 0.17 | memory | — | — | — |
| `residual_add` | prefill | 94/184 | 0 | 3.087e+9 | 0 | 0 | 1.235e+10 | 6.174e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 94/184 | 0 | 1.507e+6 | 0 | 0 | 6.029e+6 | 3.015e+6 | 0.00 | memory | — | — | — |
| `gated_rmsnorm` | prefill | 23/69 | 0 | 1.158e+10 | 4.631e+9 | 1.766e+4 | 9.261e+9 | 4.631e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.342e+8 | 0 |
| `gated_rmsnorm` | decode | 23/69 | 0 | 5.652e+6 | 2.261e+6 | 1.766e+4 | 4.522e+6 | 2.261e+6 | 0.00 | memory | 计算✓ 字节✓ | 6.554e+4 | 0 |
| `gemma_rmsnorm` | prefill | 146/231 | 0 | 1.962e+10 | 4.731e+5 | 3.055e+6 | 7.848e+9 | 7.848e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.342e+8 | 0 |
| `gemma_rmsnorm` | decode | 146/231 | 0 | 9.579e+6 | 2.310e+2 | 3.055e+6 | 3.832e+6 | 3.832e+6 | 0.00 | memory | 计算✓ 字节✓ | 6.554e+4 | 0 |
| `softmax` | prefill | 24/23 | 0 | 9.266e+9 | 6.177e+9 | 0 | 6.177e+9 | 6.177e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 24/23 | 0 | 1.809e+7 | 1.206e+7 | 0 | 1.206e+7 | 1.206e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_add` | prefill | 47/92 | 0 | 1.544e+9 | 0 | 0 | 6.174e+9 | 3.087e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 47/92 | 0 | 7.537e+5 | 0 | 0 | 3.015e+6 | 1.507e+6 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 47/92 | 0 | 0 | 0 | 0 | 3.087e+9 | 3.087e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 47/92 | 0 | 0 | 0 | 0 | 1.507e+6 | 1.507e+7 | 0.00 | memory | — | — | — |
| `shared_expert_gate` | prefill | 47/92 | 0 | 1.544e+9 | 3.087e+9 | 0 | 3.087e+9 | 3.087e+9 | 0.00 | memory | — | — | — |
| `shared_expert_gate` | decode | 47/92 | 0 | 7.537e+5 | 1.507e+6 | 0 | 1.507e+6 | 1.507e+6 | 0.00 | memory | — | — | — |
| `attention_output_gate` | prefill | 24/23 | 0 | 7.718e+8 | 1.544e+9 | 0 | 1.544e+9 | 1.544e+9 | 0.00 | memory | — | — | — |
| `attention_output_gate` | decode | 24/23 | 0 | 3.768e+5 | 7.537e+5 | 0 | 7.537e+5 | 7.537e+5 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 47/92 | 0 | 7.718e+8 | 7.718e+8 | 0 | 1.544e+9 | 7.718e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `swiglu` | decode | 47/92 | 0 | 3.768e+5 | 3.768e+5 | 0 | 7.537e+5 | 3.768e+5 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | prefill | 24/23 | 0 | 6.150e+8 | 0 | 0 | 8.200e+8 | 4.100e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 24/23 | 0 | 3.003e+5 | 0 | 0 | 4.004e+5 | 2.002e+5 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `topk` | prefill | 47/92 | 0 | 9.816e+7 | 1.884e+6 | 0 | 1.929e+8 | 7.537e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 47/92 | 0 | 4.793e+4 | 9.200e+2 | 0 | 9.421e+4 | 3.680e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 3.355e+7 | 3.355e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.638e+4 | 1.638e+4 | 0.00 | memory | — | — | — |
| `qwen_qkvz_split` | prefill | 23/69 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `qwen_qkvz_split` | decode | 23/69 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | prefill | 24/23 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | decode | 24/23 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S11 · Qwen/Qwen3.8-Flash-Next

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | prefill | 118/328 | 6.795e+12 | 0 | 0 | 7.277e+9 | 3.234e+9 | 4.726e+9 | 445.97 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 118/328 | 2.600e+11 | 0 | 0 | 7.277e+9 | 2.552e+8 | 3.248e+8 | 33.09 | memory | 计算✓ 字节✓ | 0 | 0 |
| `fused_moe_mlp` | prefill | 27/48 | 4.832e+12 | 1.258e+9 | 1.258e+9 | 2.416e+11 | 2.517e+9 | 1.258e+9 | 19.69 | memory | — | — | — |
| `fused_moe_mlp` | decode | 27/48 | 2.359e+9 | 6.144e+5 | 6.144e+5 | 4.719e+9 | 1.229e+6 | 6.144e+5 | 0.50 | memory | — | — | — |
| `hyper_connection` | prefill | 55/97 | 1.310e+12 | 1.430e+10 | 4.196e+9 | 1.281e+9 | 2.462e+10 | 1.653e+10 | 30.87 | memory | — | — | — |
| `hyper_connection` | decode | 55/97 | 6.396e+8 | 6.984e+6 | 2.049e+6 | 1.281e+9 | 1.202e+7 | 8.071e+6 | 0.49 | memory | — | — | — |
| `qsa_sparse_attention` | prefill | 13/12 | 3.094e+11 | 0 | 0 | 0 | 2.870e+9 | 2.769e+9 | 54.86 | memory | — | — | — |
| `qsa_sparse_attention` | decode | 13/12 | 3.020e+8 | 0 | 0 | 0 | 5.289e+7 | 2.531e+6 | 5.45 | memory | — | — | — |
| `gated_delta_attention` | prefill | 14/36 | 1.739e+11 | 9.060e+8 | 1.659e+5 | 1.382e+4 | 1.883e+9 | 1.883e+9 | 46.20 | memory | 计算✓ 字节✓ | 0 | 0 |
| `gated_delta_attention` | decode | 14/36 | 8.493e+7 | 2.831e+7 | 5.184e+3 | 1.382e+4 | 5.883e+7 | 5.883e+7 | 0.72 | memory | 计算✓ 字节✓ | 0 | 0 |
| `ple` | prefill | 1/1 | 2.686e+10 | 3.146e+7 | 1.049e+7 | 2.623e+7 | 5.243e+7 | 5.243e+7 | 204.88 | matrix | — | — | — |
| `ple` | decode | 1/1 | 1.311e+7 | 1.536e+4 | 5.121e+3 | 2.623e+7 | 2.560e+4 | 2.560e+4 | 0.50 | memory | — | — | — |
| `matmul` | prefill | 2/54 | 1.034e+10 | 0 | 0 | 0 | 2.511e+8 | 1.794e+8 | 24.01 | memory | — | — | — |
| `matmul` | decode | 2/54 | 2.064e+10 | 0 | 0 | 0 | 3.941e+8 | 3.225e+8 | 28.80 | memory | — | — | — |
| `moe_combine` | prefill | 27/48 | 0 | 5.033e+9 | 0 | 0 | 5.035e+9 | 5.033e+8 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 27/48 | 0 | 2.458e+6 | 0 | 0 | 2.459e+6 | 2.458e+5 | 0.00 | memory | — | — | — |
| `causal_conv1d` | prefill | 14/36 | 3.020e+9 | 0 | 0 | 2.949e+6 | 1.510e+9 | 1.510e+9 | 1.00 | memory | — | — | — |
| `causal_conv1d` | decode | 14/36 | 1.475e+6 | 0 | 0 | 2.949e+6 | 2.949e+6 | 2.949e+6 | 0.17 | memory | — | — | — |
| `residual_add` | prefill | 54/96 | 0 | 5.033e+8 | 0 | 0 | 2.013e+9 | 1.007e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 54/96 | 0 | 2.458e+5 | 0 | 0 | 9.830e+5 | 4.915e+5 | 0.00 | memory | — | — | — |
| `gated_rmsnorm` | prefill | 14/36 | 0 | 2.265e+9 | 9.060e+8 | 9.216e+3 | 1.812e+9 | 9.060e+8 | 0.00 | memory | 计算✓ 字节✓ | 4.196e+7 | 0 |
| `gated_rmsnorm` | decode | 14/36 | 0 | 1.106e+6 | 4.424e+5 | 9.216e+3 | 8.847e+5 | 4.424e+5 | 0.00 | memory | 计算✓ 字节✓ | 2.049e+4 | 0 |
| `qsa_indexer` | prefill | 13/12 | 8.069e+8 | 2.755e+7 | 0 | 0 | 1.070e+8 | 1.196e+8 | 3.56 | memory | 计算✓ 字节✓ | 8.667e+6 | 0 |
| `qsa_indexer` | decode | 13/12 | 6.291e+6 | 4.866e+6 | 0 | 0 | 1.284e+7 | 1.751e+5 | 0.48 | memory | 计算✓ 字节✓ | 5.898e+5 | 0 |
| `moe_dispatch` | prefill | 27/48 | 0 | 0 | 0 | 0 | 5.033e+8 | 5.033e+9 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 27/48 | 0 | 0 | 0 | 0 | 2.458e+5 | 2.458e+6 | 0.00 | memory | — | — | — |
| `shared_expert_gate` | prefill | 27/48 | 0 | 2.517e+8 | 5.033e+8 | 0 | 5.033e+8 | 5.033e+8 | 0.00 | memory | — | — | — |
| `shared_expert_gate` | decode | 27/48 | 0 | 1.229e+5 | 2.458e+5 | 0 | 2.458e+5 | 2.458e+5 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 33/80 | 0 | 8.291e+8 | 8.288e+4 | 1.510e+5 | 4.146e+8 | 4.146e+8 | 0.00 | memory | 计算✓ 字节✓ | 4.196e+7 | 0 |
| `rmsnorm` | decode | 33/80 | 0 | 1.542e+8 | 3.171e+4 | 1.510e+5 | 7.714e+7 | 7.714e+7 | 0.00 | memory | 计算✓ 字节✓ | 2.049e+4 | 0 |
| `vision_activation` | prefill | 2/28 | 0 | 1.392e+8 | 1.392e+8 | 0 | 2.784e+8 | 1.392e+8 | 0.00 | memory | — | — | — |
| `vision_activation` | decode | 2/28 | 0 | 1.392e+8 | 1.392e+8 | 0 | 2.784e+8 | 1.392e+8 | 0.00 | memory | — | — | — |
| `rope` | prefill | 13/12 | 0 | 1.227e+8 | 0 | 0 | 1.636e+8 | 8.179e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 13/12 | 0 | 5.990e+4 | 0 | 0 | 7.987e+4 | 3.994e+4 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | prefill | 1/27 | 0 | 2.154e+8 | 1.436e+8 | 0 | 1.436e+8 | 1.436e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 1/27 | 0 | 4.300e+8 | 2.867e+8 | 0 | 2.867e+8 | 2.867e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `topk` | prefill | 27/48 | 0 | 5.122e+7 | 9.830e+5 | 0 | 1.007e+8 | 3.932e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 27/48 | 0 | 2.501e+4 | 4.800e+2 | 0 | 4.915e+4 | 1.920e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 1.049e+7 | 1.049e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 5.120e+3 | 5.120e+3 | 0.00 | memory | — | — | — |
| `vision_position` | prefill | 1/1 | 0 | 6.636e+5 | 0 | 0 | 2.654e+6 | 1.327e+6 | 0.00 | memory | — | — | — |
| `vision_position` | decode | 1/1 | 0 | 6.636e+5 | 0 | 0 | 2.654e+6 | 1.327e+6 | 0.00 | memory | — | — | — |
| `vision_merge` | prefill | 1/1 | 0 | 0 | 0 | 0 | 1.327e+6 | 1.327e+6 | 0.00 | memory | — | — | — |
| `vision_merge` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.327e+6 | 1.327e+6 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 1/27 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 1/27 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `qwen_qkvz_split` | prefill | 14/36 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `qwen_qkvz_split` | decode | 14/36 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S12 · deepseek-ai/DeepSeek-V3.2

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_mlp` | prefill | 2/58 | 4.185e+13 | 3.892e+9 | 3.892e+9 | 1.308e+12 | 7.785e+9 | 3.892e+9 | 31.72 | memory | — | — | — |
| `fused_moe_mlp` | decode | 2/58 | 2.043e+10 | 1.901e+6 | 1.901e+6 | 4.087e+10 | 3.801e+6 | 1.901e+6 | 0.50 | memory | — | — | — |
| `linear` | prefill | 28/547 | 3.301e+13 | 0 | 0 | 3.224e+10 | 1.281e+10 | 2.202e+10 | 492.21 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 28/547 | 1.612e+10 | 0 | 0 | 3.224e+10 | 6.254e+6 | 1.075e+7 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `dsa_sparse_mla` | prefill | 3/61 | 5.242e+12 | 0 | 0 | 0 | 7.233e+10 | 6.962e+10 | 36.93 | memory | — | — | — |
| `dsa_sparse_mla` | decode | 3/61 | 5.117e+9 | 0 | 0 | 0 | 2.111e+8 | 6.596e+7 | 18.47 | memory | — | — | — |
| `mla_query_compress` | prefill | 3/61 | 1.375e+12 | 0 | 0 | 1.343e+9 | 1.791e+9 | 3.838e+8 | 390.98 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | decode | 3/61 | 6.716e+8 | 0 | 0 | 1.343e+9 | 8.745e+5 | 1.874e+5 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `dsa_indexer` | prefill | 3/61 | 1.048e+12 | 3.277e+10 | 0 | 0 | 1.014e+11 | 5.123e+10 | 6.87 | memory | 计算✓ 字节✓ | 2.149e+9 | 0 |
| `dsa_indexer` | decode | 3/61 | 2.047e+9 | 6.396e+7 | 0 | 0 | 2.579e+8 | 9.746e+7 | 5.76 | memory | 计算✓ 字节✓ | 4.194e+6 | 0 |
| `mla_kv_compress` | prefill | 3/61 | 5.158e+11 | 0 | 0 | 5.037e+8 | 1.791e+9 | 1.439e+8 | 211.51 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | decode | 3/61 | 2.519e+8 | 0 | 0 | 5.037e+8 | 8.745e+5 | 7.027e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | prefill | 3/61 | 0 | 1.842e+10 | 0 | 0 | 2.456e+10 | 1.228e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 3/61 | 0 | 8.995e+6 | 0 | 0 | 1.199e+7 | 5.997e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 2/58 | 0 | 1.362e+10 | 0 | 0 | 1.362e+10 | 1.703e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 2/58 | 0 | 6.652e+6 | 0 | 0 | 6.653e+6 | 8.315e+5 | 0.00 | memory | — | — | — |
| `residual_add` | prefill | 6/122 | 0 | 1.791e+9 | 0 | 0 | 7.164e+9 | 3.582e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 6/122 | 0 | 8.745e+5 | 0 | 0 | 3.498e+6 | 1.749e+6 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 19/306 | 0 | 8.309e+9 | 6.267e+5 | 2.029e+6 | 4.155e+9 | 4.155e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.175e+8 | 0 |
| `rmsnorm` | decode | 19/306 | 0 | 4.057e+6 | 3.060e+2 | 2.029e+6 | 2.029e+6 | 2.029e+6 | 0.00 | memory | 计算✓ 字节✓ | 5.735e+4 | 0 |
| `moe_add` | prefill | 2/58 | 0 | 8.514e+8 | 0 | 0 | 3.406e+9 | 1.703e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 2/58 | 0 | 4.157e+5 | 0 | 0 | 1.663e+6 | 8.315e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 2/58 | 0 | 0 | 0 | 0 | 1.703e+9 | 1.362e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 2/58 | 0 | 0 | 0 | 0 | 8.315e+5 | 6.652e+6 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 3/61 | 0 | 7.130e+8 | 7.130e+8 | 0 | 1.426e+9 | 7.130e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.510e+8 | 0 |
| `swiglu` | decode | 3/61 | 0 | 3.482e+5 | 3.482e+5 | 0 | 6.963e+5 | 3.482e+5 | 0.00 | memory | 计算✓ 字节✓ | 7.373e+4 | 0 |
| `topk` | prefill | 2/58 | 0 | 3.124e+7 | 9.503e+5 | 0 | 6.082e+7 | 3.801e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 2/58 | 0 | 1.525e+4 | 4.640e+2 | 0 | 2.970e+4 | 1.856e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 2.936e+7 | 2.936e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.434e+4 | 1.434e+4 | 0.00 | memory | — | — | — |
| `mla_kv_split` | prefill | 3/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | decode | 3/61 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S13 · deepseek-ai/DeepSeek-V4-Flash-Vision-Exp

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | prefill | 391/514 | 3.399e+13 | 0 | 0 | 3.372e+10 | 1.119e+10 | 1.113e+10 | 606.48 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 391/514 | 1.408e+11 | 0 | 0 | 3.372e+10 | 1.513e+8 | 2.043e+8 | 4.13 | memory | 计算✓ 字节✓ | 0 | 0 |
| `fused_moe_mlp` | prefill | 43/43 | 1.330e+13 | 2.164e+9 | 2.164e+9 | 5.541e+11 | 4.329e+9 | 2.164e+9 | 23.72 | memory | — | — | — |
| `fused_moe_mlp` | decode | 43/43 | 6.493e+9 | 1.057e+6 | 1.057e+6 | 1.299e+10 | 2.114e+6 | 1.057e+6 | 0.50 | memory | — | — | — |
| `mla_kv_compress` | prefill | 41/41 | 5.326e+11 | 0 | 0 | 5.201e+8 | 6.879e+8 | 2.600e+8 | 362.79 | matrix | — | — | — |
| `mla_kv_compress` | decode | 41/41 | 2.600e+8 | 0 | 0 | 5.201e+8 | 3.359e+5 | 1.270e+5 | 0.50 | memory | — | — | — |
| `dsv4_indexer` | prefill | 21/21 | 3.610e+11 | 1.128e+10 | 0 | 0 | 3.491e+10 | 1.737e+10 | 6.90 | memory | — | — | — |
| `dsv4_indexer` | decode | 21/21 | 7.046e+8 | 2.202e+7 | 0 | 0 | 8.877e+7 | 3.342e+7 | 5.77 | memory | — | — | — |
| `dsv4_sparse_mla` | prefill | 21/21 | 1.807e+11 | 0 | 0 | 0 | 3.593e+9 | 3.527e+9 | 25.38 | memory | — | — | — |
| `dsv4_sparse_mla` | decode | 21/21 | 7.046e+8 | 0 | 0 | 0 | 2.892e+7 | 6.881e+6 | 19.68 | memory | — | — | — |
| `dsv4_compressed_attention` | prefill | 20/20 | 4.295e+10 | 0 | 0 | 0 | 2.855e+9 | 2.894e+9 | 7.47 | memory | — | — | — |
| `dsv4_compressed_attention` | decode | 20/20 | 1.311e+6 | 0 | 0 | 0 | 5.407e+6 | 1.495e+6 | 0.19 | memory | — | — | — |
| `mhc_fused_post_pre` | prefill | 43/43 | 3.463e+10 | 2.583e+9 | 1.499e+9 | 6.799e+7 | 6.924e+9 | 3.107e+9 | 3.43 | memory | — | — | — |
| `mhc_fused_post_pre` | decode | 43/43 | 1.691e+7 | 1.261e+6 | 7.321e+5 | 6.799e+7 | 3.381e+6 | 1.517e+6 | 0.23 | memory | — | — | — |
| `mhc_pre` | prefill | 43/43 | 3.463e+10 | 2.222e+9 | 7.779e+8 | 6.799e+7 | 6.203e+9 | 2.385e+9 | 4.00 | memory | — | — | — |
| `mhc_pre` | decode | 43/43 | 1.691e+7 | 1.085e+6 | 3.798e+5 | 6.799e+7 | 3.029e+6 | 1.165e+6 | 0.23 | memory | — | — | — |
| `rope` | prefill | 86/86 | 0 | 1.758e+10 | 0 | 0 | 2.345e+10 | 1.172e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 86/86 | 0 | 8.586e+6 | 0 | 0 | 1.145e+7 | 5.724e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `matmul` | prefill | 2/64 | 4.844e+9 | 0 | 0 | 0 | 1.512e+8 | 1.009e+8 | 19.22 | memory | — | — | — |
| `matmul` | decode | 2/64 | 9.664e+9 | 0 | 0 | 0 | 2.265e+8 | 1.762e+8 | 24.00 | memory | — | — | — |
| `moe_combine` | prefill | 43/43 | 0 | 4.329e+9 | 0 | 0 | 4.330e+9 | 7.214e+8 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 43/43 | 0 | 2.114e+6 | 0 | 0 | 2.114e+6 | 3.523e+5 | 0.00 | memory | — | — | — |
| `residual_add` | prefill | 86/86 | 0 | 7.214e+8 | 0 | 0 | 2.886e+9 | 1.443e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 86/86 | 0 | 3.523e+5 | 0 | 0 | 1.409e+6 | 7.045e+5 | 0.00 | memory | — | — | — |
| `moe_add` | prefill | 43/43 | 0 | 3.607e+8 | 0 | 0 | 1.443e+9 | 7.214e+8 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 43/43 | 0 | 1.761e+5 | 0 | 0 | 7.045e+5 | 3.523e+5 | 0.00 | memory | — | — | — |
| `dsv4_swa_attention` | prefill | 2/2 | 1.082e+9 | 0 | 0 | 0 | 2.729e+8 | 2.769e+8 | 1.97 | memory | — | — | — |
| `dsv4_swa_attention` | decode | 2/2 | 1.678e+7 | 0 | 0 | 0 | 4.588e+5 | 1.987e+5 | 25.52 | memory | — | — | — |
| `mhc_post` | prefill | 1/1 | 8.053e+8 | 8.389e+6 | 0 | 0 | 1.007e+8 | 1.688e+7 | 6.85 | memory | — | — | — |
| `mhc_post` | decode | 1/1 | 3.932e+5 | 4.096e+3 | 0 | 0 | 4.915e+4 | 8.240e+3 | 6.85 | memory | — | — | — |
| `moe_dispatch` | prefill | 43/43 | 0 | 0 | 0 | 0 | 7.214e+8 | 4.329e+9 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 43/43 | 0 | 0 | 0 | 0 | 3.523e+5 | 2.114e+6 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 43/43 | 0 | 3.607e+8 | 3.607e+8 | 0 | 7.214e+8 | 3.607e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `swiglu` | decode | 43/43 | 0 | 1.761e+5 | 1.761e+5 | 0 | 3.523e+5 | 1.761e+5 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rmsnorm` | prefill | 92/151 | 0 | 6.751e+8 | 2.028e+5 | 2.714e+5 | 3.376e+8 | 3.376e+8 | 0.00 | memory | 计算✓ 字节✓ | 6.713e+7 | 0 |
| `rmsnorm` | decode | 92/151 | 0 | 1.009e+8 | 2.466e+4 | 2.714e+5 | 5.047e+7 | 5.047e+7 | 0.00 | memory | 计算✓ 字节✓ | 3.278e+4 | 0 |
| `vision_activation` | prefill | 1/32 | 0 | 6.921e+7 | 6.921e+7 | 0 | 1.384e+8 | 6.921e+7 | 0.00 | memory | — | — | — |
| `vision_activation` | decode | 1/32 | 0 | 6.921e+7 | 6.921e+7 | 0 | 1.384e+8 | 6.921e+7 | 0.00 | memory | — | — | — |
| `softmax` | prefill | 1/32 | 0 | 1.135e+8 | 7.569e+7 | 0 | 7.569e+7 | 7.569e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 1/32 | 0 | 2.265e+8 | 1.510e+8 | 0 | 1.510e+8 | 1.510e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `topk` | prefill | 40/40 | 0 | 2.138e+7 | 4.915e+5 | 0 | 4.194e+7 | 1.966e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 40/40 | 0 | 1.044e+4 | 2.400e+2 | 0 | 2.048e+4 | 9.600e+2 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `mhc_contract` | prefill | 1/1 | 0 | 8.389e+6 | 0 | 0 | 3.355e+7 | 1.678e+7 | 0.00 | memory | — | — | — |
| `mhc_contract` | decode | 1/1 | 0 | 4.096e+3 | 0 | 0 | 1.638e+4 | 8.192e+3 | 0.00 | memory | — | — | — |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 1.678e+7 | 1.678e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 8.192e+3 | 8.192e+3 | 0.00 | memory | — | — | — |
| `vision_position` | prefill | 1/1 | 0 | 3.932e+5 | 0 | 0 | 1.573e+6 | 7.864e+5 | 0.00 | memory | — | — | — |
| `vision_position` | decode | 1/1 | 0 | 3.932e+5 | 0 | 0 | 1.573e+6 | 7.864e+5 | 0.00 | memory | — | — | — |
| `dsv4_hash_route` | prefill | 3/3 | 0 | 0 | 0 | 0 | 7.373e+4 | 7.373e+4 | 0.00 | memory | — | — | — |
| `dsv4_hash_route` | decode | 3/3 | 0 | 0 | 0 | 0 | 3.600e+1 | 3.600e+1 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 1/32 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 1/32 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | prefill | 43/43 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `split` | decode | 43/43 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S14 · zai-org/GLM-4.7

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_mlp` | prefill | 2/89 | 3.440e+13 | 4.480e+9 | 4.480e+9 | 6.719e+11 | 8.959e+9 | 4.480e+9 | 50.20 | memory | — | — | — |
| `fused_moe_mlp` | decode | 2/89 | 1.680e+10 | 2.187e+6 | 2.187e+6 | 3.360e+10 | 4.375e+6 | 2.187e+6 | 0.50 | memory | — | — | — |
| `linear` | prefill | 19/550 | 3.288e+13 | 2.701e+9 | 0 | 3.211e+10 | 1.302e+10 | 1.136e+10 | 582.06 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 19/550 | 1.606e+10 | 1.319e+6 | 0 | 3.211e+10 | 6.356e+6 | 5.548e+6 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `matmul` | prefill | 6/184 | 4.744e+12 | 0 | 0 | 0 | 4.246e+10 | 4.169e+10 | 56.37 | memory | — | — | — |
| `matmul` | decode | 6/184 | 9.261e+9 | 0 | 0 | 0 | 1.618e+9 | 7.461e+7 | 5.47 | memory | — | — | — |
| `softmax` | prefill | 3/92 | 0 | 5.559e+10 | 3.706e+10 | 0 | 3.706e+10 | 3.706e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 3/92 | 0 | 1.085e+8 | 7.235e+7 | 0 | 7.235e+7 | 7.235e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 2/89 | 0 | 1.493e+10 | 0 | 0 | 1.493e+10 | 1.866e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 2/89 | 0 | 7.291e+6 | 0 | 0 | 7.292e+6 | 9.114e+5 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 16/369 | 0 | 1.779e+10 | 7.557e+5 | 1.942e+6 | 8.896e+9 | 8.896e+9 | 0.00 | memory | 计算✓ 字节✓ | 8.390e+7 | 0 |
| `rmsnorm` | decode | 16/369 | 0 | 8.687e+6 | 3.690e+2 | 1.942e+6 | 4.344e+6 | 4.344e+6 | 0.00 | memory | 计算✓ 字节✓ | 4.097e+4 | 0 |
| `residual_add` | prefill | 6/184 | 0 | 1.929e+9 | 0 | 0 | 7.718e+9 | 3.859e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 6/184 | 0 | 9.421e+5 | 0 | 0 | 3.768e+6 | 1.884e+6 | 0.00 | memory | — | — | — |
| `rope` | prefill | 3/92 | 0 | 3.762e+9 | 0 | 0 | 5.016e+9 | 2.508e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 3/92 | 0 | 1.837e+6 | 0 | 0 | 2.449e+6 | 1.225e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_add` | prefill | 2/89 | 0 | 9.332e+8 | 0 | 0 | 3.733e+9 | 1.866e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 2/89 | 0 | 4.557e+5 | 0 | 0 | 1.823e+6 | 9.114e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 2/89 | 0 | 0 | 0 | 0 | 1.866e+9 | 1.493e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 2/89 | 0 | 0 | 0 | 0 | 9.114e+5 | 7.291e+6 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 3/92 | 0 | 7.109e+8 | 7.109e+8 | 0 | 1.422e+9 | 7.109e+8 | 0.00 | memory | 计算✓ 字节✓ | 1.007e+8 | 0 |
| `swiglu` | decode | 3/92 | 0 | 3.471e+5 | 3.471e+5 | 0 | 6.943e+5 | 3.471e+5 | 0.00 | memory | 计算✓ 字节✓ | 4.915e+4 | 0 |
| `topk` | prefill | 2/89 | 0 | 3.044e+7 | 1.458e+6 | 0 | 5.833e+7 | 5.833e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 2/89 | 0 | 1.486e+4 | 7.120e+2 | 0 | 2.848e+4 | 2.848e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 2.097e+7 | 2.097e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.024e+4 | 1.024e+4 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 3/92 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 3/92 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S15 · moonshotai/Kimi-K3

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | prefill | 546/1157 | 1.157e+14 | 1.153e+7 | 0 | 1.134e+11 | 2.828e+10 | 3.952e+10 | 638.32 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 546/1157 | 5.100e+11 | 1.153e+7 | 0 | 1.134e+11 | 4.562e+8 | 6.387e+8 | 4.45 | memory | 计算✓ 字节✓ | 0 | 0 |
| `fused_moe_mlp` | prefill | 46/92 | 9.957e+13 | 1.852e+10 | 1.852e+10 | 5.445e+12 | 3.704e+10 | 1.852e+10 | 18.10 | memory | — | — | — |
| `fused_moe_mlp` | decode | 46/92 | 4.862e+10 | 9.044e+6 | 9.044e+6 | 9.724e+10 | 1.809e+7 | 9.044e+6 | 0.50 | memory | — | — | — |
| `matmul` | prefill | 48/102 | 1.590e+12 | 0 | 0 | 0 | 1.213e+10 | 1.130e+10 | 67.87 | memory | — | — | — |
| `matmul` | decode | 48/102 | 8.999e+10 | 0 | 0 | 0 | 1.067e+9 | 7.839e+8 | 48.61 | memory | — | — | — |
| `gated_delta_attention` | prefill | 24/69 | 6.668e+11 | 3.473e+9 | 6.359e+5 | 3.418e+6 | 7.434e+9 | 7.434e+9 | 44.84 | memory | 计算✓ 字节✓ | 0 | 0 |
| `gated_delta_attention` | decode | 24/69 | 3.256e+8 | 1.085e+8 | 1.987e+4 | 3.418e+6 | 2.323e+8 | 2.323e+8 | 0.70 | memory | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | prefill | 23/24 | 5.412e+11 | 0 | 0 | 5.285e+8 | 7.046e+8 | 1.510e+8 | 390.98 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_query_compress` | decode | 23/24 | 2.642e+8 | 0 | 0 | 5.285e+8 | 3.441e+5 | 7.373e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | prefill | 23/24 | 2.029e+11 | 0 | 0 | 1.982e+8 | 7.046e+8 | 5.662e+7 | 211.51 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `mla_kv_compress` | decode | 23/24 | 9.909e+7 | 0 | 0 | 1.982e+8 | 3.441e+5 | 2.765e+4 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `causal_conv1d` | prefill | 24/69 | 2.084e+10 | 0 | 0 | 2.035e+7 | 1.042e+10 | 1.042e+10 | 1.00 | memory | — | — | — |
| `causal_conv1d` | decode | 24/69 | 1.017e+7 | 0 | 0 | 2.035e+7 | 2.035e+7 | 2.035e+7 | 0.17 | memory | — | — | — |
| `moe_combine` | prefill | 46/92 | 0 | 2.161e+10 | 0 | 0 | 2.162e+10 | 1.351e+9 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 46/92 | 0 | 1.055e+7 | 0 | 0 | 1.055e+7 | 6.595e+5 | 0.00 | memory | — | — | — |
| `rmsnorm` | prefill | 285/569 | 0 | 2.529e+10 | 1.109e+6 | 6.234e+6 | 1.265e+10 | 1.265e+10 | 0.00 | memory | 计算✓ 字节✓ | 1.175e+8 | 0 |
| `rmsnorm` | decode | 285/569 | 0 | 2.429e+8 | 5.683e+4 | 6.234e+6 | 1.215e+8 | 1.215e+8 | 0.00 | memory | 计算✓ 字节✓ | 5.735e+4 | 0 |
| `residual_add` | prefill | 94/186 | 0 | 2.730e+9 | 0 | 0 | 1.092e+10 | 5.461e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 94/186 | 0 | 1.333e+6 | 0 | 0 | 5.333e+6 | 2.666e+6 | 0.00 | memory | — | — | — |
| `softmax` | prefill | 24/51 | 0 | 1.501e+10 | 1.001e+10 | 0 | 1.001e+10 | 1.001e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 24/51 | 0 | 1.048e+9 | 6.984e+8 | 0 | 6.984e+8 | 6.984e+8 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `attention_residual` | prefill | 47/93 | 0 | 5.461e+9 | 2.730e+9 | 0 | 8.191e+9 | 5.461e+9 | 0.00 | memory | — | — | — |
| `attention_residual` | decode | 47/93 | 0 | 2.666e+6 | 1.333e+6 | 0 | 4.000e+6 | 2.666e+6 | 0.00 | memory | — | — | — |
| `rope` | prefill | 23/24 | 0 | 5.436e+9 | 0 | 0 | 7.248e+9 | 3.624e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 23/24 | 0 | 2.654e+6 | 0 | 0 | 3.539e+6 | 1.769e+6 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `gated_rmsnorm` | prefill | 24/69 | 0 | 8.682e+9 | 3.473e+9 | 1.766e+4 | 6.946e+9 | 3.473e+9 | 0.00 | memory | 计算✓ 字节✓ | 1.175e+8 | 0 |
| `gated_rmsnorm` | decode | 24/69 | 0 | 4.239e+6 | 1.696e+6 | 1.766e+4 | 3.391e+6 | 1.696e+6 | 0.00 | memory | 计算✓ 字节✓ | 5.735e+4 | 0 |
| `moe_add` | prefill | 46/92 | 0 | 1.351e+9 | 0 | 0 | 5.402e+9 | 2.701e+9 | 0.00 | memory | — | — | — |
| `moe_add` | decode | 46/92 | 0 | 6.595e+5 | 0 | 0 | 2.638e+6 | 1.319e+6 | 0.00 | memory | — | — | — |
| `swiglu` | prefill | 47/93 | 0 | 2.454e+9 | 2.454e+9 | 0 | 4.907e+9 | 2.454e+9 | 0.00 | memory | 计算✓ 字节✓ | 2.768e+8 | 0 |
| `swiglu` | decode | 47/93 | 0 | 1.198e+6 | 1.198e+6 | 0 | 2.396e+6 | 1.198e+6 | 0.00 | memory | 计算✓ 字节✓ | 1.352e+5 | 0 |
| `moe_dispatch` | prefill | 46/92 | 0 | 0 | 0 | 0 | 1.351e+9 | 2.161e+10 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 46/92 | 0 | 0 | 0 | 0 | 6.595e+5 | 1.055e+7 | 0.00 | memory | — | — | — |
| `mla_output_gate` | prefill | 23/24 | 0 | 6.040e+8 | 1.208e+9 | 0 | 1.208e+9 | 1.208e+9 | 0.00 | memory | — | — | — |
| `mla_output_gate` | decode | 23/24 | 0 | 2.949e+5 | 5.898e+5 | 0 | 5.898e+5 | 5.898e+5 | 0.00 | memory | — | — | — |
| `vision_activation` | prefill | 2/28 | 0 | 2.349e+8 | 2.349e+8 | 0 | 4.698e+8 | 2.349e+8 | 0.00 | memory | — | — | — |
| `vision_activation` | decode | 2/28 | 0 | 2.349e+8 | 2.349e+8 | 0 | 4.698e+8 | 2.349e+8 | 0.00 | memory | — | — | — |
| `topk` | prefill | 46/92 | 0 | 1.716e+8 | 3.015e+6 | 0 | 3.376e+8 | 1.206e+7 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 46/92 | 0 | 8.381e+4 | 1.472e+3 | 0 | 1.649e+5 | 5.888e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 2.936e+7 | 2.936e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 1.434e+4 | 1.434e+4 | 0.00 | memory | — | — | — |
| `vision_position` | prefill | 1/1 | 0 | 1.049e+6 | 0 | 0 | 4.194e+6 | 2.097e+6 | 0.00 | memory | — | — | — |
| `vision_position` | decode | 1/1 | 0 | 1.049e+6 | 0 | 0 | 4.194e+6 | 2.097e+6 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 1/27 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 1/27 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | prefill | 23/24 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `mla_kv_split` | decode | 23/24 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

### S16 · MiniMaxAI/MiniMax-M2.7

| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_mlp` | prefill | 2/62 | 1.438e+13 | 3.121e+9 | 3.121e+9 | 4.494e+11 | 6.241e+9 | 3.121e+9 | 31.35 | memory | — | — | — |
| `fused_moe_mlp` | decode | 2/62 | 7.021e+9 | 1.524e+6 | 1.524e+6 | 1.404e+10 | 3.047e+6 | 1.524e+6 | 0.50 | memory | — | — | — |
| `linear` | prefill | 8/187 | 6.951e+12 | 0 | 0 | 6.788e+9 | 3.133e+9 | 3.745e+9 | 508.61 | matrix | 计算✓ 字节✓ | 0 | 0 |
| `linear` | decode | 8/187 | 3.394e+9 | 0 | 0 | 6.788e+9 | 1.530e+6 | 1.829e+6 | 0.50 | memory | 计算✓ 字节✓ | 0 | 0 |
| `matmul` | prefill | 4/124 | 1.599e+12 | 0 | 0 | 0 | 1.457e+10 | 1.405e+10 | 55.86 | memory | — | — | — |
| `matmul` | decode | 4/124 | 3.121e+9 | 0 | 0 | 0 | 1.065e+9 | 2.514e+7 | 2.86 | memory | — | — | — |
| `softmax` | prefill | 2/62 | 0 | 1.873e+10 | 1.249e+10 | 0 | 1.249e+10 | 1.249e+10 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `softmax` | decode | 2/62 | 0 | 3.657e+7 | 2.438e+7 | 0 | 2.438e+7 | 2.438e+7 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `moe_combine` | prefill | 2/62 | 0 | 6.241e+9 | 0 | 0 | 6.243e+9 | 7.801e+8 | 0.00 | memory | — | — | — |
| `moe_combine` | decode | 2/62 | 0 | 3.047e+6 | 0 | 0 | 3.048e+6 | 3.809e+5 | 0.00 | memory | — | — | — |
| `rope` | prefill | 2/62 | 0 | 2.730e+9 | 0 | 0 | 3.641e+9 | 1.820e+9 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rope` | decode | 2/62 | 0 | 1.333e+6 | 0 | 0 | 1.778e+6 | 8.888e+5 | 0.00 | memory | 计算✓ 字节✓ | 0 | 0 |
| `rmsnorm` | prefill | 12/249 | 0 | 6.786e+9 | 5.100e+5 | 7.997e+5 | 3.393e+9 | 3.393e+9 | 0.00 | memory | 计算✓ 字节✓ | 5.035e+7 | 0 |
| `rmsnorm` | decode | 12/249 | 0 | 3.313e+6 | 2.490e+2 | 7.997e+5 | 1.657e+6 | 1.657e+6 | 0.00 | memory | 计算✓ 字节✓ | 2.458e+4 | 0 |
| `residual_add` | prefill | 4/124 | 0 | 7.801e+8 | 0 | 0 | 3.121e+9 | 1.560e+9 | 0.00 | memory | — | — | — |
| `residual_add` | decode | 4/124 | 0 | 3.809e+5 | 0 | 0 | 1.524e+6 | 7.619e+5 | 0.00 | memory | — | — | — |
| `moe_dispatch` | prefill | 2/62 | 0 | 0 | 0 | 0 | 7.801e+8 | 6.241e+9 | 0.00 | memory | — | — | — |
| `moe_dispatch` | decode | 2/62 | 0 | 0 | 0 | 0 | 3.809e+5 | 3.047e+6 | 0.00 | memory | — | — | — |
| `topk` | prefill | 2/62 | 0 | 3.339e+7 | 1.016e+6 | 0 | 6.501e+7 | 4.063e+6 | 0.00 | memory | 计算✓ 字节✓ | 8.192e+3 | 0 |
| `topk` | decode | 2/62 | 0 | 1.631e+4 | 4.960e+2 | 0 | 3.174e+4 | 1.984e+3 | 0.00 | memory | 计算✓ 字节✓ | 4.000e+0 | 0 |
| `embedding` | prefill | 1/1 | 0 | 0 | 0 | 0 | 1.258e+7 | 1.258e+7 | 0.00 | memory | — | — | — |
| `embedding` | decode | 1/1 | 0 | 0 | 0 | 0 | 6.144e+3 | 6.144e+3 | 0.00 | memory | — | — | — |
| `attention_qkv_split` | prefill | 2/62 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |
| `attention_qkv_split` | decode | 2/62 | 0 | 0 | 0 | 0 | 0 | 0 | — | matrix | — | — | — |

<!-- END GENERATED: operators -->
