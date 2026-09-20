# 架构复用审计（CSA2 bug 同类排查）

> 起因：V4.1 CSA2 被当 V4 SWA（逐字复用 assembleDeepseekV4）。系统排查"一个架构复用另一个 builder/
> attentionKind 却机制不同"的隐藏 bug。方法：跑全 60 catalog 模型，dump 每模型注意力族算子分布，人工核对
> 机制是否与该架构真实注意力一致。

## 全 16 model_type 主干注意力核对（结论：主干均正确）

| model_type | 前端注意力族 | 真实机制 | 判定 |
|---|---|---|---|
| deepseek_v3 (R1/V3.1) | sdpa + mla_compress | 纯 MLA（无 DSA） | ✓ 主干正确 |
| deepseek_v32 (V3.2) | dsa_indexer + dsa_sparse_mla | MLA + DSA | ✓ |
| deepseek_v4 (V4-Flash) | dsv4_swa/compressed/sparse + indexer | CSA+HCA+SWA 混合 | ✓ |
| deepseek_v41 (V4.1) | dsv4_sparse_mla + indexer（**已修**） | CSA2 | ✓（本轮修复） |
| glm4_moe (GLM-4.7) | sdpa | 标准注意力（无 DSA） | ✓ |
| glm5_next / glm_moe_dsa (GLM-5) | dsa(_kpool)_indexer + dsa_sparse_mla + gated_delta + conv1d | DSA + 线性注意力 | ✓ |
| kimi_k2 | mla_compress + sdpa | 纯 MLA | ✓ |
| kimi_k3 | mla + gated_delta + conv1d + attention_residual | KDA 线性 + MLA + AttnRes | ✓ |
| minimax_m2 | sdpa | 标准注意力 | ✓ |
| minimax_m3 | minimax_sparse_indexer + minimax_sparse_attention | 块稀疏 | ✓ |
| qwen3_5(_moe/_text) | sdpa + gated_delta + conv1d + attention_output_gate | GDN 线性 + 全注意力混合 | ✓ |
| qwen4_exp | qsa_indexer + qsa_sparse_attention + gated_delta + conv1d | QSA 稀疏 + 线性 | ✓ |

**主干注意力：16/16 架构机制正确**——未发现第二个 CSA2 级（主干注意力误建）bug。

## 发现并修复：纯 MLA 模型的 MTP 草稿层错用稀疏注意力

- **现象**：R1 / V3.1（纯 MLA，主干 sdpa）的 **MTP 草稿层**错 emit `dsa_indexer`/`dsa_sparse_mla`——与主干矛盾。
- **根因**：`deepseek_mtp.js:ehProjKind` 的 `kvLoraRank ? "qsa" : "mla"` **无条件**把所有 MLA 模型的 MTP 判成稀疏；
  纯 MLA（无 index_topk）也被判稀疏。
- **修复**：判据改看真实 DSA 信号——`hasDsa = dsaIndexTopk>0 || dsaIndexKpool>1`，`hasDsa ? "qsa" : "mla"`。
  纯 MLA(R1/V3.1) → mla（草稿层与主干一致的 sdpa）；DSA 模型(V3.2/GLM-5) 行为不变。
- **验证**：R1/V3.1 MTP 从 dsa → sdpa（对齐主干）；ops-spec-tree golden **仅 R1/V3.1 两个 hash 变**，
  V3.2/GLM-5/其余 58 不变；node --test 410 / verify:models 60/60 / docs:check 全绿。

## 结论

CSA2 bug 属**特例**（V4.1 逐字 re-export V4 + dsv4 路径 ratio 硬编码），主干注意力未见同类问题；
但同源的 MTP attentionKind 选择（ehProjKind）存在一个纯-MLA 误判稀疏的次生 bug，已一并修复。
MTP 是草稿层、成本占比小，但机制一致性已纠正。
