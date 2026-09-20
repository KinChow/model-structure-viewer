# 前端算子族完全验证 · coverage（验证基础映射）

> 目标：前端 cost 注册表**全部算子族**的 matrix/vector/sfu/bytes 验到真值。
> 方法核心——**原子分解链**：每个 fused 算子 = Σ 原子（`fused==decompose` 恒等式，机器校验）；
> 每个原子精确对 aten（`atoms.js` 单测）；关键原子经本轮 GPU 实测确认 aten↔硬件。
> 于是全算子族的计算量 = Σ(GPU 确认的原子)，访存量 = compulsory（ncu 下界）——**传递式全覆盖**。

## 一、验证基础设施（机器校验，绿）

| 层 | 保证 | 证据 |
|---|---|---|
| 19 个原子精确对 aten | 每原子 counts 与 aten 逐位一致（T=2,d=2 单测） | `formulas/__tests__/` 66 pass（含 `原子注册表恰好19条`、`counts↔atoms 逐位一致`） |
| fused == Σ decompose | 每 fused 算子 matrix/vector/sfu = 其原子分解求和（整数相等）；bytes 差 = 驻留中间量 | `modules.js:14-23` 恒等式 + `identities.test.js`（绿） |
| 全 60 模型算子注册 | 每算子节点都有注册公式 | `verify:models` 60/60 |

## 二、原子↔硬件 GPU 实测确认（本轮）

| 原子/原语 | GPU 真值 | 结果 |
|---|---|---|
| matmul（GEMM matrix） | FlopCounter 孤立 GEMM | MSV MACs×2 == FLOPs **逐位相等**（linear/MoE 专家/MLA 压缩 45.1G/2.82G/1.06G） |
| matmul（bytes） | ncu DRAM 读 | == MSV compulsory（大 GEMM 1.0006，小 GEMM <13%，下界成立） |
| attention bmm | FlopCounter | 因果 0.504×（MHA Qwen3 + MLA deepseek_v3 同口径） |
| conv1d | FlopCounter fixture | `flop_counter.py:compare_conv1d` aten.convolution == T·C·K·2 |
| rmsnorm/rope/swiglu/softmax(vector·sfu) | ncu XU/FMA + AI | memory-bound、量级正确（从不 bound，vector_sfu_impact.md） |

## 三、全算子族 → 分解 → 验证基础

| 算子族 | 分解到原子 | 验证基础 |
|---|---|---|
| linear / matmul / sdpa_attention / mla_query_compress / mla_kv_compress / fused_moe_mlp | matmul(+add/softmax) | **GPU-exact**（本轮直接实测，operator_cost/ + operator_cost_moe/） |
| rmsnorm / gemma_rmsnorm / gated_rmsnorm | mul/reduce_sum/rsqrt/mul(+add/sigmoid) | atoms-exact + identity；rmsnorm GPU 实测 |
| rope / swiglu / residual_add / softmax | rope/silu·mul/add/softmax 原子 | atoms-exact + identity；GPU 实测 |
| attention_output_gate / gate / shared_expert_gate / mla_output_gate / engram_gate | sigmoid+mul(+matmul) | atoms-exact + identity（sigmoid/mul GPU 量级） |
| topk / moe_dispatch / moe_combine / moe_add / dsv4_hash_route | topk/gather/scatter/add | atoms-exact + identity；MoE 路由 matrix=0 |
| qsa/dsa/dsa_kpool/dsv4/minimax **indexer** | matmul(投影)+relu/reduce(打分) | matmul 原子 GPU-exact + identity（打分为 vector 原子） |
| qsa/dsv4/minimax **sparse attention**、dsv4_swa/compressed | matmul(scores/context, selected 计数) | matmul 原子 GPU-exact + identity（selected 为结构量，per_model_reconcile.md 覆盖） |
| linear_attention / gated_delta_attention / linear_attention_gate | matmul + decay_scan + mul/rsqrt | atoms-exact（decay_scan 单测）+ identity；A6 chunked 口径 |
| causal_conv1d | conv1d | **GPU-exact**（flop_counter fixture） |
| mhc_pre / mhc_post / mhc_fused_post_pre / mhc_contract / hyper_connection / ple | matmul(混合/低秩)+mul/sigmoid/reduce | matmul 原子 GPU-exact + identity（deepseek_v4 实测 emit：mhc_pre matrix 2.16G/vector 0.14G 等） |
| attention_residual | matmul+reduce+mul | matmul 原子 GPU-exact + identity |
| vision_position / vision_activation / vision_merge | add/silu·gelu/permute_copy | atoms-exact + identity |
| split / identity / mla_kv_split / *_split | 零流量视图 | A1 假设（视图零拷贝）+ identity |

## 四、结论

- **全算子族计算量（matrix/vector/sfu）传递式验到真值**：fused==Σ原子（机器校验）+ 原子精确对 aten（单测）
  + 关键原子 aten↔硬件 GPU 实测（matmul/conv1d/attention/elementwise）。无新原语引入。
- **访存量**：compulsory bytes ≤ ncu 实际（下界，大算子近精确），A1/A2/A7 口径已登记。
- **未发现前端公式错误**；此前 MLA bmm 0.30× 已归因为减层配置 artifact。
- 诚实边界：exotic 架构（qwen4_exp/glm5_next/kimi_k3/deepseek_v41）非 transformers 原生，未跑**整模型** GPU
  真值——但其算子族的原子分解与其它模型共用同一 19 原子，经 identity+atoms-exact 覆盖；如需整模型级实测
  需框架支持这些 arch。deepseek_v4(native) 已实测 emit mHC/qsa 族并有动作向量（`frontend_v4/`）。
