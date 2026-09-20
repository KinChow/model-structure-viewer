# MSV 验证证据（evidence）

本目录只放**结论文档（.md）**。复现脚本在 `scripts/evidence/`，原始 dump 不入库（可由脚本重生）。

## 分类：按 MSV 的四类预测断言

证据按"验证什么断言"分维度，而非按做的时间。新证据据下表归类：

| 维度 | 验证的断言（MSV 预测） | 真值来源 |
|---|---|---|
| **structure** | 模块树 / 张量名 / shape / 叶参数 / 算子存在性 | checkpoint safetensors index、参考 modeling、SGLang serve 算子 trace |
| **memory** | KV cache 字节 + 线性注意力 state 字节 + 参数字节 | SGLang serve KV 分配、减层前向 buffer |
| **cost** | 每算子 FLOPs / 访存字节 / arithmetic intensity / roofline / kernel 口径 | torch FlopCounter、ncu、bench serving |
| **parallelism** | TP/EP/PP 通信字节与标度、PD 分离 | 多卡 SGLang、NCCL 微基准、nsys |

- method（静态对账 / 运行时 profile / kernel ncu）为次级，进文件名后缀，不单开目录。
- 一份文档覆盖多断言时按主断言归档，正文交叉引用其它维度。
- 每份结论文档顶部标注：对应复现脚本路径（`scripts/evidence/<维度>/…`）+ 关键真值（内联，勿依赖已删 dump）。

## 覆盖矩阵（断言 × 模型族）

状态：✅ 已对账 / 🟡 部分或口径-代表 / ⬜ 待补 / ⛔ 硬件边界（本机不可验）。
structure 全族基线 = [`structure/per_model_reconcile.md`](structure/per_model_reconcile.md)（60 模型逐张量，59 零残差）。
cost / parallelism 多为**口径-代表模型**验证（Qwen3-0.6B、减层 MoE、DeepSeek-V2-Lite），非逐族穷举，故多数族标 🟡/⬜。

| 模型族 | structure | memory | cost | parallelism |
|---|---|---|---|---|
| DeepSeek V3/V3.1/R1 | ✅ | ⬜ | ✅ | 🟡 |
| DeepSeek V3.2 (DSA) | ✅ | ⬜ | 🟡 | 🟡 |
| DeepSeek V4 / V4-Flash / Pro | ✅ | ✅ | ✅ | ⛔ |
| DeepSeek V4.1-Flash (CSA2) | 🟡 | ✅ | ⛔ | ⛔ |
| Kimi K2.x / K3 (KDA) | ✅ | ✅ | ⬜ | ⬜ |
| GLM 4.x / 5.x (MoE/DSA) | ✅ | ⬜ | 🟡 | 🟡 |
| GLM-5-Next | ✅ | ✅ | ⬜ | ⬜ |
| Qwen3 / Qwen3-MoE | ✅ | 🟡 | ✅ | ✅ |
| Qwen3.5/3.6/3.8 (GDN) | ✅ | ⬜ | 🟡 | ⬜ |
| Qwen4-Exp (PLE/QSA) | ✅ | ⬜ | ⬜ | ⬜ |
| MiniMax M2 / M3 | ✅ | ✅ | ⬜ | ⬜ |

- structure 深度证据：`structure/deepseek_v41_*`、`deepseek_v4flash_tensor.md`、`deepseek_v4_proxy.md`、`runtime_profiles/*`。
- memory：`memory/deepseek_v41_csa2_kv_bytes.md`、`glm5next_minimax_m3_cache.md`、`kimi_k3_kda_state.md`、`sglang_width_kv.md`、`deepseek_v4_v41_a100_boundary.md`（⛔ fp8/fp4 前向留 H20）、`cache_dtype_audit.md`（H20 逐字节 dtype 审计：DSA index fp8 / 线性 ssm fp32 两处前端低估/高估 bug）。
- cost：`cost/operator_cost.md`（Qwen3 稠密）、`operator_cost_moe*.md`（DeepSeek MoE）、`roofline_*`、`stage_vectors.md`、`flash_kernel_caliber.md`。
- parallelism：`parallelism/tp*.md`、`ep*.md`、`allreduce*.md`、`nsys.md`、`pd_disaggregation.md`（Qwen3 + 减层 qwen3_moe/DeepSeek-V2-Lite 代表）。
- **总排查**：`frontend_problem_inventory.md` —— 对标 vLLM+SGLang 两框架的前端问题总清单（真 bug / 框架分叉 / 干净 / 缺项 分类，本阶段只摸排不修）。
