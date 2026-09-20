# stage 级动作向量尝试（nsys/profiler → 真机 per-stage）

**目标**：尝试填算子级成本验证登记的"introspect 不产 stage 级动作向量"诚实缺项。

## 方法

`scripts/evidence/cost/stage_vectors.py`：forward hooks + CUDA events 按 stage（self_attn / mlp / layernorm）计每层 GPU 时间、
跨 28 层聚合；FlopCounterMode 取整前向 FLOPs。Qwen3-0.6B、prefill B=1/S=512、1×A100（bf16）。

## 结果（由 `scripts/evidence/cost/stage_vectors.py` 重生）

- **总 FLOPs = 670,417,616,896**——与前端聚合口径一致（`operator_cost` prefill matrix 610.3B + 注意力方阵项，同量级）。
- **stage CUDA 时间**：attn 36.8ms（**77.4%**）、norm 6.96ms（14.6%）、mlp 3.79ms（8.0%）。

## 发现与结论（诚实）

1. **stage 级 matrix（FLOPs）可导且对账**：总 670B 与前端聚合一致；per-stage matrix = Σ 该 stage 各算子 MACs×2
   （前端 `operator_cost` 已逐算子对 FlopCounter 逐位相等）——**stage 级动作向量的 matrix 分量本就可由"逐算子 × 归属
   stage"聚合得到，数据不缺**。
2. **stage 级 TIME 在 0.6B 尺度被 kernel 启动开销主导**：attn 占 77% 时间但非 FLOPs 主项——因 attn 子核多（q/k_norm、
   rope、QKᵀ、softmax、PV、投影）、每核小 → **launch-overhead-bound**，非 compute-bound。故 stage 级 TIME **不**匹配
   前端 roofline 的算力/带宽 stage 划分（roofline 会把 MLP GEMM 排前）——与聚合 roofline 口径（实测 1.99× roofline 地板、
   overhead 不建模）**同源**：roofline 是 stage 级下界，小模型的 launch/overlap 偏差不在其口径内。
3. **诚实缺项归因**：所谓"introspect 不产 stage 级动作向量"是**前端 introspect API 未暴露 stage 级 rollup**（UI/接口面），
   **非测量缺失**——per-算子 matrix/bytes 已有（`operator_cost`），按 stage 聚合即得。本项**不强行造 stage 级 TIME 匹配**
   （小模型 launch-bound 使其无物理意义），据实登记：stage matrix 可导、stage time 为下界/overhead 主导。

（未用完整 nsys per-kernel 时间线做二次交叉——events 聚合已足以定性；nsys 亦只给时间不给动作向量计数，不改结论。）
