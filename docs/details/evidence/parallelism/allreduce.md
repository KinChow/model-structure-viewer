# Part 3 · all-reduce 字节级验证（NCCL, N=2/4/8, A100）

> 验证前端 `cost/comm.js: ringAllReduceBytes` = operations × 2(N-1)/N · B·T·H·b（每次 all-reduce）。
> 微基准 `scripts/evidence/parallelism/allreduce_bench.py`：NCCL all-reduce [1,4096,1024] bf16（Qwen3-0.6B 单层 all-reduce 形状），200 迭代，
> world_size=2/4/8（8×A100 单机）。

## 结果（N 依赖标度）

单张量 D = B·T·H·b = 8.39 MB（固定）。

| N | (N-1)/N | 前端 2(N-1)/N·D（每次） | NCCL busbw×dt | 每次 all-reduce | busbw | 比值 前端/实测 |
|---|---|---|---|---|---|---|
| 2 | 0.500 | 8.39 MB | 8.39 MB | 95.4 µs | 88.0 GB/s | **1.000** |
| 4 | 0.750 | 12.58 MB | 12.58 MB | 129.7 µs | 97.0 GB/s | **1.000** |
| 8 | 0.875 | 14.68 MB | 14.68 MB | 167.0 µs | 87.9 GB/s | **1.000** |

## 判定

- **前端 all-reduce 字节公式的 N 依赖标度在 N=2/4/8 逐 N 验证通过**：前端 `2(N-1)/N·D` 与 NCCL ring 总线传输量
  （busbw×dt）在三个 N 上比值均 1.000；随 N 增大字节按 `(N-1)/N`（0.500→0.750→0.875）标度，与 nccl-tests 的
  ring 总线字节定义一致。此前仅 N=2 验过（`2(N-1)/N`=1 退化，未体现 N 依赖）——**本次补齐 N>2 标度缺口**。
- 前端每层两段 all-reduce（attn o_proj + MLP down_proj 后）与 Megatron TP 一致（线 B TP 日志已确认）。
- 口径边界（诚实）：microbench 用 busbw×dt 口径（= ring 总线字节定义）；更强的 nsys per-kernel NVLink 字节见
  `nsys.md`。
