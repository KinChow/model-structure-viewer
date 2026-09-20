# reduce-scatter / all-gather 字节级（RS+AG == ring all-reduce）

## 背景

前端 `cost/comm.js` 只建 `ringAllReduceBytes = 2(N-1)/N·D`（TP all-reduce）+ `expertAllToAllBytes` 等，
**未单列 reduce-scatter / all-gather**。ring all-reduce 本就 = reduce_scatter + all_gather 两阶段（各
`(N-1)/N·D`），故本项验证前端 all-reduce 口径已涵盖 RS+AG 分解。

## 微基准（`scripts/evidence/parallelism/rs_ag_bench.py`，NCCL，[T=4096,H=1024] bf16，D=8.39MB，8×A100）

`dist.reduce_scatter_tensor` + `dist.all_gather_into_tensor`，各测 busbw×dt 总线字节：

| N | (N-1)/N | RS(MB) | AG(MB) | RS+AG(MB) | 前端 all-reduce 2(N-1)/N·D(MB) | 比值 |
|---|---|---|---|---|---|---|
| 2 | 0.500 | 4.19 | 4.19 | 8.39 | 8.39 | **1.000** |
| 4 | 0.750 | 6.29 | 6.29 | 12.58 | 12.58 | **1.000** |
| 8 | 0.875 | 7.34 | 7.34 | 14.68 | 14.68 | **1.000** |

RS busbw≈69.8、AG busbw≈76.2 GB/s（N=8，走 NVLink，与 all-reduce 同量级）。

## 结论

- **RS + AG 总线字节逐 N 精确 == 前端 `ringAllReduceBytes`（2(N-1)/N·D）**，比值 1.000。
- 即前端 TP 通信的 all-reduce 口径**已完整涵盖 reduce-scatter + all-gather 分解**（ring all-reduce 的两阶段），
  **无需新增 RS/AG 原语**。若未来要按序列并行分栏（RS/AG 单列），本 microbench 即其字节 oracle。
- 与既有 `allreduce.md`（all-reduce N=2/4/8 比值 1.000）+ `nsys.md`（per-kernel）一致收口
  单机 TP 通信字节口径。
