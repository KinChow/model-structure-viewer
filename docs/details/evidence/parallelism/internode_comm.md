# R5 · inter-node 通信费率 + 通信字节公式跨节点验证（真·多机 A100 81↔41，2026-09-21）

> 收口 `validation_status` R5 子项「inter-node roofline 通信费率 / 跨节点 NCCL」。
> 验证前端 `cost/comm.js` 三式在**跨物理节点**成立，并实测跨节点有效 busbw 作为 `interconnect.inter_node.bandwidth` 的参考。
> 微基准 `scripts/evidence/parallelism/internode_comm.py`：`torchrun --nnodes=2`（10.55.87.81 + 10.55.87.41），
> NCCL over RoCE（`NCCL_IB_HCA=mlx5_0 NCCL_SOCKET_IFNAME=xgbe4 NCCL_IB_GID_INDEX=3`），前置 `nv_peer_mem` 两端加载。
> world = 2·K：K=1→N=2 纯 inter-node（单 rail mlx5_0）；K=2/4→N=4/8 混 intra-NVLink + inter-RoCE。

## 结果（前端公式比值，逐 N 跨节点）

all-reduce（`ringAllReduceBytes=2(N-1)/N·D`）：

| N（跨节点） | 4 MB busbw | 16 MB busbw | 64 MB busbw | 2(N-1)/N·D 比值 |
|---|---|---|---|---|
| 2（1+1 卡，纯 inter-node） | 8.51 GB/s (68.1 Gb/s) | 9.22 GB/s (73.8) | 9.95 GB/s (79.6 Gb/s) | 1.000 |
| 4（2+2 卡，混合） | 8.84 | 11.70 | 15.06 GB/s (120.5) | 1.000 |
| 8（4+4 卡，混合） | 8.28 | 14.58 | 17.27 GB/s (138.1) | 1.000 |

all-to-all（`expertAllToAllBytes=B·T·topk·H·b`，BT=4096·topk6·H2048·bf16=100.66 MB）：

| N | 前端 B·T·topk·H·b | 实测搬运 | 比值 | algbw |
|---|---|---|---|---|
| 2 | 100.66 MB | 100.66 MB | 1.0000 | 28.08 GB/s |
| 4 | 100.66 MB | 100.66 MB | 1.0000 | 47.50 GB/s |
| 8 | 100.66 MB | 100.66 MB | 1.0000 | 48.08 GB/s |

reduce_scatter + all_gather（= ring all-reduce 分解，各 (N-1)/N·D）：三个 N 上 `RS+AG == 2(N-1)/N·D` 比值均 1.000。

## 判定

- 前端通信字节三式（all-reduce N 依赖 / all-to-all dispatch / RS+AG 分解）在真·多机跨节点逐 N 成立，比值全 1.000，
  与单机（`allreduce.md` / `alltoall_bench` / `allreduce_rs_ag.md`）一致 —— 字节口径与拓扑（intra vs inter）无关，符合 ring 总线字节定义。
- 跨节点有效 busbw（measured，参考值）：单 rail mlx5_0、N=2 纯 inter-node、64 MB all-reduce ≈ 9.95 GB/s = 79.6 Gb/s
  （与 `ib_write_bw` 裸链路 98 Gb/s 比值 ~0.81，为 NCCL ring 开销后的有效带宽）。N=4/8 因引入 intra-NVLink 段 busbw 抬升到 120/138 Gb/s。
- 诚实边界（按用户决定）：本项仅记录实测 busbw，不写入 `chips/public.js` 或 `chips.local.json` 的 A100
  `interconnect.inter_node.bandwidth`；因此前端 A100 的 `interNodeBytesPerSecond` / inter-node `commTime` 仍为 `null`（unknown）。
  本文验证的是通信字节公式在跨节点正确 + 提供后续若要落 `inter_node.bandwidth` 的实测参考（单 rail ≈ 79.6 Gb/s）。

复现：见 `scripts/evidence/parallelism/internode_comm.py` 头部；两节点 `torchrun --nnodes=2 --node_rank={0,1}` 同参并发启动。
