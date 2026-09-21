# R5 · 跨节点 PD 多 rail KV 传输带宽（真·多机 A100 81↔41，2026-09-21）

> 收口 `validation_status` R5 的 PD KV 传输带宽（`comm.js pdKvTransferBytes` 的带宽项）。
> prefill(81 GPU2, tp1) → decode(41 GPU0, tp1)，`--disaggregation-transfer-backend mooncake`，
> `--disaggregation-ib-device` 取 rail 数 1/4/8（mlx5_0 / mlx5_0..3 / mlx5_0..7），`MC_TE_METRIC=1` 取 mooncake
> Transfer Engine 自报吞吐。负载：16 线程持续发 ~3000-token prompt（每请求 KV=3000·114,688 B ≈ 344 MB）跨节点搬运。
> 前置 `nv_peer_mem` 两端加载（GPUDirect RDMA）。

## 结果

Qwen3-0.6B（28 层, kv_heads 8, head_dim 128, bf16 → 114,688 B/token）：

| rails | mooncake TE 峰值吞吐（纯 KV 传输） | 备注 |
|---|---|---|
| 1 (mlx5_0) | ~9.5 GB/s (76 Gb/s) | 单 rail，见 `pd_disaggregation.md` RDMA-KV 节 |
| 4 (mlx5_0..3) | ~18.25 GB/s (146 Gb/s) | 相对单 rail ~1.9× |
| 8 (mlx5_0..7) | ~18.3 GB/s (146 Gb/s) | 与 4 rail 持平（饱和） |

单次 3001-token 请求实测搬运 KV = 3000 × 114,688 = 344 MB（输出正确），与 MSV `pdKvTransferBytes` 逐点一致（口径见 `pd_disaggregation.md`）。

## 判定

- 多 rail 对 KV 传输带宽有效：单 decode-tp1 rank 下 mooncake TE 峰值从单 rail ~9.5 GB/s 提升到 ~18.3 GB/s（~1.9×），
  在 4 rail 处饱和（8 rail 无进一步提升）—— 饱和点受单 decode rank 的 QP/ingest 并行度约束，非链路上限。
- app-level KV 吞吐（含 prefill 计算共占用，噪声大、run 间波动）为 9.7~18.7 GB/s，非纯传输度量；**以 mooncake TE 峰值为准**。
- 裸链路上限（`ib_write_bw` 98 Gb/s/卡）× 多 rail 的理论聚合远高于 18.3 GB/s，未达即因单 rank decode 端瓶颈——与建模无关。

## 未闭合（本轮 deferred，方法学已定）

- **PD vs colocated 完整 serving 矩阵（TTFT/TPOT × 输入/输出长度 × 并发）**：`sglang.bench_serving` 对 PD router 需
  `--pd-separated` 专用路径，本轮未跑通完整矩阵（endpoint/warmup 需专门调参）。方法：PD 用 router:8000、colocated 用单机同构
  server，同 `--dataset-name random --random-input-len/--random-output-len --max-concurrency` 扫参，`perf-analysis` skill → xlsx。

复现：见 `pd_disaggregation.md` 启动命令，prefill/decode 加 `--disaggregation-ib-device mlx5_0[,mlx5_1,...] MC_TE_METRIC=1`；
负载脚本发多 3000-token 并发请求，读 prefill 日志 `Transfer Engine Stats ... Throughput` 峰值。
