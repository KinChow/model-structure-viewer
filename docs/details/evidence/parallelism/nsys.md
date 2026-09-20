# A3 · nsys per-kernel 采集（8×A100, NCCL 集合通信）

> 比 busbw microbench 更强的形式：在 `nsys profile --trace=cuda,nvtx --gpu-metrics-devices=all` 下抓
> NCCL 集合通信的 **kernel 级** 时间线与 NVLink/DRAM GPU 指标。负载 `scripts/evidence/parallelism/nsys_allreduce.py`（N=8，NVTX 标注）。
> 产物 `*.nsys-rep`/`*.sqlite` 不入库（见 `.gitignore`），数字如下。

## kernel 级确认

| 场景 | NCCL kernel | 实例数 | 每 kernel 均时 | busbw microbench dt（同 N/ep=8） |
|---|---|---|---|---|
| all-reduce N=8 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 480（=60 迭代×8 rank） | 177.0 µs | 167.0 µs |
| all-to-all ep=8 | `ncclDevKernel_SendRecv`（all_to_all_single 实现） | 480 | 104.5 µs | 48.0 µs |

- **all-reduce 确为 NCCL Ring 协议 kernel**（`_RING_LL`），bf16、求和——与前端 `ringAllReduceBytes` 的 ring 总线字节
  口径同源；per-kernel 均时 177µs 与 busbw microbench 的 167µs 同量级（nsys+GPU 指标采样有额外开销）。
- **all-to-all 确为 NCCL SendRecv kernel**——`all_to_all_single` 在 NCCL 下展开为点对点 SendRecv，与 A2 的载荷口径一致。

## GPU 指标（NVLink/DRAM，采样为峰值百分比）

`--gpu-metrics-devices=all` 在 A100(GA100) 采到的是 **Throughput %（占峰值百分比）** 采样，非原始字节计数器：

| 指标 | 采样峰值 | 说明 |
|---|---|---|
| NVLink TX Requests User Data % | 42% | 集合通信期间 NVLink 发送用户数据活跃 |
| NVLink RX Requests User Data % | 43% | 同上，接收侧 |
| PCIe RX/TX % | ~0% | 流量走 NVLink 而非 PCIe（单机 intra-node NVLink） |
| DRAM Read/Write % | 3% / 10% | 集合通信非 DRAM 瓶颈 |

- 确认单机 8 卡集合通信**走 NVLink User Data 路径**（PCIe ≈ 0），与 A100 NVLink3 intra-node 拓扑一致。

## 判定与口径边界（诚实）

- **kernel 级形式验证通过**：all-reduce=Ring 集合 kernel、all-to-all=SendRecv kernel，per-kernel 时序与 busbw
  microbench 同量级；流量确走 NVLink。
- **字节精确口径仍以 A1/A2 为准**（前端公式 vs busbw×dt / 载荷构造，比值 1.000）。nsys GPU 指标是**占峰值百分比的采样**，
  精确 per-kernel NVLink 字节积分只能近似（需峰值带宽×%×时间），故此处只作**路径与量级的形式佐证**，不宣称字节逐位。
- 复现：`python scripts/evidence/parallelism/nsys_allreduce.py 8 ar` / `... 8 a2a`，再 `nsys stats --report cuda_gpu_kern_sum`。
