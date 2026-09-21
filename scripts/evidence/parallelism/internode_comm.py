#!/usr/bin/env python3
"""R5 · inter-node（真·多机 81↔41）通信字节公式验证（NCCL over RoCE, torchrun --nnodes=2）。

验证前端 cost/comm.js 三式在**跨节点**成立：
  - ringAllReduceBytes = 2(N-1)/N·D                （all-reduce）
  - expertAllToAllBytes = B·T·topk·H·b             （MoE dispatch all-to-all）
  - ring all-reduce == reduce_scatter + all_gather （各 (N-1)/N·D，和 = 2(N-1)/N·D）
并给出跨节点有效 busbw（仅记录 measured，不写入芯片常量）。

跑法（两节点各一条，node_rank 0 在 master 机）：
  # node A (master, 10.55.87.81), K GPU:
  NCCL_IB_HCA=mlx5_0 NCCL_SOCKET_IFNAME=xgbe4 NCCL_IB_GID_INDEX=3 \
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=K \
      --master_addr=10.55.87.81 --master_port=29600 internode_comm.py
  # node B (10.55.87.41), 同 K：
  NCCL_IB_HCA=mlx5_0 NCCL_SOCKET_IFNAME=xgbe4 NCCL_IB_GID_INDEX=3 \
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=K \
      --master_addr=10.55.87.81 --master_port=29600 internode_comm.py
world = 2*K（K=1→N=2 纯 inter-node；K=2/4→N=4/8 混 intra-NVLink+inter-RoCE）。
"""
import os, time, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank(); world = dist.get_world_size()
lr = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(lr); dev = f"cuda:{lr}"


def bench(fn, it=100, wu=20):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / it


if rank == 0:
    print(f"=== world={world} ===", flush=True)

# all-reduce size sweep：前端 2(N-1)/N·D
for MB in [4, 16, 64]:
    n = MB * 1024 * 1024 // 2
    x = torch.randn(n, dtype=torch.bfloat16, device=dev); D = x.numel() * 2
    dt = bench(lambda: dist.all_reduce(x))
    if rank == 0:
        busbw = D / dt * 2 * (world - 1) / world
        fe = 2 * (world - 1) / world * D
        print(f"[AR] D={D/1e6:.1f}MB busbw={busbw/1e9:.2f}GB/s({busbw*8/1e9:.1f}Gb/s) "
              f"fe2(N-1)/N*D={fe/1e6:.1f}MB busbw*dt={busbw*dt/1e6:.1f}MB ratio={fe/(busbw*dt):.3f}", flush=True)

# MoE dispatch all-to-all：前端 B·T·topk·H·b（V2-Lite: H=2048, topk=6, bf16）
BT, topk, H, b = 4096, 6, 2048, 2
R = BT * topk; rows = R // world
x = torch.randn(rows, H, dtype=torch.bfloat16, device=dev); y = torch.empty_like(x)
dt = bench(lambda: dist.all_to_all_single(y, x))
if rank == 0:
    moved = x.numel() * b * world; fe = BT * topk * H * b
    print(f"[A2A] fe=B*T*topk*H*b={fe/1e6:.2f}MB moved={moved/1e6:.2f}MB "
          f"ratio={moved/fe:.4f} algbw={moved/dt/1e9:.2f}GB/s", flush=True)

# ring all-reduce == RS + AG 分解
full = torch.randn(BT, H, dtype=torch.bfloat16, device=dev); D = full.numel() * 2
chunk = torch.empty(BT // world, H, dtype=torch.bfloat16, device=dev)
dt_rs = bench(lambda: dist.reduce_scatter_tensor(chunk, full))
dt_ag = bench(lambda: dist.all_gather_into_tensor(full, chunk))
if rank == 0:
    f = (world - 1) / world; rs = f * D; ag = f * D; fe = 2 * f * D
    print(f"[RSAG] RS={rs/1e6:.1f} AG={ag/1e6:.1f} RS+AG={(rs+ag)/1e6:.1f}MB "
          f"fe_AR={fe/1e6:.1f}MB ratio={(rs+ag)/fe:.3f} "
          f"rs_busbw={D/dt_rs*f/1e9:.2f} ag_busbw={D/dt_ag*f/1e9:.2f}GB/s", flush=True)

dist.destroy_process_group()
