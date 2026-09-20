#!/usr/bin/env python3
"""Part 2b：expert all-to-all 字节级微基准（NCCL, ep=2/4/8）。
验证前端 cost/comm.js: expertAllToAllBytes = operations·B·T·expertsPerToken·H·b。

MoE dispatch 把每个 token 复制到其 topk 个专家所在 rank；combine 反向。
一次 dispatch 在系统内搬运的 token-expert 载荷 = B·T·topk·H·b（与 ep 无关，ep 只改延迟）。
本 bench 用 all_to_all_single 等分重分布 rows=B·T·topk 行×H 的载荷，测实测 busbw，
并逐 ep 核对搬运字节 == 前端 dispatch 字节；dispatch+combine = ×2。

用法: python alltoall_bench.py [ep1 ep2 ...]   # 默认 2 4 8
"""
import os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

B, T, H, TOPK = 1, 4096, 1024, 2   # Qwen3-0.6B 类 MoE 单层形状
ROWS = B * T * TOPK                 # 8192 个 token-expert 载荷行
ITERS, WARMUP = 200, 30
BASE_PORT = 29611


def worker(rank, world, port, ret):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    rows_per_rank = ROWS // world
    x = torch.randn(rows_per_rank, H, dtype=torch.bfloat16, device=f"cuda:{rank}")
    y = torch.empty_like(x)
    for _ in range(WARMUP):
        dist.all_to_all_single(y, x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        dist.all_to_all_single(y, x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / ITERS
    if rank == 0:
        per_rank_send = rows_per_rank * H * x.element_size()
        aggregate = per_rank_send * world               # 系统内搬运的 dispatch 载荷
        busbw = aggregate * (world - 1) / world / dt     # all-to-all 总线口径
        ret["agg"], ret["dt"], ret["busbw"] = aggregate, dt, busbw
    dist.destroy_process_group()


if __name__ == "__main__":
    eps = [int(a) for a in sys.argv[1:]] or [2, 4, 8]
    avail = torch.cuda.device_count()
    fe_dispatch = B * T * TOPK * H * 2               # 前端 dispatch 字节
    rows = []
    for i, ep in enumerate(eps):
        if ep > avail:
            print(f"[skip] ep={ep} > 可用 GPU {avail}")
            continue
        mgr = mp.Manager(); ret = mgr.dict()
        mp.spawn(worker, args=(ep, BASE_PORT + i, ret), nprocs=ep, join=True)
        agg, dt, busbw = ret["agg"], ret["dt"], ret["busbw"]
        rows.append((ep, agg, dt, busbw))
        print(f"ep={ep} 搬运 dispatch 载荷={agg/1e6:.2f}MB dt={dt*1e6:.1f}us busbw={busbw/1e9:.1f}GB/s "
              f"| 前端 dispatch B·T·topk·H·b={fe_dispatch/1e6:.2f}MB 比值={agg/fe_dispatch:.3f}")
    print(f"\n=== all-to-all 字节级汇总 (B={B},T={T},H={H},topk={TOPK}) ===")
    print(f"前端 dispatch = B·T·topk·H·b = {fe_dispatch/1e6:.2f}MB；dispatch+combine = {2*fe_dispatch/1e6:.2f}MB/层")
    print("ep | 搬运载荷(MB) | dt(us) | busbw(GB/s) | 搬运/前端dispatch")
    for ep, agg, dt, busbw in rows:
        print(f"{ep} | {agg/1e6:.2f} | {dt*1e6:.1f} | {busbw/1e9:.1f} | {agg/fe_dispatch:.3f}")
