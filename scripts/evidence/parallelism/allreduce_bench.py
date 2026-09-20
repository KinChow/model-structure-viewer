#!/usr/bin/env python3
"""Part 3：all-reduce 字节级微基准（NCCL, N=2/4/8）。
验证前端 ringAllReduceBytes = 2(N-1)/N·D（NCCL ring 总线字节定义）的 **N 依赖标度**。
每 rank all-reduce 一个 [1,T,H] bf16 张量；算 algbw、busbw，打印 D 与前端公式字节。

用法: python allreduce_bench.py [N1 N2 ...]   # 默认 2 4 8
"""
import os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

T, H = 4096, 1024   # Qwen3-0.6B 单层 all-reduce 形状（B=1）
ITERS, WARMUP = 200, 30
BASE_PORT = 29591


def worker(rank, world, port, ret):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    x = torch.randn(1, T, H, dtype=torch.bfloat16, device=f"cuda:{rank}")
    D = x.numel() * x.element_size()          # 单张量字节
    for _ in range(WARMUP):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / ITERS
    if rank == 0:
        algbw = D / dt
        busbw = algbw * 2 * (world - 1) / world
        frontend = 2 * (world - 1) / world * D    # 前端每次 all-reduce 字节
        ret["D"], ret["dt"], ret["algbw"], ret["busbw"], ret["frontend"] = D, dt, algbw, busbw, frontend
    dist.destroy_process_group()


if __name__ == "__main__":
    worlds = [int(a) for a in sys.argv[1:]] or [2, 4, 8]
    avail = torch.cuda.device_count()
    rows = []
    for i, world in enumerate(worlds):
        if world > avail:
            print(f"[skip] world={world} > 可用 GPU {avail}")
            continue
        mgr = mp.Manager()
        ret = mgr.dict()
        mp.spawn(worker, args=(world, BASE_PORT + i, ret), nprocs=world, join=True)
        D, dt, algbw, busbw, fe = ret["D"], ret["dt"], ret["algbw"], ret["busbw"], ret["frontend"]
        ratio = fe / (busbw * dt)
        rows.append((world, D, dt, algbw, busbw, fe, ratio))
        print(f"world={world} D={D/1e6:.2f}MB dt={dt*1e6:.1f}us algbw={algbw/1e9:.1f} busbw={busbw/1e9:.1f}GB/s "
              f"| 前端 2(N-1)/N·D={fe/1e6:.2f}MB busbw×dt={busbw*dt/1e6:.2f}MB 比值={ratio:.3f}")
    print("\n=== N 依赖标度汇总 (前端 ringAllReduceBytes vs NCCL ring busbw×dt) ===")
    print("N | (N-1)/N | 前端每次(MB) | busbw×dt(MB) | 比值")
    for world, D, dt, algbw, busbw, fe, ratio in rows:
        print(f"{world} | {(world-1)/world:.3f} | {fe/1e6:.2f} | {busbw*dt/1e6:.2f} | {ratio:.3f}")
