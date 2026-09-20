#!/usr/bin/env python3
"""A3：供 nsys 采集的最小 all-reduce / all-to-all 负载（NVTX 标注）。
在 nsys profile 下运行，抓 NCCL kernel 级时间线与（若可用）NVLink 吞吐。
用法: nsys profile ... python nsys_allreduce.py <world> <mode:ar|a2a>
"""
import os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx

T, H, TOPK = 4096, 1024, 2
ITERS = 50


def worker(rank, world, mode, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    if mode == "ar":
        x = torch.randn(1, T, H, dtype=torch.bfloat16, device=f"cuda:{rank}")
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        nvtx.range_push(f"allreduce_x{ITERS}_N{world}")
        for _ in range(ITERS):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        nvtx.range_pop()
    else:
        rows = (1 * T * TOPK) // world
        x = torch.randn(rows, H, dtype=torch.bfloat16, device=f"cuda:{rank}")
        y = torch.empty_like(x)
        for _ in range(10):
            dist.all_to_all_single(y, x)
        torch.cuda.synchronize()
        nvtx.range_push(f"alltoall_x{ITERS}_ep{world}")
        for _ in range(ITERS):
            dist.all_to_all_single(y, x)
        torch.cuda.synchronize()
        nvtx.range_pop()
    dist.destroy_process_group()


if __name__ == "__main__":
    world = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    mode = sys.argv[2] if len(sys.argv) > 2 else "ar"
    mp.spawn(worker, args=(world, mode, 29631), nprocs=world, join=True)
