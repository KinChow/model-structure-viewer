#!/usr/bin/env python3
"""NV-2 comm 补齐：reduce-scatter / all-gather 字节级微基准（NCCL, N=2/4/8）。
验证 ring all-reduce = reduce_scatter + all_gather 分解：各阶段总线字节 = (N-1)/N·D，
两者之和 = 2(N-1)/N·D == 前端 ringAllReduceBytes。D 与 allreduce_bench.py 同（[T,H] bf16）。

用法: python rs_ag_bench.py [N1 N2 ...]   # 默认 2 4 8
"""
import os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

T, H = 4096, 1024
ITERS, WARMUP = 200, 30
BASE_PORT = 29711


def worker(rank, world, port, ret):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    full = torch.randn(T, H, dtype=torch.bfloat16, device=f"cuda:{rank}")
    D = full.numel() * full.element_size()          # 全张量字节
    chunk = torch.empty(T // world, H, dtype=torch.bfloat16, device=f"cuda:{rank}")

    def bench(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / ITERS

    dt_rs = bench(lambda: dist.reduce_scatter_tensor(chunk, full))
    dt_ag = bench(lambda: dist.all_gather_into_tensor(full, chunk))
    if rank == 0:
        f = (world - 1) / world
        ret["D"] = D
        ret["rs_busbw"] = (D / dt_rs) * f
        ret["ag_busbw"] = (D / dt_ag) * f
        ret["rs_bytes"] = f * D          # reduce_scatter 总线字节
        ret["ag_bytes"] = f * D          # all_gather 总线字节
        ret["allreduce_frontend"] = 2 * f * D   # 前端 ringAllReduceBytes
        ret["dt_rs"], ret["dt_ag"] = dt_rs, dt_ag
    dist.destroy_process_group()


if __name__ == "__main__":
    worlds = [int(a) for a in sys.argv[1:]] or [2, 4, 8]
    avail = torch.cuda.device_count()
    rows = []
    for i, world in enumerate(worlds):
        if world > avail:
            print(f"[skip] world={world} > 可用 GPU {avail}")
            continue
        mgr = mp.Manager(); ret = mgr.dict()
        mp.spawn(worker, args=(world, BASE_PORT + i, ret), nprocs=world, join=True)
        rs, ag, fe = ret["rs_bytes"], ret["ag_bytes"], ret["allreduce_frontend"]
        rows.append((world, ret["D"], rs, ag, rs + ag, fe))
        print(f"world={world} D={ret['D']/1e6:.2f}MB | RS busbw={ret['rs_busbw']/1e9:.1f} AG busbw={ret['ag_busbw']/1e9:.1f}GB/s "
              f"| RS={rs/1e6:.2f}MB AG={ag/1e6:.2f}MB RS+AG={(rs+ag)/1e6:.2f}MB vs 前端 all-reduce {fe/1e6:.2f}MB "
              f"比值={(rs+ag)/fe:.3f}")
    print("\n=== RS+AG == ring all-reduce (2(N-1)/N·D) 分解验证 ===")
    print("N | (N-1)/N | RS(MB) | AG(MB) | RS+AG(MB) | 前端 all-reduce(MB) | 比值")
    for world, D, rs, ag, tot, fe in rows:
        print(f"{world} | {(world-1)/world:.3f} | {rs/1e6:.2f} | {ag/1e6:.2f} | {tot/1e6:.2f} | {fe/1e6:.2f} | {tot/fe:.3f}")
