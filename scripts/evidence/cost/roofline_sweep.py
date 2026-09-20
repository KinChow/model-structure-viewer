#!/usr/bin/env python3
"""NV-3 深化 · roofline 随数据大小的拟合：算子尺寸扫描，实测 kernel 时延 vs MSV 五路地板。

时延用 CUDA events（多次取均值，不经 ncu 以免 replay 扰动）；MSV 地板 = max(matrix/费率, mem/带宽)，
含 η（flops0.7/hbm0.9），是"效率地板下界"——实测应 ≥ 地板，比值越接近 1 拟合越紧。
考察：① 比值随尺寸变化（小尺寸受固定开销/低占用抬高）；② bound 从 memory→matrix 的脊点翻转是否与实测一致。
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dt = torch.bfloat16
MAT = 312e12 * 0.7 / 2      # 109.2e12 MACs/s
BW = 2039e9 * 0.7          # 1427.3e9 B/s（η_hbm NV-3 校准 0.9→0.7）
RIDGE = (MAT * 2) / BW      # 脊点算术强度 FLOP/byte


def timeit(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3  # seconds


def row(name, meas_s, macs, byt):
    t_mat = macs / MAT
    t_mem = byt / BW
    msv = max(t_mat, t_mem)
    bound = "matrix" if t_mat >= t_mem else "memory"
    ai = 2 * macs / byt if byt else 0
    ach_tflops = 2 * macs / meas_s / 1e12 if meas_s else 0
    ach_gbs = byt / meas_s / 1e9 if meas_s else 0
    print(f"  {name:>9} meas={meas_s*1e6:9.2f}us MSV={msv*1e6:9.2f}us ratio={meas_s/msv:6.2f} "
          f"bound={bound:6} AI={ai:8.1f} ach={ach_tflops:6.1f}TF/{ach_gbs:6.0f}GBs")


print(f"A100 脊点 AI = {RIDGE:.1f} FLOP/byte（>脊点 compute-bound，<脊点 memory-bound）\n")

print("=== GEMM [M,1024]x[1024,3072]（gate_proj 形状，扫 M=tokens）===")
K, N = 1024, 3072
w = torch.randn(N, K, device="cuda", dtype=dt)
for M in [1, 4, 16, 64, 256, 1024, 4096, 16384]:
    x = torch.randn(M, K, device="cuda", dtype=dt)
    t = timeit(lambda: x @ w.T)
    row(f"M={M}", t, M * N * K, (N * K + M * K + M * N) * 2)

print("\n=== FLASH 注意力（causal, H=16/8 GQA, D=128，扫 S）===")
for S in [128, 512, 2048, 8192, 16384]:
    q = torch.randn(1, 16, S, 128, device="cuda", dtype=dt)
    k = torch.randn(1, 8, S, 128, device="cuda", dtype=dt)
    v = torch.randn(1, 8, S, 128, device="cuda", dtype=dt)
    def run():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    t = timeit(run, iters=50)
    pairs = S * (S + 1) // 2                       # 因果对数
    macs = 2 * 16 * pairs * 128                    # QKᵀ + PV
    byt = (16 * S * 128 + 2 * 8 * S * 128 + 16 * S * 128) * 2  # Q+K+V+O（flash：scores 不落 HBM）
    row(f"S={S}", t, macs, byt)

print("\n=== SwiGLU [M,3072]（memory-bound，扫 M）===")
for M in [64, 256, 1024, 4096, 16384]:
    g = torch.randn(M, 3072, device="cuda", dtype=dt)
    u = torch.randn(M, 3072, device="cuda", dtype=dt)
    t = timeit(lambda: F.silu(g) * u)
    row(f"M={M}", t, 0, (2 * M * 3072 + M * 3072) * 2)
