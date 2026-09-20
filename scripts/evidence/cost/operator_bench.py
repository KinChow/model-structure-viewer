#!/usr/bin/env python3
"""NV-3 深化 · 访存量/指令真值：单算子微基准（Qwen3-0.6B 单层 prefill S=512 形状）。

供 ncu 采 per-kernel DRAM 字节 + 指令数。warmup 后用 cudaProfilerStart/Stop 圈定
被测 kernel（ncu --profile-from-start off），排除 randn 填充等无关 kernel。

用法：python bench_op.py <op>
  op ∈ {gate_proj, down_proj, o_proj, q_proj, attention, rmsnorm, swiglu, rope}
"""
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dt = torch.bfloat16
dev = "cuda"
H, L = 1024, 512          # hidden, tokens(prefill S)
NH, NKV, HD, FFN = 16, 8, 128, 3072
QD, KVD = NH * HD, NKV * HD  # 2048, 1024
op = sys.argv[1] if len(sys.argv) > 1 else "gate_proj"


def linear(out_dim, in_dim):
    x = torch.randn(L, in_dim, device=dev, dtype=dt)
    w = torch.randn(out_dim, in_dim, device=dev, dtype=dt)
    return lambda: x @ w.T


if op == "gate_proj":
    run = linear(FFN, H)            # [512,1024]x[1024,3072] -> [512,3072]
elif op == "down_proj":
    run = linear(H, FFN)            # [512,3072]x[3072,1024] -> [512,1024]
elif op == "o_proj":
    run = linear(H, QD)             # [512,2048]x[2048,1024] -> [512,1024]
elif op == "q_proj":
    run = linear(QD, H)             # [512,1024]x[1024,2048] -> [512,2048]
elif op == "rmsnorm":
    x = torch.randn(L, H, device=dev, dtype=dt)
    w = torch.randn(H, device=dev, dtype=dt)
    def run():
        v = x.float()
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dt) * w
elif op == "swiglu":
    g = torch.randn(L, FFN, device=dev, dtype=dt)
    u = torch.randn(L, FFN, device=dev, dtype=dt)
    run = lambda: F.silu(g) * u
elif op == "rope":
    x = torch.randn(L, NH + NKV, HD, device=dev, dtype=dt)
    cos = torch.randn(L, HD, device=dev, dtype=dt)
    sin = torch.randn(L, HD, device=dev, dtype=dt)
    def rot(t):
        t1, t2 = t[..., : HD // 2], t[..., HD // 2:]
        return torch.cat((-t2, t1), dim=-1)
    run = lambda: x * cos.unsqueeze(1) + rot(x) * sin.unsqueeze(1)
elif op == "attention":
    q = torch.randn(1, NH, L, HD, device=dev, dtype=dt)
    k = torch.randn(1, NKV, L, HD, device=dev, dtype=dt)
    v = torch.randn(1, NKV, L, HD, device=dev, dtype=dt)
    def run():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
else:
    raise SystemExit(f"unknown op {op}")

for _ in range(5):
    run()
torch.cuda.synchronize()
torch.cuda.profiler.start()
out = run()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print(f"[done] op={op} out={tuple(out.shape)}")
