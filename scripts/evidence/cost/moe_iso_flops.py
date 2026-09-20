#!/usr/bin/env python3
"""P1-T4: FlopCounter on ISOLATED MoE expert GEMMs + MLA compress —— 补 FlopCounter 对 HF MoE loop 的盲区。
孤立 GEMM 能被 FlopCounter 正确计数；与前端 active-compute 公式对账，证明前端 routed-expert matrix 正确
（HF eager loop 只是把该计算对计数器隐藏，计算本身=公式）。DeepSeek-V3 减层形状。"""
import torch
from torch.utils.flop_counter import FlopCounterMode
dt = torch.bfloat16
H, MOE_INT, Q_LORA, KVA = 7168, 2048, 1536, 576
S, TOPK = 128, 4
ACT = S * TOPK

def flops(fn):
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        fn()
    return int(sum(sum(o.values()) for o in fc.get_flop_counts().values() if o) // 2)  # /2: 去重父子? no—Global

def flops_global(fn):
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        fn()
    g = fc.get_flop_counts().get("Global", {})
    return int(sum(g.values()))

def gemm(m, k, n):
    x = torch.randn(m, k, device="cuda", dtype=dt); w = torch.randn(n, k, device="cuda", dtype=dt)
    return lambda: x @ w.T

# MoE routed active = gate+up+down over ACT token-expert pairs
g = gemm(ACT, H, MOE_INT); u = gemm(ACT, H, MOE_INT); d = gemm(ACT, MOE_INT, H)
moe = flops_global(lambda: (g(), u(), d()))
qa = flops_global(gemm(S, H, Q_LORA))
kva = flops_global(gemm(S, H, KVA))
print(f"MoE routed active(gate+up+down) torch FLOPs = {moe:,}  前端 fused_moe_mlp MACs×2 = {22548578304*2:,}  match={moe==22548578304*2}")
print(f"MLA q_a torch FLOPs = {qa:,}  前端 mla_query_compress(1层) MACs×2 = {1409286144*2:,}  match={qa==1409286144*2}")
print(f"MLA kv_a torch FLOPs = {kva:,}  前端 mla_kv_compress(1层) MACs×2 = {528482304*2:,}  match={kva==528482304*2}")
