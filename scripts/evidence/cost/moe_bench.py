#!/usr/bin/env python3
"""P1-T4 ncu: MoE routed-expert GEMM + MLA compress microbench (DeepSeek-V3 reduced shapes).
For ncu DRAM/tensor capture (bytes channel + routed-expert truth, covers FlopCounter HF-MoE-loop blind spot).
cudaProfilerStart/Stop brackets the measured kernel. Usage: python bench_moe.py <op>
"""
import sys, torch
dt, dev = torch.bfloat16, "cuda"
H, MOE_INT, Q_LORA, KVA = 7168, 2048, 1536, 576
S, TOPK = 128, 4
ACT = S * TOPK
op = sys.argv[1] if len(sys.argv) > 1 else "moe_gate"

def gemm(m, k, n):
    x = torch.randn(m, k, device=dev, dtype=dt)
    w = torch.randn(n, k, device=dev, dtype=dt)
    return lambda: x @ w.T

RUN = {
    "moe_gate": gemm(ACT, H, MOE_INT),
    "moe_down": gemm(ACT, MOE_INT, H),
    "mla_q_a": gemm(S, H, Q_LORA),
    "mla_kv_a": gemm(S, H, KVA),
}[op]

for _ in range(5):
    RUN()
torch.cuda.synchronize()
torch.cuda.profiler.start()
out = RUN()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print(f"[done] op={op} out={tuple(out.shape)}")
