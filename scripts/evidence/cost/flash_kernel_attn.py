"""NV-4：跑一次 flash SDPA（bf16, causal），供 ncu 采 attention kernel 访存。
同时打印理论口径参考值（scores/probs 物化 vs flash 只搬 Q/K/V/O）。"""
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, S, D = 1, 32, int(sys.argv[1]) if len(sys.argv) > 1 else 4096, 128
dt = torch.bfloat16
elem = 2  # bytes per bf16 元素

q = torch.randn(B, H, S, D, device="cuda", dtype=dt)
k = torch.randn_like(q)
v = torch.randn_like(q)

qkvo_bytes = 4 * B * H * S * D * elem          # Q/K/V/O 各一遍
scores_full = B * H * S * S * elem             # 一次物化 scores（bf16）
scores_causal = scores_full // 2               # 因果只算下三角
print(f"[shape] B={B} H={H} S={S} D={D} dtype=bf16")
print(f"[理论] Q/K/V/O 搬运 ≈ {qkvo_bytes/2**20:.1f} MiB")
print(f"[理论] scores 单次物化(full) ≈ {scores_full/2**20:.1f} MiB；"
      f"A2 上限≈4×物化(scores+probs 读写) ≈ {4*scores_full/2**20:.1f} MiB")
print(f"[理论] causal 半三角 scores 物化 ≈ {scores_causal/2**20:.1f} MiB；4× ≈ {4*scores_causal/2**20:.1f} MiB")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    for _ in range(2):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
print("[done] out", tuple(o.shape))
