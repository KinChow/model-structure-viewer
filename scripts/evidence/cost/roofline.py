#!/usr/bin/env python3
"""NV-3: MSV theoretical roofline vs SGLang measured serving benchmark (Qwen3-0.6B, A100-80GB-SXM, TP=1).

This script applies MSV's own roofline caliber to the aggregate model FLOPs/bytes and
compares against the measured benchmark. The roofline formulas replicate
frontend/src/cost/chips/rates.js + frontend/src/cost/roofline.js exactly:

  matrixPerSecond = peak_flops[dtype] * eta_flops / 2      (rates.js:30)  -> MACs/s
  bytesPerSecond  = memory_bandwidth   * eta_hbm            (rates.js:35)
  per-path time   = quantity / rate                         (roofline.js:80)
  bound           = argmax over {matrix, memory, ...}       (roofline.js:118-120)  "overlap upper bound"

Chip spec source: frontend/src/cost/chips/public.js  (nvidia-a100-80gb-sxm)
Efficiency source: frontend/src/cost/efficiency.js   (DEFAULT_EFFICIENCY)

No model knowledge lives in the roofline; the model FLOPs/bytes are derived from
config.json (Qwen3-0.6B) below and fed in as action quantities, mirroring MSV's
"action vector x rate table" separation (roofline.js is model-agnostic).
"""

# ---- A100-80GB-SXM (MSV public.js: nvidia-a100-80gb-sxm) --------------------
PEAK_BF16 = 312e12          # peak_flops.bf16 (dense, no sparsity)
MEM_BW = 2039e9             # memory_bandwidth (B/s)
ETA_FLOPS = 0.7             # DEFAULT_EFFICIENCY.flops
ETA_HBM = 0.9               # DEFAULT_EFFICIENCY.hbm

MATRIX_MACS_PER_S = PEAK_BF16 * ETA_FLOPS / 2.0   # rates.js:30
BYTES_PER_S = MEM_BW * ETA_HBM                     # rates.js:35

# ---- Qwen3-0.6B config.json -------------------------------------------------
H = 1024            # hidden_size
L = 28              # num_hidden_layers
N_Q = 16            # num_attention_heads
N_KV = 8            # num_key_value_heads
HEAD_DIM = 128      # head_dim  (note: N_Q*HEAD_DIM = 2048 != H)
FFN = 3072          # intermediate_size
VOCAB = 151936      # vocab_size
DBYTES = 2          # bf16

Q_DIM = N_Q * HEAD_DIM     # 2048
KV_DIM = N_KV * HEAD_DIM   # 1024

# ---- Parameter count (tie_word_embeddings=true -> no separate lm_head) -------
per_layer_attn = H * Q_DIM + H * KV_DIM + H * KV_DIM + Q_DIM * H + 2 * HEAD_DIM  # +q_norm,k_norm
per_layer_mlp = 3 * (H * FFN)
per_layer_norm = 2 * H
per_layer = per_layer_attn + per_layer_mlp + per_layer_norm
embed = VOCAB * H
params = per_layer * L + embed + H  # + final norm; lm_head tied
weight_bytes = params * DBYTES
nonembed_params = per_layer * L + H

# ---- Per-token linear MACs (per layer) --------------------------------------
lin_macs_per_tok = (H * Q_DIM + 2 * (H * KV_DIM) + Q_DIM * H) + 3 * (H * FFN)  # attn proj + mlp
per_token_kv_bytes = L * N_KV * HEAD_DIM * 2 * DBYTES  # K+V, all layers  = 112 KiB


def attn_macs_prefill(seq):
    # per layer: QK^T + PV = 2 * N_Q * seq^2 * HEAD_DIM ; over L layers
    return 2 * N_Q * seq * seq * HEAD_DIM * L


def attn_macs_decode(ctx):
    # per layer, 1 query token over ctx keys: 2 * N_Q * ctx * HEAD_DIM ; over L layers
    return 2 * N_Q * ctx * HEAD_DIM * L


def fmt_bytes(b):
    return f"{b/1e9:.3f} GB"


def roofline(macs, bytes_moved, label):
    tm = macs / MATRIX_MACS_PER_S
    tb = bytes_moved / BYTES_PER_S
    bound = "matrix" if tm >= tb else "memory"
    return {
        "label": label, "macs": macs, "bytes": bytes_moved,
        "t_matrix_ms": tm * 1e3, "t_memory_ms": tb * 1e3,
        "t_ms": max(tm, tb) * 1e3, "bound": bound,
    }


print("=" * 74)
print("Qwen3-0.6B structural totals")
print("=" * 74)
print(f"params (total)       : {params:,}")
print(f"params (non-embed)   : {nonembed_params:,}")
print(f"weight bytes (bf16)  : {fmt_bytes(weight_bytes)} ({weight_bytes/2**30:.3f} GiB)")
print(f"per-token KV bytes   : {per_token_kv_bytes:,} B ({per_token_kv_bytes/1024:.0f} KiB)")
print(f"MSV effective rates  : matrix={MATRIX_MACS_PER_S/1e12:.2f} TMACs/s "
      f"(= {2*MATRIX_MACS_PER_S/1e12:.1f} TFLOP/s eff), mem={BYTES_PER_S/1e9:.1f} GB/s")

S_IN, S_OUT, N_REQ = 512, 128, 100
CTX_AVG = S_IN + S_OUT / 2  # 576  (context integrated over decode)

print()
print("=" * 74)
print("A. Single-sequence PREFILL roofline (S=512)")
print("=" * 74)
pf_lin = S_IN * lin_macs_per_tok * L
pf_attn = attn_macs_prefill(S_IN)
pf_macs = pf_lin + pf_attn
pf_bytes = weight_bytes + S_IN * per_token_kv_bytes  # weights once + KV write
r = roofline(pf_macs, pf_bytes, "prefill@512 (1 seq)")
print(f"  MACs={pf_macs/1e9:.2f} G (lin {pf_lin/1e9:.1f} + attn {pf_attn/1e9:.1f}), "
      f"bytes={fmt_bytes(pf_bytes)}")
print(f"  t_matrix={r['t_matrix_ms']:.3f} ms  t_memory={r['t_memory_ms']:.3f} ms  "
      f"-> bound={r['bound']}  t={r['t_ms']:.3f} ms")

print()
print("=" * 74)
print("B. Per-step DECODE roofline (memory-bound), context~576")
print("=" * 74)
for B in (1, 58, 100):
    dec_lin = B * lin_macs_per_tok * L
    dec_attn = B * attn_macs_decode(int(CTX_AVG))
    dec_macs = dec_lin + dec_attn
    dec_bytes = weight_bytes + B * int(CTX_AVG) * per_token_kv_bytes
    r = roofline(dec_macs, dec_bytes, f"decode B={B}")
    tput = B / (r["t_ms"] / 1e3)
    print(f"  B={B:>3}: bytes={fmt_bytes(dec_bytes):>10}  t_matrix={r['t_matrix_ms']:.3f} "
          f"t_memory={r['t_memory_ms']:.3f} -> bound={r['bound']:>6}  "
          f"TPOT~{r['t_ms']:.2f} ms  out~{tput:,.0f} tok/s")

print()
print("=" * 74)
print("C. Aggregate benchmark roofline (100 req x in512/out128)")
print("=" * 74)
agg_pf_macs = N_REQ * pf_macs
agg_pf_t = agg_pf_macs / MATRIX_MACS_PER_S  # prefill compute-bound
# decode: integrate KV over ctx 512..639, batch all N_REQ, weights read each of S_OUT steps
agg_dec_kv = sum(N_REQ * (S_IN + t) * per_token_kv_bytes for t in range(S_OUT))
agg_dec_w = S_OUT * weight_bytes
agg_dec_bytes = agg_dec_kv + agg_dec_w
agg_dec_t = agg_dec_bytes / BYTES_PER_S  # decode memory-bound
floor = agg_pf_t + agg_dec_t
print(f"  prefill total MACs = {agg_pf_macs/1e12:.2f} T -> compute floor {agg_pf_t*1e3:.0f} ms")
print(f"  decode  total bytes= {agg_dec_bytes/1e9:.0f} GB (KV {agg_dec_kv/1e9:.0f} + W {agg_dec_w/1e9:.0f})"
      f" -> memory floor {agg_dec_t*1e3:.0f} ms")
print(f"  MSV roofline floor (prefill+decode) = {floor*1e3:.0f} ms")

print()
print("=" * 74)
print("D. Measured (SGLang benchmark_serving, /tmp/nv3_bench.log)")
print("=" * 74)
meas = {"duration_ms": 1550, "ttft_mean_ms": 584.26, "ttft_med_ms": 571.72,
        "tpot_mean_ms": 7.03, "tpot_med_ms": 7.49, "out_tps": 8276.88,
        "total_tps": 41384.41, "concurrency": 95.50}
for k, v in meas.items():
    print(f"  {k:16}: {v}")
print()
print(f"  duration ratio measured/floor = {meas['duration_ms']/(floor*1e3):.2f}x")
print(f"  TPOT ratio measured/theory(B=100) = {meas['tpot_mean_ms']/4.25:.2f}x  (theory ~4.25 ms)")
