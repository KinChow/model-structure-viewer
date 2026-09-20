#!/usr/bin/env python3
"""NV-3 深化 · 计算量真值：torch FlopCounterMode 逐 module matmul-family FLOPs（Qwen3-0.6B）。

真值口径（对齐 verification/flop_counter.py）：FlopCounterMode 只数 matmul 家族
（mm/addmm/bmm/_scaled_dot_product_*），FMA=2 → **msv matrix(MACs)×2 == torch FLOPs**。
attn_implementation="eager"：注意力走显式 matmul，FLOP 归属到各 self_attn module，
与前端 sdpa 的 QKᵀ+PV 分解口径对齐（注意 torch 数全 S×S，前端数因果 S(S+1)/2，
比值差异在 reconcile 中按"因果 vs 全方阵"登记）。

产物 torch_flops.json：{phase: {module_fqn: {aten_op: flops}, "Global": {...}}}。
回填 docs/details/evidence/cost/operator_cost.md。
用法：MSV_PROBE_MODEL=<Qwen3-0.6B 目录> [MSV_EVIDENCE_OUT=<out 目录>] python operator_collect_flops.py
"""
from __future__ import annotations
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM
from torch.utils.flop_counter import FlopCounterMode

MODEL = os.environ.get("MSV_PROBE_MODEL", "<path-to>/Qwen3-0.6B")
OUT = os.path.join(os.environ.get("MSV_EVIDENCE_OUT", "_evidence_out/cost"), "torch_flops.json")
S_PREFILL = 512
CTX_DECODE = 576  # 575 cached + 1 stepped


def flop_counts(fn) -> dict:
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        fn()
    # get_flop_counts(): {module_fqn: {aten_packet: flops}}；转成纯 dict[str,int]
    raw = fc.get_flop_counts()
    return {mod: {str(op): int(v) for op, v in ops.items()} for mod, ops in raw.items()}


def main() -> int:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    vocab = model.config.vocab_size

    # prefill：默认 model(ids) 对全部位置算 lm_head（与前端"全 token 投影"口径一致）
    ids = torch.randint(0, vocab, (1, S_PREFILL), device="cuda")
    prefill = flop_counts(lambda: model(ids, use_cache=False))

    # decode：先 prefill 575 建 KV cache，再 step 1 token（query=1, key=576）
    warm = torch.randint(0, vocab, (1, CTX_DECODE - 1), device="cuda")
    with torch.no_grad():
        cache = model(warm, use_cache=True).past_key_values
    step = torch.randint(0, vocab, (1, 1), device="cuda")
    decode = flop_counts(lambda: model(step, past_key_values=cache, use_cache=True))

    out = {
        "model": MODEL,
        "attn_implementation": "eager",
        "prefill": {"seq": S_PREFILL, "modules": prefill},
        "decode": {"ctx": CTX_DECODE, "modules": decode},
    }
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")
    print("prefill Global:", prefill.get("Global"))
    print("decode  Global:", decode.get("Global"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
