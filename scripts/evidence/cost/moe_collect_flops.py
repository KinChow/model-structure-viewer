#!/usr/bin/env python3
"""P1-T3：FlopCounterMode 逐 module matmul FLOPs 真值（减层 MoE/MLA，eager attention）。
用 moe_build_reduced 写的同一 reduced config（scripts/evidence/_fixtures/），保证与前端 dump 同结构。
回填 docs/details/evidence/cost/operator_cost_moe.md。
"""
from __future__ import annotations
import json, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from torch.utils.flop_counter import FlopCounterMode

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
FIXTURES = os.path.join(_REPO, "scripts/evidence/_fixtures")
OUT = os.environ.get("MSV_EVIDENCE_OUT", os.path.join(_REPO, "_evidence_out/cost"))
os.makedirs(OUT, exist_ok=True)
S = 128


def counts(fn):
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        fn()
    raw = fc.get_flop_counts()
    return {m: {str(k): int(v) for k, v in ops.items()} for m, ops in raw.items()}


def run(arch):
    cfg = AutoConfig.from_pretrained(os.path.join(FIXTURES, f"{arch}.json"))
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, S), device="cuda")
    prefill = counts(lambda: m(ids, use_cache=False))
    out = {"arch": arch, "seq": S, "prefill": prefill}
    path = os.path.join(OUT, f"torch_flops_{arch}.json")
    json.dump(out, open(path, "w"), indent=1)
    g = prefill.get("Global", {})
    print(f"[{arch}] Global FLOPs: " + ", ".join(f"{k}={v/1e9:.3f}G" for k, v in g.items()))
    return path


if __name__ == "__main__":
    for arch in (sys.argv[1:] or ["deepseek_v3", "deepseek_v4"]):
        try:
            run(arch)
        except Exception as e:
            print(f"[{arch}] FlopCounter 失败: {type(e).__name__}: {e}")
