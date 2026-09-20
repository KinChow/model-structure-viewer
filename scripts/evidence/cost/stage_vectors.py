#!/usr/bin/env python3
"""NV-3 · stage 级动作向量尝试：真机 per-stage CUDA 时间 + FLOPs（Qwen3-0.6B, 1×A100）。
用 forward hooks + cuda events 按 stage（self_attn / mlp / layernorm）计每层耗时并跨层聚合；
torch FlopCounterMode 取整前向 FLOPs。产出 stage 级时间分布，对前端 roofline 的 stage 口径定性对齐。
用法: MSV_PROBE_MODEL=<path-or-hf-id-of-Qwen3-0.6B> CUDA_VISIBLE_DEVICES=0 python dump_stage_vectors.py
"""
import json
import os
import sys
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM

MODEL = os.environ.get("MSV_PROBE_MODEL", "Qwen/Qwen3-0.6B")
B, S = 1, 512


def stage_of(name):
    if "self_attn" in name:
        return "attn"
    if name.endswith(".mlp"):
        return "mlp"
    if "layernorm" in name or name.endswith(".norm"):
        return "norm"
    return None


def main():
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    times = defaultdict(float)
    handles, evs = [], {}

    def pre(name):
        def h(mod, inp):
            e = torch.cuda.Event(enable_timing=True); e.record(); evs[name] = e
        return h

    def post(name, stage):
        def h(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True); e.record()
            evs[name + "#end"] = (stage, evs[name], e)
        return h

    for n, m in model.named_modules():
        st = stage_of(n)
        if st:
            handles.append(m.register_forward_pre_hook(pre(n)))
            handles.append(m.register_forward_hook(post(n, st)))

    ids = torch.randint(0, 1000, (B, S), device="cuda")
    for _ in range(3):
        model(ids)  # warmup
    torch.cuda.synchronize()
    times.clear(); evs.clear()
    model(ids)
    torch.cuda.synchronize()  # 单次全局 sync 后再取 elapsed（事件按流有序，无逐 hook sync 干扰）
    for key, val in evs.items():
        if key.endswith("#end"):
            stage, e0, e1 = val
            times[stage] += e0.elapsed_time(e1)

    total = sum(times.values())
    report = {"model": "Qwen3-0.6B", "batch": B, "seq": S,
              "stage_cuda_ms": {k: round(v, 3) for k, v in times.items()},
              "stage_fraction": {k: round(v / total, 4) for k, v in times.items()}}
    # 整前向 FLOPs（FlopCounterMode）
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc:
            model(ids)
        report["total_flops"] = fc.get_total_flops()
    except Exception as e:
        report["total_flops_error"] = str(e)[:120]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
