#!/usr/bin/env python3
"""R1 后端真值：参考栈 meta 构造 dump DeepSeek-V4.1-Flash 运行时完整模块树。

`deepseek_v41` 在 transformers 里不被识别，`msv verify` 的 transformers-meta 路径拿不到
后端 evidence（"第 60 模型"）。本脚本用 DeepSeek 官方参考栈（inference/model.py）在
`meta` 设备上全量构造（不占显存、不触发 GEMM），遍历 named_modules 产出后端 evidence
`{path, class, weight_shapes, params}`，供 `compare_structure.diff_module_evidence`
对 MSV `deepseek_v41` 图做三桶 diff。

用法（在含参考栈依赖的环境，如 H20 dsv41 容器）：
  V41_INFER_DIR=/ssd4/models/inference \
  V41_CKPT_DIR=/ssd4/models/DeepSeek-V4.1-Flash \
    python scripts/evidence/structure/deepseek_v41_runtime_module_tree.py --out /tmp/ev.json

回填 doc：docs/details/evidence/structure/deepseek_v41_runtime_module_tree_h20.md。
注意：参考栈命名（attn/ffn/wq_a）与 MSV 前端 HF 词汇（self_attn/mlp/q_proj）不同源，
零残留需 renaming 表或改用 SGLang 原生模型类作后端真值（见回填 doc 边界）。
"""
import argparse
import dataclasses
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dsv41_module_evidence.json")
    ap.add_argument("--infer", default=os.environ.get("V41_INFER_DIR", "/ssd4/models/inference"))
    ap.add_argument("--ckpt", default=os.environ.get("V41_CKPT_DIR", "/ssd4/models/DeepSeek-V4.1-Flash"))
    args = ap.parse_args()
    sys.path.insert(0, args.infer)
    import torch
    from transformers import AutoTokenizer
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("meta")
    torch.manual_seed(0)
    import model as ref
    cfg = json.load(open(os.path.join(args.infer, "config.json")))
    fields = {f.name for f in dataclasses.fields(ref.ModelArgs)}
    margs = ref.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    try:
        net = ref.Transformer(margs, tok)
    except TypeError:
        net = ref.Transformer(margs)
    mods = []
    for path, module in net.named_modules():
        params = list(module.named_parameters(recurse=False))
        shapes = {name: list(p.shape) for name, p in params}
        numel = 0
        for _, p in params:
            try:
                numel += int(p.numel())
            except Exception:
                pass
        mods.append({"path": path, "class": type(module).__name__, "weight_shapes": shapes, "params": numel})
    out = {
        "model_args": {
            "n_layers": margs.n_layers,
            "n_mtp_layers": getattr(margs, "n_mtp_layers", None),
            "engram_layer_ids": list(getattr(margs, "engram_layer_ids", [])),
            "dspark_target_layer_ids": list(getattr(margs, "dspark_target_layer_ids", [])),
            "vision_n_layers": getattr(margs, "vision_n_layers", None),
        },
        "modules": mods,
    }
    json.dump(out, open(args.out, "w"))
    print("MODULES", len(mods), "-> ", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
