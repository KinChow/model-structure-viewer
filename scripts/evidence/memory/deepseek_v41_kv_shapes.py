#!/usr/bin/env python3
"""只读探针：dump DeepSeek-V4.1-Flash 官方参考栈减层小模型的每层 KV cache buffer shape。
回填 docs/details/evidence/memory/deepseek_v41_csa2_kv_bytes.md。

目的：验证 CSA2 跨层 KV 共享真值——仅 kv_source_layers 的层拥有 compress_kv_cache，
Reuse 层压缩 KV 常驻为 0（读 source 层 cache）。index k_cache 仅 index_source_layers 拥有。

用法：
  # V41_INFER_DIR 指向 DeepSeek-V4.1-Flash 官方参考栈的 inference/ 目录（含 model.py）。
  V41_INFER_DIR=<path-to>/DeepSeek-V4.1-Flash/inference \
    ./.venv/bin/python scripts/evidence/memory/deepseek_v41_kv_shapes.py

只构造小模型（随机权重）并读取 named_buffers()，不做前向、不改任何状态。
"""
import json
import os
import sys

INFER_DIR = os.environ.get("V41_INFER_DIR", "<path-to>/DeepSeek-V4.1-Flash/inference")
sys.path.insert(0, INFER_DIR)

import torch  # noqa: E402

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(0)

import model as ref  # noqa: E402  # 有 __main__ 守卫，import 不触发自测


def layer_of(name):
    parts = name.split(".")
    for i, tok in enumerate(parts):
        if tok == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def main():
    args = ref.ModelArgs()
    report = {
        "model_args": {
            "n_layers": args.n_layers,
            "dim": args.dim,
            "n_heads": args.n_heads,
            "head_dim": getattr(args, "head_dim", None),
            "compress_ratios": list(args.compress_ratios),
            "kv_source_layers": list(args.kv_source_layers),
            "index_source_layers": list(args.index_source_layers),
        },
        "buffers": [],
    }
    try:
        net = ref.Transformer(args)
    except Exception as exc:  # ABI/kernel 失败 → 触发减层兜底
        report["build_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    for name, buf in net.named_buffers():
        if "cache" in name.lower():
            report["buffers"].append({
                "name": name,
                "layer": layer_of(name),
                "shape": list(buf.shape),
                "dtype": str(buf.dtype),
                "numel": int(buf.numel()),
            })

    def layers_with(substr):
        return sorted({b["layer"] for b in report["buffers"]
                       if substr in b["name"] and b["layer"] is not None})

    report["layers_owning_compress_kv_cache"] = layers_with("compress_kv_cache")
    report["layers_owning_index_k_cache"] = layers_with("k_cache")
    report["csa2_kv_share_ok"] = (
        report["layers_owning_compress_kv_cache"] == list(args.kv_source_layers)
    )
    report["indexer_source_ok"] = (
        report["layers_owning_index_k_cache"] == list(args.index_source_layers)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
