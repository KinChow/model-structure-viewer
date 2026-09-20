#!/usr/bin/env python3
"""dump 参考栈减层小模型的完整模块树（注意力之外）。回填 docs/details/evidence/structure/deepseek_v41_module_tree.md。
只构造小模型（随机权重）+ 读 named_modules，不做前向、不触发 GEMM。
用法:
  # V41_INFER_DIR 指向 DeepSeek-V4.1-Flash 官方参考栈的 inference/ 目录（含 model.py）。
  V41_INFER_DIR=<path-to>/DeepSeek-V4.1-Flash/inference \
    ./.venv/bin/python scripts/evidence/structure/deepseek_v41_module_tree.py
"""
import collections
import json
import os
import sys

INFER_DIR = os.environ.get("V41_INFER_DIR", "<path-to>/DeepSeek-V4.1-Flash/inference")
sys.path.insert(0, INFER_DIR)

import torch  # noqa: E402

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(0)

import model as ref  # noqa: E402


def main():
    args = ref.ModelArgs()
    net = ref.Transformer(args)
    report = {"model_args": {
        "n_layers": args.n_layers,
        "engram_layer_ids": list(getattr(args, "engram_layer_ids", [])),
        "dspark_target_layer_ids": list(getattr(args, "dspark_target_layer_ids", [])),
        "n_routed_experts": args.n_routed_experts,
        "n_shared_experts": getattr(args, "n_shared_experts", None),
        "vision_enabled": getattr(args, "vision_enabled", None),
    }}
    # 模块类名直方图
    cls_hist = collections.Counter(type(m).__name__ for _, m in net.named_modules())
    report["module_class_histogram"] = dict(cls_hist.most_common())
    # 顶层 children
    report["top_level_children"] = [f"{n}:{type(m).__name__}" for n, m in net.named_children()]
    # 每 backbone 层的直接子模块名集合（组件覆盖）
    layer_children = {}
    for i in range(args.n_layers):
        try:
            layer = net.layers[i]
            layer_children[i] = sorted(n for n, _ in layer.named_children())
        except Exception:
            pass
    report["backbone_layer_children"] = layer_children
    # MTP / DSpark 子模块（markov/confidence/main_proj 等）
    mtp_children = {}
    mtp = getattr(net, "mtp", None)
    if mtp is not None:
        for i, blk in enumerate(mtp):
            mtp_children[i] = sorted(n for n, _ in blk.named_children())
    report["mtp_children"] = mtp_children
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
