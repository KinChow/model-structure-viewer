#!/usr/bin/env python3
"""glm5_next（GLM-5.3-Flash，`Glm5NextForConditionalGeneration`）减层器。

把 45 层的 GLM-5.3-Flash `text_config` 缩到 8 层，供 SGLang `--load-format dummy` 在 H20(SM90)
跑 DSA 稀疏前向 + hybrid（KDA/MLA/DSA）三 cache 口径对账（`assembleGlm5Next`）。
只改 config（层数/层类型/专家数/去 fp8→bf16、**保全部 per-head 维度**）+ 拷贝 tokenizer/remote-code/
processor 辅助文件；不写权重（dummy 随机权重，验 kernel 路径 + cache 口径、非输出正确性）。

减层布局（与 `evidence/structure/runtime_profiles/sglang_glm5next.md` 一致）：
  8 层：DSA 层 [3,7]（deepseek_sparse_attention）+ KDA 层 [0,1,2,4,5,6]（linear_attention）；
  MoE first_k_dense_replace=3（层 0/1/2 dense、其余 sparse）；n_routed_experts=16。

用法（GLM5_SRC 指向 GLM-5.3-Flash 目录，含 config.json / tokenizer / *.py remote code）：
  GLM5_SRC=/ssd2/models/GLM-5.3-Flash MSV_REDUCED_OUT=/ssd2/models/_reduced/glm5_next_reduced \
    python scripts/evidence/structure/glm5_next_reduce.py
  sglang.launch_server --model-path <OUT> --load-format dummy --tp-size 1 --trust-remote-code

回填 doc：evidence/structure/runtime_profiles/sglang_glm5next.md、evidence/memory/cache_dtype_audit.md。
"""
import json
import os
import shutil

SRC = os.environ.get("GLM5_SRC", "/ssd2/models/GLM-5.3-Flash")
OUT = os.environ.get("MSV_REDUCED_OUT", "_reduced/glm5_next_reduced")
N = int(os.environ.get("GLM5_REDUCED_LAYERS", "8"))
DSA_LAYERS = [int(x) for x in os.environ.get("GLM5_DSA_LAYERS", "3,7").split(",")]
N_EXPERTS = int(os.environ.get("GLM5_REDUCED_EXPERTS", "16"))


def build_reduced(cfg: dict) -> dict:
    tc = cfg.get("text_config", cfg)  # text_config nested (multimodal) or flat
    first_k_dense = int(tc.get("first_k_dense_replace", 3))
    # per-layer attention type: DSA at DSA_LAYERS, KDA (linear_attention) elsewhere
    layer_types = [
        "deepseek_sparse_attention" if i in DSA_LAYERS else "linear_attention"
        for i in range(N)
    ]
    # per-layer MoE type: first_k_dense_replace dense, rest sparse
    mlp_layer_types = ["dense" if i < first_k_dense else "sparse" for i in range(N)]
    # indexer types align to layer count; DSA layers index ('full'), else carry a valid marker
    src_idx = tc.get("indexer_types") or ["full"]
    idx_val = src_idx[0] if src_idx else "full"
    indexer_types = [idx_val for _ in range(N)]
    tc.update({
        "num_hidden_layers": N,
        "layer_types": layer_types,
        "mlp_layer_types": mlp_layer_types,
        "indexer_types": indexer_types,
        "n_routed_experts": N_EXPERTS,
        "first_k_dense_replace": min(first_k_dense, N),
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
    })
    # keep num_experts_per_tok within experts
    if int(tc.get("num_experts_per_tok", 8)) > N_EXPERTS:
        tc["num_experts_per_tok"] = min(8, N_EXPERTS)
    if "text_config" in cfg:
        cfg["text_config"] = tc
    else:
        cfg = tc
    # de-quant: fp8 -> bf16 (dummy load), drop quantization metadata
    cfg.pop("quantization_config", None)
    cfg.pop("torch_dtype", None)
    cfg["torch_dtype"] = "bfloat16"
    return cfg


def main() -> int:
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    cfg = build_reduced(cfg)
    os.makedirs(OUT, exist_ok=True)
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    # copy tokenizer / remote-code / processor aux files (no weights: dummy load)
    copied = []
    for name in sorted(os.listdir(SRC)):
        if name == "config.json":
            continue
        if name.endswith((".safetensors", ".bin", ".pt", ".gguf")) or name == "model.safetensors.index.json":
            continue  # skip weights; --load-format dummy regenerates
        src = os.path.join(SRC, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(OUT, name))
            copied.append(name)
    tc = cfg.get("text_config", cfg)
    print(f"reduced -> {OUT}")
    print(f"  layers={tc['num_hidden_layers']} layer_types={tc['layer_types']}")
    print(f"  mlp_layer_types={tc['mlp_layer_types']} n_routed_experts={tc['n_routed_experts']}")
    print(f"  quantization stripped: {'quantization_config' not in cfg}")
    print(f"  aux files copied: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
