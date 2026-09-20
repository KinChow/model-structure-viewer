#!/usr/bin/env python3
"""B（后端真值）：deepseek_v4 逐层注意力类型 —— config 映射 + 减层随机实例化确认。
1) 用 transformers 自带的 _COMPRESS_RATIO_TO_LAYER_TYPE 把真实 V4-Flash 的 compress_ratios 映射成 layer_types，计数。
2) 减层随机 from_config（零下载）实例化，逐层核对 self_attn 实际构建的 layer_type == 映射值。
输出 JSON 供 reconcile。回填 docs/details/evidence/structure/deepseek_v4_proxy.md。
用法：[MSV_V4_CONFIG=<config.json>] [MSV_EVIDENCE_OUT=<out 目录>] python v4_proxy_backend.py
"""
import json, os, sys
from collections import Counter
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.deepseek_v4.configuration_deepseek_v4 import _COMPRESS_RATIO_TO_LAYER_TYPE

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CFG = os.environ.get("MSV_V4_CONFIG", os.path.join(_REPO, "models/deepseek-ai/DeepSeek-V4-Flash/config.json"))
raw = json.load(open(CFG))
tc = raw.get("text_config", raw)
n = tc["num_hidden_layers"]
ratios = tc["compress_ratios"][:n]

# 1) config → layer_types（用 transformers 自己的映射常量，非硬编码）
mapped = [_COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in ratios]
full_counts = Counter(mapped)
print("[full] ratio 映射常量:", _COMPRESS_RATIO_TO_LAYER_TYPE)
print(f"[full] num_hidden_layers={n}  layer_types 计数:", dict(full_counts))

# 2) 减层随机实例化，确认模块真按映射构建
small = dict(tc)
small.update(dict(
    num_hidden_layers=6, compress_ratios=[0, 0, 4, 128, 4, 128],
    hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
    num_attention_heads=4, num_key_value_heads=4,
    n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
    first_k_dense_replace=1, vocab_size=1000,
))
small.pop("text_config", None)
small.pop("model_type", None)
small.pop("architectures", None)
cfg = AutoConfig.for_model("deepseek_v4", **small)
built = [cfg.layer_types[i] for i in range(cfg.num_hidden_layers)]
print("[reduced] cfg.layer_types:", built)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)
inst = [getattr(model.model.layers[i].self_attn, "layer_type", "?") for i in range(cfg.num_hidden_layers)]
print("[reduced] 实例化 self_attn.layer_type:", inst)
ok = (built == inst == [_COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in small["compress_ratios"]])
print("[reduced] 映射==配置==实例化:", ok)

_OUT_DIR = os.environ.get("MSV_EVIDENCE_OUT", os.path.join(_REPO, "_evidence_out/structure"))
os.makedirs(_OUT_DIR, exist_ok=True)
json.dump({
    "num_hidden_layers": n,
    "ratio_to_type": {str(k): v for k, v in _COMPRESS_RATIO_TO_LAYER_TYPE.items()},
    "full_layer_type_counts": dict(full_counts),
    "reduced_built": built, "reduced_instantiated": inst, "reduced_ok": ok,
}, open(os.path.join(_OUT_DIR, "v4_proxy_backend.json"), "w"), indent=2)
print("written v4_proxy_backend.json")
