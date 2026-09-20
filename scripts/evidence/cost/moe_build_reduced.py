#!/usr/bin/env python3
"""P1-T1：减层随机初始化 MoE/MLA/DSA 模型（零权重下载）。
写 reduced config 到 scripts/evidence/_fixtures/<arch>.json（前端与真值共用同一 config），并 smoke-test forward。
HF_HUB_OFFLINE=1 强制离线——证明只用本地 config + transformers 原生 modeling，不下载任何权重。
回填 docs/details/evidence/cost/operator_cost_moe.md。
"""
from __future__ import annotations
import json, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoConfig, AutoModelForCausalLM

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, "../../.."))
CFGDIR = os.path.join(REPO, "scripts/evidence/_fixtures")
os.makedirs(CFGDIR, exist_ok=True)

# 每架构：基础 config 路径 + 减层减宽 overrides（set-if-exists，保持一致性）
SPECS = {
    "deepseek_v3": ("models/deepseek-ai/DeepSeek-V3.1/config.json", {
        "num_hidden_layers": 2, "first_k_dense_replace": 1, "n_routed_experts": 8,
        "num_experts_per_tok": 4, "n_group": 1, "topk_group": 1, "num_nextn_predict_layers": 0,
    }),
    "deepseek_v4": ("models/deepseek-ai/DeepSeek-V4-Flash/config.json", {
        "num_hidden_layers": 2, "first_k_dense_replace": 1, "n_routed_experts": 8,
        "num_experts_per_tok": 4, "n_group": 1, "topk_group": 1, "num_nextn_predict_layers": 0,
    }),
}


def build(arch):
    rel, ov = SPECS[arch]
    cfg = AutoConfig.from_pretrained(os.path.join(REPO, rel), trust_remote_code=False)
    for k, v in ov.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    out = os.path.join(CFGDIR, f"{arch}.json")
    cfg.to_json_file(out)
    print(f"[{arch}] reduced config -> {out}")
    try:
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    except Exception as e:
        print(f"[{arch}] eager 构建失败({e})；尝试默认 attn")
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).cuda().eval()
    n = sum(p.numel() for p in m.parameters())
    ids = torch.randint(0, cfg.vocab_size, (1, 32), device="cuda")
    with torch.no_grad():
        m(ids, use_cache=False)
    mods = [name for name, _ in m.named_modules()]
    has = {t: any(t in x for x in mods) for t in ["q_a_proj", "kv_a_proj", "experts.0", "gate", "indexer"]}
    print(f"[{arch}] OK 随机参数 {n/1e9:.2f}B, 前向通过. 子模块存在: {has}")
    del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for arch in (sys.argv[1:] or list(SPECS)):
        try:
            build(arch)
        except Exception as e:
            print(f"[{arch}] 失败: {type(e).__name__}: {e}")
