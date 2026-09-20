#!/usr/bin/env python3
"""Part 2 · EP：减层随机 qwen3_moe checkpoint（零下载），供 SGLang --ep-size 加载。
vocab 对齐 Qwen3-0.6B tokenizer 以便 SGLang 加载。save_pretrained 到本地磁盘（不下载权重数据）。
回填 docs/details/evidence/parallelism/ep.md。
用法：MSV_REDUCED_OUT=<减层件输出目录> MSV_PROBE_MODEL=<Qwen3-0.6B 目录> python build_ep_ckpt.py
"""
import os, shutil, glob
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

OUT = os.environ.get("MSV_REDUCED_OUT", "_reduced/qwen3_moe_tiny")
TOK_SRC = os.environ.get("MSV_PROBE_MODEL", "<path-to>/Qwen3-0.6B")

cfg = Qwen3MoeConfig(
    hidden_size=1024, num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=8,
    head_dim=128, intermediate_size=768, moe_intermediate_size=768,
    num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1, norm_topk_prob=True,
    vocab_size=151936, max_position_embeddings=4096, tie_word_embeddings=True,
)
m = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()
os.makedirs(OUT, exist_ok=True)
m.save_pretrained(OUT, safe_serialization=True)
cfg.save_pretrained(OUT)
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]:
    src = os.path.join(TOK_SRC, f)
    if os.path.exists(src):
        shutil.copy(src, OUT)
print(f"saved {OUT}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, "
      f"experts={cfg.num_experts} topk={cfg.num_experts_per_tok} layers={cfg.num_hidden_layers}")
print("files:", sorted(os.path.basename(x) for x in glob.glob(OUT + "/*")))
