#!/usr/bin/env python3
"""R1 收口：DeepSeek-V4.1-Flash 运行时模块树 ↔ MSV `deepseek_v41` 图三桶对账，断言 unclassified=0。

后端真值 = 参考栈 meta 构造的运行时模块树（`_fixtures/deepseek_v41_runtime_modules_reduced.json`，
由 `deepseek_v41_runtime_module_tree.py` 在 H20 dsv41 容器产出、canonical 去重、md5 校验传输）。
前端 = `scripts/verify-builtin-models.mjs --dump-graphs` 产出的 `deepseek-ai__DeepSeek-V4.1-Flash.graph.json`。

参考栈（DeepSeek 官方 model.py）与 MSV 的 HF/vLLM 融合图是两套词汇/粒度，故本项在共享契约
`canonical_path_contract.json` 之外**局部**叠加 R1 专用 renaming + known_divergences：
每条都对应真实的 1:1 改名、融合、粒度或量化打包差（见 reason），不改共享契约、不污染其余 59 模型。
numel 口径对齐 MSV（只计逻辑 `weight`，剔除 fp8 `scale`/`bias`）。

用法：
  cd frontend && node ../scripts/verify-builtin-models.mjs --dump-graphs /tmp/msv_graphs
  .venv/bin/python scripts/evidence/structure/deepseek_v41_r1_reconcile.py \
    --graph /tmp/msv_graphs/deepseek-ai__DeepSeek-V4.1-Flash.graph.json

回填 doc：docs/details/evidence/structure/deepseek_v41_runtime_module_tree_h20.md。
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model_structure_viewer.verification.compare_structure import (
    diff_module_evidence,
    load_reconciliation_rules,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "_fixtures" / "deepseek_v41_runtime_modules_reduced.json"

R1_RENAMING = [
    {"backend": "attn", "frontend": "self_attn"},
    {"backend": "ffn", "frontend": "mlp"},
    {"backend": "wq_b", "frontend": "q_proj"},
    {"backend": "gate", "frontend": "router"},
    {"backend": "vision", "frontend": "visual"},
    {"backend": "aligner", "frontend": "projector"},
]
R1_KNOWN_DIVERGENCES = [
    # backend-only: reference modules MSV fuses / represents as ops
    {"pattern": r"(^|\.)wq_a$", "reason": "MSV fuses wq_a+wkv into self_attn.fused_wqa_wkv", "apply_to": ["only_transformers"]},
    {"pattern": r"(^|\.)wkv$", "reason": "MSV fuses wq_a+wkv into fused_wqa_wkv; compressor.wkv is a single mla_kv_compress op", "apply_to": ["only_transformers"]},
    {"pattern": r"self_attn\.compressor\.(norm|wgate)$", "reason": "MSV models the compressor as one mla_kv_compress op", "apply_to": ["only_transformers"]},
    {"pattern": r"self_attn\.indexer\.(wk|k_norm)$", "reason": "MSV folds indexer key proj/norm into the indexer op", "apply_to": ["only_transformers"]},
    {"pattern": r"(^|\.)(attn_norm|ffn_norm)$", "reason": "MSV represents pre-norms as mhc_* fused ops", "apply_to": ["only_transformers"]},
    {"pattern": r"(^|\.)experts$", "reason": "MSV fuses the per-expert ModuleList into a single expert_mlp op", "apply_to": ["only_transformers", "only_msv"]},
    {"pattern": r"experts\.w[123]$", "reason": "MSV fuses per-expert SwiGLU (w1/w2/w3) into expert_mlp", "apply_to": ["only_transformers"]},
    {"pattern": r"shared_experts\.w[123]$", "reason": "reference w1/w2/w3 == MSV shared_experts.gate/up/down_proj", "apply_to": ["only_transformers"]},
    {"pattern": r"^engram_hash$", "reason": "NgramHashState is a non-parametric hash buffer, folded into MSV engram op", "apply_to": ["only_transformers"]},
    {"pattern": r"^visual\.blocks", "reason": "MSV models the ViT at coarser granularity than the reference per-block tree", "apply_to": ["only_transformers"]},
    {"pattern": r"^visual\.(patch_embed|norm)", "reason": "vision granularity differs (MSV visual.* op decomposition)", "apply_to": ["only_transformers"]},
    {"pattern": r"^visual$", "reason": "ViT container vs MSV VisionModel", "apply_to": ["only_transformers", "only_msv"]},
    {"pattern": r"^projector", "reason": "reference Aligner(w1/w2) vs MSV Projector.linear (fused)", "apply_to": ["only_transformers", "only_msv"]},
    {"pattern": r"markov_head\.(embed|head)$", "reason": "reference markov embed/head == MSV markov_w1/markov_w2", "apply_to": ["only_transformers"]},
    {"pattern": r"confidence_head\.proj$", "reason": "MSV models confidence_head as one linear op", "apply_to": ["only_transformers"]},
    {"pattern": r"^embed$", "reason": "reference token embedding == MSV embed_tokens", "apply_to": ["only_transformers"]},
    {"pattern": r"^head$", "reason": "reference ParallelHead == MSV lm_head", "apply_to": ["only_transformers"]},
    # frontend-only: MSV fused ops / renamed leaves with no 1:1 reference module
    {"pattern": r"self_attn\.fused_wqa_wkv$", "reason": "MSV fused q/kv projection (== reference wq_a + wkv)", "apply_to": ["only_msv"]},
    {"pattern": r"self_attn\.attention$", "reason": "MSV SWA attention compute op (dsv4_swa_attention)", "apply_to": ["only_msv"]},
    {"pattern": r"(^|\.)mlp\.expert_mlp$", "reason": "MSV fused expert SwiGLU op (== reference experts.*)", "apply_to": ["only_msv"]},
    {"pattern": r"shared_experts\.(gate|up|down)_proj$", "reason": "MSV shared_experts gate/up/down_proj (== reference w1/w3/w2)", "apply_to": ["only_msv"]},
    {"pattern": r"engram\.engram_gate$", "reason": "MSV engram match-gated write op, non-parametric in reference", "apply_to": ["only_msv"]},
    {"pattern": r"(^|\.)mhc_", "reason": "MSV mHC operator decomposition (pre/contract/post)", "apply_to": ["only_msv"]},
    {"pattern": r"markov_head\.markov_w[12]$", "reason": "MSV markov_w1/w2 (== reference markov_head.embed/head)", "apply_to": ["only_msv"]},
    {"pattern": r"(^|\.)hc_head$", "reason": "MSV DSpark hc_head op", "apply_to": ["only_msv"]},
    {"pattern": r"visual\.(fc1|fc2|qkv_proj|out_proj|input_norm|post_norm|patch_embed)$", "reason": "MSV coarse ViT op decomposition", "apply_to": ["only_msv"]},
    {"pattern": r"projector\.linear$", "reason": "MSV fused vision-text projection (== reference aligner.w1/w2)", "apply_to": ["only_msv"]},
    {"pattern": r"^embed_tokens$", "reason": "MSV embed_tokens == reference embed", "apply_to": ["only_msv"]},
    # class-vocabulary divergences (same module, reference container class vs MSV class)
    {"pattern": r"(^|\.)mlp\.router$", "reason": "reference Gate == MSV router logits", "apply_to": ["mismatches"]},
    {"pattern": r"shared_experts$", "reason": "reference Expert container == MSV MLP", "apply_to": ["mismatches"]},
    {"pattern": r"^projector$", "reason": "reference Aligner == MSV Projector", "apply_to": ["mismatches"]},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="MSV deepseek_v41 graph.json (verify-builtin-models --dump-graphs)")
    ap.add_argument("--evidence", default=str(FIXTURE))
    args = ap.parse_args()
    be = json.load(open(args.evidence))["modules"]
    # align numel to MSV logical-weight convention (drop fp8 scale/bias)
    for m in be:
        ws = m.get("weight_shapes") or {}
        if "weight" in ws:
            m["params"] = int(math.prod(ws["weight"]))
    graph = json.load(open(args.graph))
    base = load_reconciliation_rules()
    rules = dict(base)
    rules["renaming"] = list(base.get("renaming", [])) + R1_RENAMING
    rules["known_divergences"] = list(base.get("known_divergences", [])) + R1_KNOWN_DIVERGENCES
    res = diff_module_evidence(transformers_modules=be, msv_graph=graph, rules=rules)
    unclassified = len(res["only_transformers"]) + len(res["only_msv"]) + len(res["mismatches"])
    print("classified:", res["classified"])
    print("unclassified only_transformers:", res["only_transformers"])
    print("unclassified only_msv:", res["only_msv"])
    print("unclassified mismatches:", res["mismatches"])
    print("UNCLASSIFIED TOTAL:", unclassified)
    if unclassified:
        print("R1 NOT closed: unclassified residual remains")
        return 1
    print("R1 structurally_consistent: three buckets have zero unclassified residual")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
