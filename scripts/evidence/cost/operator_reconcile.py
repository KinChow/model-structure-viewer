#!/usr/bin/env python3
"""NV-3 深化 · 逐通道对账：前端算子动作向量 vs FlopCounterMode（matrix）+ ncu（bytes/inst）。

三通道口径：
  matrix : MSV MACs×2 == torch FLOPs（GEMM 精确；attention 因果 vs 全方阵登记）
  bytes  : MSV compulsory READ(weights+actIn) vs ncu dram_read（GEMM 精确）；
           写侧常驻 L2（ncu write≈0）→ MSV total 为 DRAM 保守上界
  vec/sfu: MSV 解析逻辑算子数（atoms.js 单测精确锁定），非硬件 SASS 1:1；
           ncu 证这些算子低算术强度 = memory-bound，与 MSV roofline bound 分类一致
产物：reconcile.json + reconcile.md（回填 docs/details/evidence/cost/operator_cost.md）
用法：先跑 operator_cost.mjs / operator_collect_flops.py / run_ncu.sh 产出中间件到 MSV_EVIDENCE_OUT，再跑本脚本。
"""
from __future__ import annotations
import csv, glob, io, json, os, collections

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BASE = os.environ.get("MSV_EVIDENCE_OUT", os.path.join(_REPO, "_evidence_out/cost"))
FE = json.load(open(os.path.join(BASE, "frontend_ops.json")))
TF = json.load(open(os.path.join(BASE, "torch_flops.json")))


def ncu_agg(op):
    fn = os.path.join(BASE, "ncu_raw", f"{op}.csv")
    lines = open(fn).read().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    agg = collections.defaultdict(float)
    for row in csv.DictReader(io.StringIO("\n".join(lines[hdr:]))):
        try:
            agg[row["Metric Name"]] += float(row["Metric Value"].replace(",", ""))
        except ValueError:
            pass
    return agg


def fe_ops(phase):
    return {o["id"].split(".")[-2] + "." + o["id"].split(".")[-1] if "." in o["id"] else o["id"]: o
            for o in FE[phase]["operators"]}


def per_instance(o):
    m = o["multiplier"] or 1
    b = o["actions"]["bytes"]
    return {
        "matrix": o["actions"]["matrix"] / m,
        "vector": (o["actions"]["vector"] or 0) / m,
        "sfu": (o["actions"]["sfu"] or 0) / m,
        "weights": (b.get("weights") or 0) / m,
        "actIn": (b.get("actIn") or 0) / m,
        "actOut": (b.get("actOut") or 0) / m,
        "kvRead": (b.get("kvRead") or 0) / m,
    }


# ---- MATRIX 通道（聚合） ----
def agg_matrix(phase):
    lin = sum(o["actions"]["matrix"] for o in FE[phase]["operators"]
              if o["operator_id"] == "linear" or o["type"] == "embedding")
    attn = sum(o["actions"]["matrix"] for o in FE[phase]["operators"]
               if o["operator_id"] == "sdpa_attention")
    g = TF[phase]["modules"]["Global"]
    return {
        "linear_msv_macs": lin, "linear_msv_flops": lin * 2, "torch_mm": g.get("aten.mm", 0),
        "linear_exact": lin * 2 == g.get("aten.mm", 0),
        "attn_msv_macs": attn, "attn_msv_flops": attn * 2, "torch_bmm": g.get("aten.bmm", 0),
        "attn_ratio_fe_over_torch": round(attn * 2 / g.get("aten.bmm", 1), 4),
    }


# ---- BYTES 通道（单层实例，GEMM）----
OP_MAP = {  # ncu op -> frontend leaf id 关键字
    "gate_proj": "mlp.gate_proj", "down_proj": "mlp.down_proj",
    "o_proj": "self_attn.o_proj", "q_proj": "self_attn.q_proj",
    "rmsnorm": "input_layernorm.rmsnorm", "swiglu": "mlp.swiglu",
    "rope": "self_attn.rope", "attention": "self_attn.sdpa",
}
fe_by_id = {o["id"]: o for o in FE["prefill"]["operators"]}
fe_by_key = {}
for o in FE["prefill"]["operators"]:
    for k in OP_MAP.values():
        if o["id"].endswith(k):
            fe_by_key[k] = o

rows = []
for op, key in OP_MAP.items():
    o = fe_by_key.get(key)
    if not o:
        continue
    pi = per_instance(o)
    n = ncu_agg(op)
    dr, dw = n.get("dram__bytes_read.sum", 0), n.get("dram__bytes_write.sum", 0)
    msv_read = pi["weights"] + pi["actIn"] + pi["kvRead"]
    msv_total = msv_read + pi["actOut"]
    rows.append({
        "op": op, "kind": "gemm" if o["operator_id"] == "linear" else o["operator_id"],
        "msv_weights": pi["weights"], "msv_actIn": pi["actIn"], "msv_actOut": pi["actOut"],
        "msv_kvRead": pi["kvRead"], "msv_read_compulsory": msv_read, "msv_total_compulsory": msv_total,
        "ncu_dram_read": dr, "ncu_dram_write": dw, "ncu_dram_total": dr + dw,
        "read_ratio_ncu_over_msv": round(dr / msv_read, 4) if msv_read else None,
        "msv_vector": pi["vector"], "msv_sfu": pi["sfu"],
        "ncu_tensor_inst": n.get("sm__inst_executed_pipe_tensor.sum", 0),
        "ncu_L2_bytes": n.get("lts__t_bytes.sum", 0),
        "arith_intensity_flop_per_byte": round((pi["matrix"] * 2 + pi["vector"]) / max(dr, 1), 3),
    })

result = {
    "model": "Qwen/Qwen3-0.6B", "gpu": "A100-SXM4-80GB",
    "matrix_channel": {"prefill": agg_matrix("prefill"), "decode": agg_matrix("decode")},
    "bytes_and_bound_channel": rows,
}
json.dump(result, open(os.path.join(BASE, "reconcile.json"), "w"), indent=2)
print("wrote reconcile.json")
for ph in ("prefill", "decode"):
    m = result["matrix_channel"][ph]
    print(f"[{ph}] linear exact={m['linear_exact']}  attn fe/torch={m['attn_ratio_fe_over_torch']}")
for r in rows:
    print(f"  {r['op']:10} read ncu/msv={r['read_ratio_ncu_over_msv']}  AI={r['arith_intensity_flop_per_byte']} FLOP/B")
