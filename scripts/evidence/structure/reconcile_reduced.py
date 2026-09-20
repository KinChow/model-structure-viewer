#!/usr/bin/env python3
"""NV-1 结构对账取证 harness：逐模型跑 ``msv verify --graph``，聚合三桶 diff 残留。

对 ``models/catalog.json`` 全部模型：
- 后端真值：``msv verify`` meta 实例化 → ``evidence.modules``；
- 前端图：``--graphs`` 目录下 ``<org>__<model>.graph.json``
  （``scripts/verify-builtin-models.mjs --dump-graphs`` 产）；
- 对账：``evidence.diff`` 三桶 + ``classified`` triage；
  残留 = only_transformers + only_msv + mismatches（均为 triage 后的 unclassified）。

产 per-model diff JSON + 聚合 ``summary.json``；存在构造失败或未分类残留时退出码非零。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DEFAULT = os.environ.get("MSV_EVIDENCE_OUT", str(REPO / "_evidence_out/structure/diffs"))


def graph_path(graphs_dir: Path, model_id: str) -> Path:
    """前端 graph.json 落盘路径：model_id 的 `/` 换成 `__`。"""
    return graphs_dir / (model_id.replace("/", "__") + ".graph.json")


def run_verify(msv: str, model_id: str, graph_file: Path) -> dict:
    """跑 `msv verify --graph`，返回解析后的 JSON；graph 缺失则不带 --graph，
    stdout 非 JSON（CLI 报错）时返回 {ok:false, status:'cli-error', error:...}。"""
    cmd = [msv, "verify", "--model", model_id, "--source", "builtin", "--format", "json"]
    if graph_file.exists():
        cmd += ["--graph", str(graph_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "status": "cli-error", "error": (proc.stderr or "no json").strip()[-2000:]}


def main(argv: list[str] | None = None) -> int:
    """对 catalog 全模型（或 --models 子集）跑对账，落 per-model diff + 聚合 summary，
    存在构造失败或未分类残留时返回非零。"""
    ap = argparse.ArgumentParser(description="NV-1 structure reconciliation harness")
    ap.add_argument("--graphs", default=str(REPO / "_evidence_out/nv1_graphs"), help="前端 graph.json 目录")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--msv", default=str(REPO / ".venv/bin/msv"))
    ap.add_argument("--models", nargs="*", default=None, help="只跑这些 model_id")
    args = ap.parse_args(argv)

    graphs_dir = Path(args.graphs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = REPO / "models/catalog.json"
    try:
        entries = json.loads(catalog_path.read_text())["models"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: 无法读取 {catalog_path}: {exc}", file=sys.stderr)
        return 2
    if args.models:
        wanted = set(args.models)
        entries = [e for e in entries if e["model_id"] in wanted]

    rows = []
    for entry in entries:
        mid = entry["model_id"]
        gfile = graph_path(graphs_dir, mid)
        data = run_verify(args.msv, mid, gfile)
        evidence = data.get("evidence") or {}
        diff = evidence.get("diff") or {}
        only_t = diff.get("only_transformers") or []
        only_m = diff.get("only_msv") or []
        mism = diff.get("mismatches") or []
        residual = len(only_t) + len(only_m) + len(mism)
        row = {
            "model_id": mid,
            "status": data.get("status"),
            "ok": bool(data.get("ok")),
            "graph_present": gfile.exists(),
            "structurally_consistent": (evidence.get("summary") or {}).get("structurally_consistent"),
            "residual_count": residual,
            "only_transformers": only_t,
            "only_msv": only_m,
            "mismatches": mism,
            "classified": diff.get("classified") or {},
            "note": diff.get("note"),
            "error": data.get("error"),
        }
        rows.append(row)
        (out_dir / (mid.replace("/", "__") + ".diff.json")).write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n"
        )
        flag = "OK" if (row["ok"] and residual == 0) else ("RESIDUAL" if row["ok"] else "CONSTRUCT-FAIL")
        print(f"[{flag}] {mid} status={row['status']} residual={residual} "
              f"consistent={row['structurally_consistent']} graph={row['graph_present']}")

    constructed = [r for r in rows if r["ok"]]
    failed = [r for r in rows if not r["ok"]]
    residual_models = [r for r in constructed if r["residual_count"] > 0]
    agg = {
        "total": len(rows),
        "constructed": len(constructed),
        "construct_failed": len(failed),
        "clean_zero_residual": len([r for r in constructed if r["residual_count"] == 0]),
        "with_residual": len(residual_models),
        "construct_failed_models": [r["model_id"] for r in failed],
        "residual_models": [
            {"model_id": r["model_id"], "residual": r["residual_count"]} for r in residual_models
        ],
    }
    (out_dir.parent / "summary.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False) + "\n")
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    return 0 if (not failed and not residual_models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
