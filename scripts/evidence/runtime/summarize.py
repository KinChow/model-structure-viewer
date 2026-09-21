#!/usr/bin/env python3
"""把只读 runtime capture 压缩为逐项证据；绝不以 nvidia-smi 总量反推公式。

用法：python3 summarize.py <run-directory> [<run-directory> ...]
依赖：Python 标准库。输入由 sitecustomize.py/run_smoke.py 产生。
回填：docs/details/evidence/memory/framework_runtime_validation_20260921.md。
"""
import json
from pathlib import Path
import sys


def tensors(value, path=""):
    if isinstance(value, dict):
        if {"shape", "dtype", "storage_ptr", "storage_bytes"} <= value.keys():
            yield path, value
        else:
            for key, child in value.items():
                yield from tensors(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from tensors(child, f"{path}.{index}")


def storage_totals(rows):
    unique = {}
    for _, tensor in rows:
        # 指针只在同一进程的同一快照内比较；不同进程的数值相同不代表共享。
        key = (tensor["device"], tensor["storage_ptr"])
        unique[key] = max(unique.get(key, 0), tensor["storage_bytes"])
    return sum(unique.values()), len(unique)


def summarize(directory):
    directory = Path(directory)
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    summary = {
        "directory": str(directory), "ready": result.get("ready"), "ok": result.get("ok"),
        "error": result.get("error"), "requests": len(result.get("requests", [])),
        "capture_errors": [p.read_text() for p in directory.glob("capture-error-*.txt")],
        "events": [],
    }
    for file in sorted(directory.glob("capture-*.jsonl")):
        for line in file.read_text().splitlines():
            event = json.loads(line)
            row = {key: event[key] for key in (
                "event", "class", "pid", "tp_rank", "tp_size", "is_draft_worker", "model_class",
            ) if key in event}
            if "parameters" in event:
                params = list(tensors(event["parameters"]))
                row.update(
                    named_parameter_bytes=sum(t["bytes"] for _, t in params),
                    unique_parameter_storage_bytes=storage_totals(params)[0],
                    parameter_count=len(params),
                    expert_shapes=[{"name": name, "shape": t["shape"], "dtype": t["dtype"]}
                                   for name, t in params if "expert" in name][:8],
                )
            else:
                allocation = event.get("allocation") or {}
                all_tensors = list(tensors(allocation))
                row["unique_storage_bytes"], row["unique_storages"] = storage_totals(all_tensors)
                row["scalars"] = {k: v for k, v in allocation.items()
                                  if v is None or isinstance(v, (str, int, float, bool))}
                if "kv_cache_config" in allocation:
                    row["kv_cache_config"] = allocation["kv_cache_config"]
                row["tensors"] = [
                    {"path": path, **tensor} for path, tensor in all_tensors
                    if not path.startswith(".index_key_cache.pool.")
                ]
            summary["events"].append(row)
    return summary


if __name__ == "__main__":
    print(json.dumps([summarize(arg) for arg in sys.argv[1:]], indent=2, ensure_ascii=False))
