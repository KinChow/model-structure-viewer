"""Collect source_ref for live nn.Module instances.

照抄 modelmap ``src/modelmap/annotate.py:116-134``：
``inspect.getsourcefile`` + ``getsourcelines``[1] + 包根前缀匹配 + 版本锚定 blob。

§5.4：inspect 失败 / 合成节点 → None；类不在已知包根下 → 只给 file:line，不给 url。
禁止编造未经校验的 GitHub 链接。
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

# (abs package dir, github repo, blob path prefix, version tag)
_ROOTS: list[tuple[str, str, str, str]] = []


def reset_package_roots() -> None:
    _ROOTS.clear()


def _package_roots() -> list[tuple[str, str, str, str]]:
    if _ROOTS:
        return _ROOTS
    try:
        import transformers

        _ROOTS.append((
            os.path.dirname(transformers.__file__),
            "huggingface/transformers",
            "src/transformers/",
            f"v{transformers.__version__}",
        ))
    except Exception:
        pass
    try:
        import torch

        ver = torch.__version__.split("+")[0]
        _ROOTS.append((
            os.path.dirname(torch.__file__),
            "pytorch/pytorch",
            "torch/",
            f"v{ver}",
        ))
    except Exception:
        pass
    return _ROOTS


def collect_source_ref(module: Any) -> dict[str, Any] | None:
    """Return source_ref for a live nn.Module, or None when inspect cannot locate it."""
    cls = type(module)
    try:
        file = inspect.getsourcefile(cls)
        line = inspect.getsourcelines(cls)[1]
    except (TypeError, OSError):
        return None
    if not file:
        return None
    abs_file = os.path.abspath(file)
    class_name = cls.__name__
    module_path = getattr(cls, "__module__", None)
    for root, repo, prefix, version in _package_roots():
        root_abs = os.path.abspath(root)
        if abs_file == root_abs or abs_file.startswith(root_abs + os.sep):
            rel = os.path.relpath(abs_file, root_abs).replace(os.sep, "/")
            framework = Path(root_abs).name
            return {
                "framework": framework,
                "module_path": module_path,
                "class_name": class_name,
                "file": f"{prefix}{rel}",
                "line": line,
                "version": version.lstrip("v"),
                "url": f"https://github.com/{repo}/blob/{version}/{prefix}{rel}#L{line}",
            }
    return {
        "framework": None,
        "module_path": module_path,
        "class_name": class_name,
        "file": abs_file,
        "line": line,
        "version": None,
        "url": None,
    }


def flatten_source_refs(graph: Any) -> list[dict[str, Any]]:
    """Emit catalog-side rows: (module_path, class_name, source_ref, has_params).

    Aggregation / fold groups (layer-group, layer-pattern-group) keep
    ``source_ref=None`` — they are not a single class definition.
    """
    nodes = getattr(graph, "nodes", None) or []
    rows: list[dict[str, Any]] = []
    for node in nodes:
        node_type = getattr(node, "type", None) or ""
        attributes = getattr(node, "attributes", None) or {}
        source_ref = attributes.get("source_ref") if isinstance(attributes, dict) else None
        if node_type in {"layer-group", "layer-pattern-group"}:
            source_ref = None
        class_name = attributes.get("class") if isinstance(attributes, dict) else None
        rows.append({
            "module_path": getattr(node, "canonical_id", None) or getattr(node, "id", None),
            "class_name": class_name,
            "source_ref": source_ref,
            "has_params": getattr(node, "params", None) is not None and getattr(node, "params", 0) > 0,
        })
    return rows
