from __future__ import annotations

import json
import re

from .schemas import ModelStructure, StructureNode


def export_structure(structure: ModelStructure, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(structure.model_dump(), indent=2, ensure_ascii=False) + "\n"
    if fmt == "mermaid":
        return export_mermaid(structure)
    if fmt == "dot":
        return export_dot(structure)
    raise ValueError(f"Unsupported export format: {fmt}")


def export_mermaid(structure: ModelStructure) -> str:
    lines = ["flowchart TD"]

    def walk(node: StructureNode, parent_id: str | None = None, path: str = "") -> None:
        node_id = _safe_id(path or node.id)
        label = _label(node)
        lines.append(f'  {node_id}["{_escape(label)}"]')
        if parent_id:
            lines.append(f"  {parent_id} --> {node_id}")
        for index, child in enumerate(node.children):
            walk(child, node_id, f"{path}.{index}.{child.id}" if path else f"{node.id}.{index}.{child.id}")

    walk(structure.root)
    return "\n".join(lines) + "\n"


def export_dot(structure: ModelStructure) -> str:
    lines = [
        "digraph ModelStructure {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", fontname="Helvetica"];',
        '  edge [color="#64748b"];',
    ]

    def walk(node: StructureNode, parent_id: str | None = None, path: str = "") -> None:
        node_id = _safe_id(path or node.id)
        lines.append(f'  {node_id} [label="{_escape(_label(node))}"];')
        if parent_id:
            lines.append(f"  {parent_id} -> {node_id};")
        for index, child in enumerate(node.children):
            walk(child, node_id, f"{path}.{index}.{child.id}" if path else f"{node.id}.{index}.{child.id}")

    walk(structure.root)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _label(node: StructureNode) -> str:
    parts = [node.name]
    if node.repeat:
        parts.append(f"x{node.repeat}")
    if node.type:
        parts.append(f"({node.type})")
    return " ".join(parts)


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not value or value[0].isdigit():
        value = f"n_{value}"
    return value


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
