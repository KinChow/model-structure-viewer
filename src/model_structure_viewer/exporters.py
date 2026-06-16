from __future__ import annotations

import json
import re
from typing import Callable

from .schemas import ModelStructure, StructureNode

NodeVisitor = Callable[[str, StructureNode], None]
EdgeVisitor = Callable[[str, str], None]


def export_structure(structure: ModelStructure, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(structure.model_dump(), indent=2, ensure_ascii=False) + "\n"
    if fmt == "mermaid":
        return export_mermaid(structure)
    if fmt == "dot":
        return export_dot(structure)
    raise ValueError(f"Unsupported export format: {fmt}")


def _walk_tree(
    root: StructureNode,
    *,
    on_node: NodeVisitor,
    on_edge: EdgeVisitor,
) -> None:
    def visit(node: StructureNode, parent_id: str | None, path: str) -> None:
        node_id = _safe_id(path or node.id)
        on_node(node_id, node)
        if parent_id is not None:
            on_edge(parent_id, node_id)
        for index, child in enumerate(node.children):
            child_path = (
                f"{path}.{index}.{child.id}" if path else f"{node.id}.{index}.{child.id}"
            )
            visit(child, node_id, child_path)

    visit(root, None, "")


def export_mermaid(structure: ModelStructure) -> str:
    lines = ["flowchart TD"]
    _walk_tree(
        structure.root,
        on_node=lambda nid, node: lines.append(f'  {nid}["{_escape(_label(node))}"]'),
        on_edge=lambda src, dst: lines.append(f"  {src} --> {dst}"),
    )
    return "\n".join(lines) + "\n"


def export_dot(structure: ModelStructure) -> str:
    lines = [
        "digraph ModelStructure {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", fontname="Helvetica"];',
        '  edge [color="#64748b"];',
    ]
    _walk_tree(
        structure.root,
        on_node=lambda nid, node: lines.append(f'  {nid} [label="{_escape(_label(node))}"];'),
        on_edge=lambda src, dst: lines.append(f"  {src} -> {dst};"),
    )
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
