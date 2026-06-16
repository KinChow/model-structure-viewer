from __future__ import annotations

import hashlib
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
        on_node=lambda nid, node: lines.append(f'  {nid}["{_escape_mermaid(_label(node))}"]'),
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
        on_node=lambda nid, node: lines.append(f'  {nid} [label="{_escape_dot(_label(node))}"];'),
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
    """Build a graph-safe node id that is unique per source path.

    Slug-only ids collapse for non-ASCII names (e.g. CJK), case variants
    (``LMHead`` vs ``lm_head``), and any path that maps to the same ascii
    skeleton. Appending a short content hash keeps the slug human-readable
    while guaranteeing uniqueness across the original ``value``.
    """
    slug = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]
    if not slug:
        return f"n_{digest}"
    if slug[0].isdigit():
        slug = f"n_{slug}"
    return f"{slug}_{digest}"


_MERMAID_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "|": "\\|",
    "{": "\\{",
    "}": "\\}",
    "<": "&lt;",
    ">": "&gt;",
}


def _escape_mermaid(value: str) -> str:
    out = []
    for ch in value:
        out.append(_MERMAID_ESCAPES.get(ch, ch))
    return "".join(out)


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
