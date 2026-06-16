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
    used: set[str] = set()

    def assign_id(path: str) -> str:
        slug = _slug(path)
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:6]
        candidate = slug or f"n_{digest}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        # Collision: fall back to slug + hash so the id stays unique while
        # remaining readable. ``digest`` is derived from the full path, so two
        # different paths cannot produce the same disambiguated id.
        disambiguated = f"{candidate}_{digest}"
        used.add(disambiguated)
        return disambiguated

    def visit(node: StructureNode, parent_id: str | None, path: str) -> None:
        node_id = assign_id(path or node.id)
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


def _slug(value: str) -> str:
    """Build a graph-safe slug. Empty when the input has no ASCII alnum chars."""
    slug = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_")
    if not slug:
        return ""
    if slug[0].isdigit():
        slug = f"n_{slug}"
    return slug


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
