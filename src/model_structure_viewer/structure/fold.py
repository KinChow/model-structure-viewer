"""Collapse isomorphic siblings inside ModuleList-style parents."""
from __future__ import annotations

from typing import Any

from ..schemas import StructureNode

_SIG_CACHE_KEY = "__sig__"


def collapse(node: StructureNode) -> StructureNode:
    """Recursively fold repeated isomorphic children under module-list nodes."""
    folded_children = [collapse(child) for child in node.children]
    if node.type == "module-list" and folded_children:
        folded_children = _fold_consecutive(folded_children)
    return node.model_copy(update={"children": folded_children})


def _fold_consecutive(children: list[StructureNode]) -> list[StructureNode]:
    if not children:
        return children
    groups: list[list[StructureNode]] = []
    current_sig: tuple | None = None
    for child in children:
        sig = _signature(child)
        if groups and sig == current_sig:
            groups[-1].append(child)
        else:
            groups.append([child])
            current_sig = sig

    folded: list[StructureNode] = []
    for index, group in enumerate(groups):
        head = group[0]
        if len(group) == 1:
            folded.append(head)
            continue
        start_name = head.name
        end_name = group[-1].name
        range_label = f"{start_name}..{end_name}" if start_name != end_name else start_name
        attributes = dict(head.attributes)
        attributes["range"] = range_label
        folded.append(
            head.model_copy(
                update={
                    "id": f"{head.id}.group{index}",
                    "name": f"{head.name} x{len(group)}",
                    "type": "layer-group",
                    "repeat": len(group),
                    "attributes": attributes,
                }
            )
        )
    return folded


def _signature(node: StructureNode) -> tuple:
    """Class-shape signature for isomorphism: type + class label + recursive children."""
    class_label = node.attributes.get("class") if isinstance(node.attributes, dict) else None
    return (
        node.type,
        class_label,
        tuple(_signature(child) for child in node.children),
    )
