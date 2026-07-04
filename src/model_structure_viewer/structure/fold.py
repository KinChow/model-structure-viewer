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
        folded_children = _fold_repeated_patterns(folded_children)
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


def _fold_repeated_patterns(children: list[StructureNode]) -> list[StructureNode]:
    """Combine alternating folded groups such as A×3 + B + A×3 into one pattern."""
    if len(children) < 3:
        return children

    folded: list[StructureNode] = []
    index = 0
    while index < len(children):
        match = _match_group_separator_pattern(children, index)
        if match is None:
            folded.append(children[index])
            index += 1
            continue
        end_index = match
        pattern_children = children[index : end_index + 1]
        folded.append(_make_pattern_group(pattern_children, len(folded)))
        index = end_index + 1
    return folded


def _match_group_separator_pattern(children: list[StructureNode], start: int) -> int | None:
    first = children[start]
    if first.type != "layer-group" or first.repeat is None:
        return None

    group_sig = _signature(first)
    separator_sig: tuple | None = None
    index = start + 1
    repeat_count = 1
    while index + 1 < len(children):
        separator = children[index]
        next_group = children[index + 1]
        if next_group.type != "layer-group" or next_group.repeat != first.repeat:
            break
        if _signature(next_group) != group_sig:
            break
        current_separator_sig = _signature(separator)
        if separator_sig is None:
            separator_sig = current_separator_sig
        elif current_separator_sig != separator_sig:
            break
        repeat_count += 1
        index += 2

    if repeat_count <= 1:
        return None
    if index < len(children) and _signature(children[index]) == separator_sig:
        return index
    return index - 2


def _make_pattern_group(children: list[StructureNode], group_index: int) -> StructureNode:
    head = children[0]
    tail = children[-1]
    attributes = dict(head.attributes)
    attributes["range"] = _range_from_nodes(head, tail)
    attributes["pattern"] = " + ".join(_pattern_part(child) for child in children[:2])
    return head.model_copy(
        update={
            "id": f"{head.id}.pattern{group_index}",
            "name": f"{_class_label(head)} pattern x{(len(children) + 1) // 2}",
            "type": "layer-pattern-group",
            "repeat": (len(children) + 1) // 2,
            "attributes": attributes,
            "children": children[:2],
        }
    )


def _range_from_nodes(head: StructureNode, tail: StructureNode) -> str:
    start = str(head.name).split(" ", 1)[0]
    tail_range = tail.attributes.get("range") if isinstance(tail.attributes, dict) else None
    end = str(tail_range).rsplit("..", 1)[-1] if tail_range else str(tail.name).split(" ", 1)[0]
    return f"{start}..{end}"


def _pattern_part(node: StructureNode) -> str:
    label = _class_label(node)
    return f"{label} x{node.repeat}" if node.repeat else label


def _class_label(node: StructureNode) -> str:
    if isinstance(node.attributes, dict) and node.attributes.get("class"):
        return str(node.attributes["class"])
    return node.name


def _signature(node: StructureNode) -> tuple:
    """Class-shape signature for isomorphism: type + class label + recursive children."""
    class_label = node.attributes.get("class") if isinstance(node.attributes, dict) else None
    return (
        node.type,
        class_label,
        tuple(_signature(child) for child in node.children),
    )
