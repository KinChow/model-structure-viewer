"""折叠规则单源：对 Graph IR 原地折叠 module-list 下的同构兄弟。

P7（执行路线步骤 7）：图→树投影与 graph→tree→fold→graph 兼容链路退役——
折叠规则从 StructureNode 树迁到 StructureGraph 节点上，签名、
分组、命名规则与原 fold.collapse（树版）逐条一致。输出契约也不变：折叠后的
图节点 id 重排为位序路径（root.0.0…），语义身份（含 group/pattern 后缀）保留在
canonical_id/module_id，边只保留 module-order 链（与旧 materialize 一致）。
"""
from __future__ import annotations

from ..schemas import StructureGraph, StructureGraphEdge, StructureGraphNode


def _weight_shapes_key(weight_shapes: dict[str, list[int]] | None) -> tuple | None:
    """把 weight_shapes 归一化为可哈希、可比较的签名（排序后的 (参数名, shape) 元组）。"""
    if not weight_shapes:
        return None
    return tuple(sorted((name, tuple(shape)) for name, shape in weight_shapes.items()))


def collapse(graph: StructureGraph) -> StructureGraph:
    """Fold repeated isomorphic children under module-list nodes, on the graph."""
    nodes_by_id = {node.id: node for node in graph.nodes}
    order_key = {node.id: (node.order, node.id) for node in graph.nodes}
    children_ids: dict[str, list[str]] = {}
    for node in graph.nodes:
        children_ids.setdefault(node.parent_id, []).append(node.id)
    for ids in children_ids.values():
        ids.sort(key=lambda nid: order_key[nid])

    # 折叠域：folded_node[node_id] 是该节点（或其折叠替身）的当前形态，
    # folded_kids[node_id] 是折叠后的子节点 id 列表（位序在 emit 阶段重排），
    # folded_canonical[node_id] 是语义身份链（canonical.groupN.patternN…）——
    # 与旧树投影的树节点 id 同源（canonical_id || module_id || id）。
    folded_node: dict[str, StructureGraphNode] = {}
    folded_kids: dict[str, list[str]] = {}
    folded_canonical: dict[str, str] = {}

    def signature(node_id: str) -> tuple:
        """Class-shape signature for isomorphism: type + class label + weight_shapes + recursive children.

        v4/G3：签名纳入真实 weight_shapes（数值形状），中间维度不同的层不再被
        误折叠（如 first_k_dense_replace 的 dense/MoE 混合层）。weight_shapes 为
        None 时行为与旧版一致。
        """
        node = folded_node[node_id]
        class_label = node.attributes.get("class") if isinstance(node.attributes, dict) else None
        return (
            node.type,
            class_label,
            _weight_shapes_key(node.weight_shapes),
            tuple(signature(child_id) for child_id in folded_kids[node_id]),
        )

    def fold_consecutive(child_ids: list[str]) -> list[str]:
        if not child_ids:
            return child_ids
        groups: list[list[str]] = []
        current_sig: tuple | None = None
        for child_id in child_ids:
            sig = signature(child_id)
            if groups and sig == current_sig:
                groups[-1].append(child_id)
            else:
                groups.append([child_id])
                current_sig = sig

        folded: list[str] = []
        for index, group in enumerate(groups):
            head_id = group[0]
            if len(group) == 1:
                folded.append(head_id)
                continue
            head = folded_node[head_id]
            tail = folded_node[group[-1]]
            start_name = head.name
            end_name = tail.name
            range_label = f"{start_name}..{end_name}" if start_name != end_name else start_name
            attributes = dict(head.attributes)
            attributes["range"] = range_label
            group_id = f"{head_id}.group{index}"
            attributes.pop("source_ref", None)
            folded_node[group_id] = head.model_copy(
                update={
                    "id": group_id,
                    "name": f"{head.name} x{len(group)}",
                    "type": "layer-group",
                    "repeat": len(group),
                    "attributes": attributes,
                }
            )
            folded_canonical[group_id] = f"{folded_canonical[head_id]}.group{index}"
            folded_kids[group_id] = list(folded_kids[head_id])
            folded.append(group_id)
        return folded

    def fold_repeated_patterns(child_ids: list[str]) -> list[str]:
        """Combine alternating folded groups such as A×3 + B + A×3 into one pattern."""
        if len(child_ids) < 3:
            return child_ids

        folded: list[str] = []
        index = 0
        while index < len(child_ids):
            match = _match_group_separator_pattern(child_ids, index, signature, folded_node)
            if match is None:
                folded.append(child_ids[index])
                index += 1
                continue
            end_index = match
            pattern_children = child_ids[index : end_index + 1]
            group_id = _make_pattern_group(
                pattern_children, len(folded), folded_node, folded_kids, folded_canonical
            )
            folded.append(group_id)
            index = end_index + 1
        return folded

    def collapse_level(node_id: str) -> list[str]:
        node = nodes_by_id[node_id]
        folded: list[str] = []
        for child_id in children_ids.get(node_id, []):
            child = nodes_by_id[child_id]
            folded_node[child_id] = child
            folded_canonical[child_id] = child.canonical_id or child.module_id or child.id
            folded_kids[child_id] = collapse_level(child_id)
            folded.append(child_id)
        if node.type == "module-list" and folded:
            folded = fold_consecutive(folded)
            folded = fold_repeated_patterns(folded)
        return folded

    root_id = graph.root_id
    root = nodes_by_id[root_id]
    folded_node[root_id] = root
    folded_canonical[root_id] = root.canonical_id or root.module_id or root.id
    folded_kids[root_id] = collapse_level(root_id)

    # 位序路径重建（旧 materialize 契约）：id=root.0.0…，canonical/module 保留
    # 语义身份；边只保留 module-order 链并回填 canonical 端点。
    nodes: list[StructureGraphNode] = []
    edges: list[StructureGraphEdge] = []

    def emit(node_id: str, path: str, parent_path: str | None) -> None:
        nodes.append(folded_node[node_id].model_copy(update={
            "id": path,
            "canonical_id": folded_canonical[node_id],
            "module_id": folded_canonical[node_id],
            "parent_id": parent_path,
            "order": int(path.rsplit(".", 1)[-1]) if "." in path else 0,
        }))
        child_paths = [f"{path}.{index}" for index in range(len(folded_kids[node_id]))]
        for source, target in zip(child_paths, child_paths[1:]):
            edges.append(
                StructureGraphEdge(
                    id=f"{source}~{target}",
                    source=source,
                    target=target,
                    evidence="module-order",
                )
            )
        for child_id, child_path in zip(folded_kids[node_id], child_paths):
            emit(child_id, child_path, path)

    emit(root_id, root_id, None)
    canonical_by_path = {node.id: node.canonical_id or node.module_id for node in nodes}
    edges = [edge.model_copy(update={
        "source_canonical_id": canonical_by_path.get(edge.source),
        "target_canonical_id": canonical_by_path.get(edge.target),
    }) for edge in edges]
    return StructureGraph(root_id=root_id, nodes=nodes, edges=edges)


def _match_group_separator_pattern(
    children: list[str],
    start: int,
    signature,
    folded_node: dict[str, StructureGraphNode],
) -> int | None:
    first = folded_node[children[start]]
    if first.type != "layer-group" or first.repeat is None:
        return None

    group_sig = signature(children[start])
    separator_sig: tuple | None = None
    index = start + 1
    repeat_count = 1
    while index + 1 < len(children):
        separator = folded_node[children[index]]
        next_group = folded_node[children[index + 1]]
        if next_group.type != "layer-group" or next_group.repeat != first.repeat:
            break
        if signature(children[index + 1]) != group_sig:
            break
        current_separator_sig = signature(children[index])
        if separator_sig is None:
            separator_sig = current_separator_sig
        elif current_separator_sig != separator_sig:
            break
        repeat_count += 1
        index += 2

    if repeat_count <= 1:
        return None
    if index < len(children) and signature(children[index]) == separator_sig:
        return index
    return index - 2


def _make_pattern_group(
    children: list[str],
    group_index: int,
    folded_node: dict[str, StructureGraphNode],
    folded_kids: dict[str, list[str]],
    folded_canonical: dict[str, str],
) -> str:
    head_id = children[0]
    tail_id = children[-1]
    head = folded_node[head_id]
    tail = folded_node[tail_id]
    attributes = dict(head.attributes)
    attributes.pop("source_ref", None)
    attributes["range"] = _range_from_nodes(head, tail)
    attributes["pattern"] = " + ".join(_pattern_part(folded_node[child_id]) for child_id in children[:2])
    group_id = f"{head_id}.pattern{group_index}"
    folded_node[group_id] = head.model_copy(
        update={
            "id": group_id,
            "name": f"{_class_label(head)} pattern x{(len(children) + 1) // 2}",
            "type": "layer-pattern-group",
            "repeat": (len(children) + 1) // 2,
            "attributes": attributes,
        }
    )
    folded_canonical[group_id] = f"{folded_canonical[head_id]}.pattern{group_index}"
    folded_kids[group_id] = list(children[:2])
    return group_id


def _range_from_nodes(head: StructureGraphNode, tail: StructureGraphNode) -> str:
    start = str(head.name).split(" ", 1)[0]
    tail_range = tail.attributes.get("range") if isinstance(tail.attributes, dict) else None
    end = str(tail_range).rsplit("..", 1)[-1] if tail_range else str(tail.name).split(" ", 1)[0]
    return f"{start}..{end}"


def _pattern_part(node: StructureGraphNode) -> str:
    label = _class_label(node)
    return f"{label} x{node.repeat}" if node.repeat else label


def _class_label(node: StructureGraphNode) -> str:
    if isinstance(node.attributes, dict) and node.attributes.get("class"):
        return str(node.attributes["class"])
    return node.name
