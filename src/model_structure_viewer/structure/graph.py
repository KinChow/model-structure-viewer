from __future__ import annotations

from ..schemas import StructureGraph, StructureGraphEdge, StructureGraphNode, StructureNode


def materialize_structure_graph(root: StructureNode) -> StructureGraph:
    nodes: list[StructureGraphNode] = []
    edges: list[StructureGraphEdge] = []

    def visit(node: StructureNode, path: str, parent_id: str | None = None) -> None:
        nodes.append(
            StructureGraphNode(
                id=path,
                canonical_id=node.id,
                module_id=node.id,
                parent_id=parent_id,
                order=int(path.rsplit(".", 1)[-1]) if "." in path else 0,
                name=node.name,
                type=node.type,
                repeat=node.repeat,
                attributes=node.attributes,
                source_fields=node.source_fields,
                confidence=node.confidence,
                params=node.params,
                weight_shapes=node.weight_shapes,
                dtype=node.dtype,
                input_shape=node.input_shape,
                output_shape=node.output_shape,
                value_source=node.value_source,
                tensor_names=node.tensor_names,
            )
        )
        child_paths = [f"{path}.{index}" for index in range(len(node.children))]
        for source, target in zip(child_paths, child_paths[1:]):
            edges.append(
                StructureGraphEdge(
                    id=f"{source}~{target}",
                    source=source,
                    target=target,
                    evidence="module-order",
                )
            )
        for child, child_path in zip(node.children, child_paths):
            visit(child, child_path, path)

    visit(root, "root")
    canonical_by_path = {node.id: node.canonical_id or node.module_id for node in nodes}
    edges = [edge.model_copy(update={
        "source_canonical_id": canonical_by_path.get(edge.source),
        "target_canonical_id": canonical_by_path.get(edge.target),
    }) for edge in edges]
    return StructureGraph(nodes=nodes, edges=edges)


def project_graph_to_tree(graph: StructureGraph) -> StructureNode:
    """Create the legacy hierarchy view from the graph node index."""
    by_id = {node.id: StructureNode(
        id=node.module_id or node.id,
        name=node.name,
        type=node.type,
        repeat=node.repeat,
        attributes=node.attributes,
        source_fields=node.source_fields,
        confidence=node.confidence,
        params=node.params,
        weight_shapes=node.weight_shapes,
        dtype=node.dtype,
        input_shape=node.input_shape,
        output_shape=node.output_shape,
        value_source=node.value_source,
        tensor_names=node.tensor_names,
    ) for node in graph.nodes}
    for node in sorted(graph.nodes, key=lambda item: (item.parent_id or "", item.order, item.id)):
        if node.parent_id is None:
            continue
        parent = by_id[node.parent_id]
        parent.children.append(by_id[node.id])
    return by_id[graph.root_id]
