from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import StructureGraph, StructureGraphEdge, StructureGraphNode, StructureNode


@dataclass
class GraphDraft:
    """Mutable graph builder used by backend adapters.

    Adapters add facts and hierarchy directly to this draft.  The legacy tree
    is only produced after the draft is finalized, for compatibility payloads.
    """

    nodes: list[StructureGraphNode] = field(default_factory=list)
    edges: list[StructureGraphEdge] = field(default_factory=list)
    root_id: str = "root"

    def add_node(
        self,
        *,
        node_id: str,
        canonical_id: str,
        parent_id: str | None,
        order: int,
        name: str,
        type: str,
        repeat: int | None = None,
        attributes: dict[str, Any] | None = None,
        source_fields: list[str] | None = None,
        confidence: str = "high",
        params: int | None = None,
        weight_shapes: dict[str, list[int]] | None = None,
        dtype: str | None = None,
        input_shape: list[int] | None = None,
        output_shape: list[int] | None = None,
        value_source: str | None = None,
        tensor_names: list[str] | None = None,
    ) -> None:
        self.nodes.append(StructureGraphNode(
            id=node_id,
            canonical_id=canonical_id,
            module_id=canonical_id,
            parent_id=parent_id,
            order=order,
            name=name,
            type=type,
            repeat=repeat,
            attributes=attributes or {},
            source_fields=source_fields or [],
            confidence=confidence,
            params=params,
            weight_shapes=weight_shapes,
            dtype=dtype,
            input_shape=input_shape,
            output_shape=output_shape,
            value_source=value_source,
            tensor_names=tensor_names,
        ))

    def add_dataflow(self, source: str, target: str, *, evidence: str = "module-order") -> None:
        by_id = {node.id: node for node in self.nodes}
        self.edges.append(StructureGraphEdge(
            id=f"{source}~{target}",
            source=source,
            target=target,
            source_canonical_id=(by_id.get(source).canonical_id if source in by_id else None),
            target_canonical_id=(by_id.get(target).canonical_id if target in by_id else None),
            evidence=evidence,
        ))

    def finalize(self) -> StructureGraph:
        return StructureGraph(root_id=self.root_id, nodes=list(self.nodes), edges=list(self.edges))


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


def collapse_graph(graph: StructureGraph) -> StructureGraph:
    """Apply legacy repeat folding at the graph boundary.

    Folding rules still live in ``fold.py`` while they are being migrated, but
    the backend adapter now owns a graph before this compatibility operation.
    """
    from . import fold

    return materialize_structure_graph(fold.collapse(project_graph_to_tree(graph)))


def project_graph_to_tree(graph: StructureGraph) -> StructureNode:
    """Create the legacy hierarchy view from the graph node index."""
    by_id = {node.id: StructureNode(
        id=node.canonical_id or node.module_id or node.id,
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
