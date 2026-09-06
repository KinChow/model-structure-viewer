from __future__ import annotations

from ..schemas import StructureGraph, StructureGraphEdge, StructureGraphNode, StructureNode


def materialize_structure_graph(root: StructureNode) -> StructureGraph:
    nodes: list[StructureGraphNode] = []
    edges: list[StructureGraphEdge] = []

    def visit(node: StructureNode, path: str, parent_id: str | None = None) -> None:
        nodes.append(
            StructureGraphNode(
                id=path,
                module_id=node.id,
                parent_id=parent_id,
                type=node.type,
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
    return StructureGraph(nodes=nodes, edges=edges)
