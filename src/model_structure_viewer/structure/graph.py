from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import StructureGraph, StructureGraphEdge, StructureGraphNode


@dataclass
class GraphDraft:
    """Mutable graph builder used by backend adapters.

    Adapters add facts and hierarchy directly to this draft.  P7（步骤 7）：
    finalize 后的 Graph IR 就是唯一载荷——不再有树视图的二次产出。
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


def collapse_graph(graph: StructureGraph) -> StructureGraph:
    """Apply repeat folding at the graph boundary.

    P7（步骤 7）：折叠规则已迁到图节点上（fold.collapse 直收 StructureGraph），
    旧的 graph→树投影→折叠→重物化兼容链路（图→树投影与树重建两函数）退役。
    输出契约不变：位序路径 id + canonical_id 语义身份 + module-order 边链。
    """
    from . import fold

    return fold.collapse(graph)
