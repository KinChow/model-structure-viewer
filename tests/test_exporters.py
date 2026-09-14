from __future__ import annotations

import pytest

from model_structure_viewer.exporters import export_dot, export_mermaid, export_structure
from model_structure_viewer.schemas import ModelStructure, StructureGraph, StructureGraphEdge, StructureGraphNode


def _structure() -> ModelStructure:
    nodes = [
        StructureGraphNode(id="root", canonical_id="model", name="模型/Root", type="model"),
        StructureGraphNode(id="root.0", canonical_id="layers", name="layers", type="module-list", parent_id="root", order=0),
        StructureGraphNode(id="root.0.0", canonical_id="layers.0", name="Layer 0", type="block", parent_id="root.0", order=0),
        StructureGraphNode(id="root.0.1", canonical_id="layers.1", name='He said "hi"', type="block\\with", parent_id="root.0", order=1),
    ]
    edges = [
        StructureGraphEdge(id="root~root.0", source="root", target="root.0", evidence="module-order"),
        StructureGraphEdge(id="root.0~root.0.0", source="root.0", target="root.0.0", evidence="module-order"),
        StructureGraphEdge(id="root.0~root.0.1", source="root.0", target="root.0.1", evidence="module-order"),
    ]
    return ModelStructure(summary={}, source={}, graph=StructureGraph(nodes=nodes, edges=edges))


def test_export_json_round_trip():
    text = export_structure(_structure(), "json")
    assert '"name": "模型/Root"' in text


def test_export_mermaid_escapes_quotes_and_backslashes():
    structure = _structure()
    text = export_mermaid(structure)
    assert text.startswith("flowchart TD\n")
    assert '\\"hi\\"' in text
    assert "block\\\\with" in text
    # Structural assertions: one line per node + one per edge + header.
    n_nodes = len(structure.graph.nodes)
    n_edges = len(structure.graph.edges)
    body_lines = [line for line in text.splitlines() if line.startswith("  ")]
    assert len(body_lines) == n_nodes + n_edges


def test_export_dot_safe_id_for_non_ascii():
    text = export_dot(_structure())
    assert text.startswith("digraph ModelStructure {")
    assert text.endswith("}\n")
    # CJK label is preserved, but ids must remain ASCII-safe.
    assert 'label="模型/Root (model)"' in text
    for line in text.splitlines():
        stripped = line.strip()
        if "[label=" in stripped:
            node_id = stripped.split(" ", 1)[0]
            assert all(ord(ch) < 128 for ch in node_id), node_id


def test_mermaid_pipe_in_label_does_not_break_graph():
    structure = ModelStructure(summary={}, source={}, graph=StructureGraph(nodes=[
        StructureGraphNode(id="root", canonical_id="root", name="root", type="model"),
        StructureGraphNode(id="root.0", canonical_id="ffn", name="gate|up|down", type="MLP", parent_id="root", order=0),
    ]))
    text = export_mermaid(structure)
    # Pipes inside labels must be escaped so mermaid does not treat them as node shape syntax.
    assert "gate\\|up\\|down" in text
    assert "|" not in text.replace("\\|", "")


def test_safe_id_disambiguates_slug_collisions():
    # 图节点 id 是唯一路径，但 slug 化仍可能相撞（"a.b" 与 "a_b" 同 slug）；
    # 哈希后缀必须保证导出 id 唯一。
    structure = ModelStructure(summary={}, source={}, graph=StructureGraph(nodes=[
        StructureGraphNode(id="a.b", canonical_id="a.b", name="dot path", type="Linear"),
        StructureGraphNode(id="a_b", canonical_id="a_b", name="underscore path", type="Linear"),
    ]))
    text = export_dot(structure)
    # Extract node ids (first token on indented lines containing '[label=').
    ids = []
    for line in text.splitlines():
        stripped = line.strip()
        if "[label=" in stripped:
            ids.append(stripped.split(" ", 1)[0])
    assert len(ids) == len(set(ids)), f"duplicate ids in {ids}"


def test_safe_id_handles_pure_cjk_paths():
    structure = ModelStructure(summary={}, source={}, graph=StructureGraph(
        nodes=[
            StructureGraphNode(id="模型", canonical_id="模型", name="模型", type="model"),
            StructureGraphNode(id="模型.0", canonical_id="模块", name="模块A", type="Linear", parent_id="模型", order=0),
        ],
        edges=[StructureGraphEdge(id="模型~模型.0", source="模型", target="模型.0", evidence="module-order")],
    ))
    text = export_mermaid(structure)
    body_lines = [line.strip() for line in text.splitlines() if line.startswith("  ")]
    # Two nodes + one edge.
    assert len(body_lines) == 3
    # Hash suffix ensures the CJK-only path produced a non-empty id.
    node_lines = [line for line in body_lines if "[" in line]
    assert all(line.split("[", 1)[0].strip() for line in node_lines)


def test_export_unsupported_format_raises():
    with pytest.raises(ValueError):
        export_structure(_structure(), "yaml")
