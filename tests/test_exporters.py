from __future__ import annotations

import pytest

from model_structure_viewer.exporters import export_dot, export_mermaid, export_structure
from model_structure_viewer.schemas import ModelStructure, StructureNode


def _structure() -> ModelStructure:
    leaf_a = StructureNode(id="layers.0", name="Layer 0", type="block")
    leaf_b = StructureNode(id="layers.1", name='He said "hi"', type="block\\with")
    list_node = StructureNode(id="layers", name="layers", type="module-list", children=[leaf_a, leaf_b])
    root = StructureNode(id="model", name="模型/Root", type="model", children=[list_node])
    return ModelStructure(summary={}, source={}, root=root)


def _count_nodes(node: StructureNode) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


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
    n_nodes = _count_nodes(structure.root)
    n_edges = n_nodes - 1
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
    leaf = StructureNode(id="ffn", name="gate|up|down", type="MLP")
    root = StructureNode(id="root", name="root", type="model", children=[leaf])
    structure = ModelStructure(summary={}, source={}, root=root)
    text = export_mermaid(structure)
    # Pipes inside labels must be escaped so mermaid does not treat them as node shape syntax.
    assert "gate\\|up\\|down" in text
    assert "|" not in text.replace("\\|", "")


def test_safe_id_disambiguates_case_collisions():
    a = StructureNode(id="lm_head", name="lm_head", type="Linear")
    b = StructureNode(id="LMHead", name="LMHead", type="Linear")
    root = StructureNode(id="root", name="root", type="model", children=[a, b])
    structure = ModelStructure(summary={}, source={}, root=root)
    text = export_dot(structure)
    # Extract node ids (first token on indented lines containing '[label=').
    ids = []
    for line in text.splitlines():
        stripped = line.strip()
        if "[label=" in stripped:
            ids.append(stripped.split(" ", 1)[0])
    assert len(ids) == len(set(ids)), f"duplicate ids in {ids}"


def test_safe_id_handles_pure_cjk_paths():
    leaf = StructureNode(id="模块", name="模块A", type="Linear")
    root = StructureNode(id="模型", name="模型", type="model", children=[leaf])
    structure = ModelStructure(summary={}, source={}, root=root)
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
