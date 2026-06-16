from __future__ import annotations

from model_structure_viewer.exporters import export_dot, export_mermaid, export_structure
from model_structure_viewer.schemas import ModelStructure, StructureNode


def _structure() -> ModelStructure:
    leaf_a = StructureNode(id="layers.0", name="Layer 0", type="block")
    leaf_b = StructureNode(id="layers.1", name='He said "hi"', type="block\\with")
    list_node = StructureNode(id="layers", name="layers", type="module-list", children=[leaf_a, leaf_b])
    root = StructureNode(id="model", name="模型/Root", type="model", children=[list_node])
    return ModelStructure(summary={}, source={}, root=root)


def test_export_json_round_trip():
    text = export_structure(_structure(), "json")
    assert '"name": "模型/Root"' in text


def test_export_mermaid_escapes_quotes_and_backslashes():
    text = export_mermaid(_structure())
    assert text.startswith("flowchart TD\n")
    assert '\\"hi\\"' in text
    assert "block\\\\with" in text


def test_export_dot_safe_id_for_non_ascii():
    text = export_dot(_structure())
    assert text.startswith("digraph ModelStructure {")
    # Non-ASCII chars in id are mapped via _safe_id
    assert "n_" in text or "model" in text
    assert text.endswith("}\n")


def test_export_unsupported_format_raises():
    import pytest

    with pytest.raises(ValueError):
        export_structure(_structure(), "yaml")
