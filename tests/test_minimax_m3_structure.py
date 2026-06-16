"""Tests for the model-agnostic structure builder using minimax_m3 fixture."""
import json
from pathlib import Path

from model_structure_viewer.exporters import export_structure
from model_structure_viewer.structure import build_model_structure

FIXTURE = Path(__file__).parent / "fixtures" / "minimax_m3" / "config.json"


def load_config():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_minimax_m3_summary_extracted():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})

    summary = structure.summary
    assert summary["model_type"] == "minimax_m3_vl"
    assert summary["text_layers"] == 60
    assert summary["vision_layers"] == 32
    assert summary["confidence"] in {"high", "low"}


def test_minimax_m3_export_contains_root_name():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})
    mermaid = export_structure(structure, "mermaid")
    dot = export_structure(structure, "dot")
    assert mermaid.startswith("flowchart TD")
    assert dot.startswith("digraph ModelStructure")
    assert structure.root.name in mermaid


def test_minimax_m3_has_nested_children():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})
    assert structure.root.children, "root should expose vision/text/output children"
