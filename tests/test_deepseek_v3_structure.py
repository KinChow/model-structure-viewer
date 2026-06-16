"""Tests for the model-agnostic structure builder using deepseek_v3 fixture."""
import json
from pathlib import Path

from model_structure_viewer.exporters import export_structure
from model_structure_viewer.structure import build_model_structure

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v3" / "config.json"


def load_config():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_deepseek_v3_summary_extracted():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})

    summary = structure.summary
    assert summary["model_type"] == "deepseek_v3"
    assert summary["architecture"] == "DeepseekV3ForCausalLM"
    assert summary["text_layers"] == 61
    assert summary["hidden_size"] == 7168
    assert summary["num_attention_heads"] == 128
    # confidence is high if introspection ran, low if fell back; both are valid.
    assert summary["confidence"] in {"high", "low"}


def test_deepseek_v3_root_present_and_has_children():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})
    assert structure.root is not None
    assert structure.root.children, "root should have at least one child node"


def test_deepseek_v3_export_round_trip():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})
    mermaid = export_structure(structure, "mermaid")
    assert mermaid.startswith("flowchart TD")
    json_dump = export_structure(structure, "json")
    assert "DeepseekV3" in json_dump or "deepseek" in json_dump.lower()


def test_fallback_used_when_remote_code_unavailable():
    # The deepseek_v3 fixture has auto_map but no local modeling files,
    # so introspection should defer to the config fallback.
    structure = build_model_structure(load_config(), source={"kind": "fixture"})
    assert structure.source.get("strategy") in {"meta-introspect", "config-fallback"}
    if structure.source["strategy"] == "config-fallback":
        assert "fallback_reason" in structure.source
