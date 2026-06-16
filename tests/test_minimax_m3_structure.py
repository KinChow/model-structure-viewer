import json
from pathlib import Path

from model_structure_viewer.exporters import export_structure
from model_structure_viewer.structure import build_model_structure, is_minimax_m3

FIXTURE = Path(__file__).parent / "fixtures" / "minimax_m3" / "config.json"


def load_config():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def find_node(node, node_id):
    if node.id == node_id:
        return node
    for child in node.children:
        found = find_node(child, node_id)
        if found:
            return found
    return None


def test_detects_minimax_m3():
    assert is_minimax_m3(load_config())


def test_builds_minimax_m3_compressed_structure():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})

    assert structure.summary["model_family"] == "MiniMax-M3"
    assert structure.summary["text_layers"] == 60
    assert structure.summary["vision_layers"] == 32
    assert structure.root.name == "MiniMaxM3 VL Wrapper"

    vision = find_node(structure.root, "vision")
    assert vision is not None
    assert vision.attributes["num_hidden_layers"] == 32
    assert vision.attributes["patch_size"] == 14

    projector = find_node(structure.root, "projector")
    assert projector is not None
    assert projector.attributes["output_dim"] == 6144

    dense = find_node(structure.root, "text.layers.group0")
    sparse = find_node(structure.root, "text.layers.group1")
    assert dense is not None
    assert dense.repeat == 3
    assert dense.attributes["moe"] is False
    assert sparse is not None
    assert sparse.repeat == 57
    assert sparse.attributes["moe"] is True
    assert sparse.attributes["sparse_attention"] is True

    moe = find_node(sparse, "text.layer.ffn")
    assert moe is not None
    assert moe.type == "moe"
    assert moe.attributes["num_local_experts"] == 128
    assert moe.attributes["num_experts_per_token"] == 4

    mtp = find_node(structure.root, "output.mtp")
    assert mtp is not None
    assert mtp.repeat == 7


def test_exports_mermaid_and_dot():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})
    mermaid = export_structure(structure, "mermaid")
    dot = export_structure(structure, "dot")

    assert mermaid.startswith("flowchart TD")
    assert "MiniMaxM3 VL Wrapper" in mermaid
    assert dot.startswith("digraph ModelStructure")
    assert "MiniMaxM3 VL Wrapper" in dot
