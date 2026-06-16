import json
from pathlib import Path

from model_structure_viewer.exporters import export_structure
from model_structure_viewer.structure import build_model_structure, is_deepseek_v3

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v3" / "config.json"


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


def test_detects_deepseek_v3():
    assert is_deepseek_v3(load_config())


def test_builds_deepseek_v3_compressed_structure():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})

    assert structure.summary["model_family"] == "DeepSeek-V3"
    assert structure.summary["text_layers"] == 61
    assert structure.summary["dense_layers"] == 3
    assert structure.summary["moe_layers"] == 58
    assert structure.summary["num_local_experts"] == 256
    assert structure.root.name == "DeepseekV3Model"

    embedding = find_node(structure.root, "deepseek.embedding")
    assert embedding is not None
    assert embedding.attributes["shape"] == "129280 x 7168"

    dense = find_node(structure.root, "deepseek.layers.dense")
    moe = find_node(structure.root, "deepseek.layers.moe")
    assert dense is not None
    assert dense.repeat == 3
    assert dense.attributes["moe"] is False
    assert moe is not None
    assert moe.repeat == 58
    assert moe.attributes["moe"] is True

    attention = find_node(moe, "deepseek.layer.attention")
    assert attention is not None
    assert attention.attributes["q_lora_rank"] == 1536
    assert attention.attributes["kv_lora_rank"] == 512

    moe_node = find_node(moe, "deepseek.layer.moe")
    assert moe_node is not None
    assert moe_node.type == "moe"
    assert moe_node.attributes["routed_experts"] == 256
    assert moe_node.attributes["experts_per_token"] == 8


def test_deepseek_export_contains_specialized_nodes():
    structure = build_model_structure(load_config(), source={"kind": "fixture"})
    mermaid = export_structure(structure, "mermaid")

    assert "DeepseekV3Model" in mermaid
    assert "DeepseekV3 MLA Attention" in mermaid
    assert "DeepseekV3MoE" in mermaid
