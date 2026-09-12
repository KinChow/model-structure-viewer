import json
from pathlib import Path

from model_structure_viewer.verification.compare_structure import (
    canonical_reconciliation_path,
    compare_structure_summary,
    diff_module_evidence,
)

# fixture 是生产规则输入，随包分发（verification/fixtures/）；tests 只是消费者。
_FIXTURE = Path(__file__).parent.parent / "src" / "model_structure_viewer" / "verification" / "fixtures" / "canonical_path_contract.json"


def test_compare_structure_summary_passes_matching_layers_and_architecture():
    result = compare_structure_summary(
        predicted={
            "summary": {
                "architecture": "DeepseekV3ForCausalLM",
                "text_layers": 61,
            }
        },
        reference={
            "summary": {
                "architecture": "DeepseekV3ForCausalLM",
                "text_layers": 61,
            }
        },
    )

    assert result["status"] == "passed"
    assert result["errors"] == []


def test_compare_structure_summary_fails_layer_mismatch():
    result = compare_structure_summary(
        predicted={"summary": {"architecture": "DeepseekV3ForCausalLM", "text_layers": 60}},
        reference={"summary": {"architecture": "DeepseekV3ForCausalLM", "text_layers": 61}},
    )

    assert result["status"] == "failed"
    assert "text_layers mismatch" in result["errors"][0]


def _backend_modules():
    """合成 evidence（transformers_verify.extract_evidence_modules 的产物形态）。"""
    return [
        {
            "path": "root.model.layers.0.self_attn.q_proj",
            "class": "Linear",
            "params": 4194304,
            "weight_shapes": {"weight": [4096, 1024]},
            "dtype": "BF16",
            "value_source": "introspect",
            "repeat": None,
        },
        {
            "path": "root.model.layers.0.self_attn.k_proj",
            "class": "Linear",
            "params": 524288,
            "weight_shapes": {"weight": [512, 1024]},
            "dtype": "BF16",
            "value_source": "introspect",
            "repeat": None,
        },
        {
            "path": "root.model.layers.0.mlp.down_proj",
            "class": "Linear",
            "params": 3670016,
            "weight_shapes": {"weight": [1024, 3584]},
            "dtype": "BF16",
            "value_source": "introspect",
            "repeat": None,
        },
        {
            "path": "root.model.norm",
            "class": "Qwen3RMSNorm",
            "params": 1024,
            "weight_shapes": {"weight": [1024]},
            "dtype": "BF16",
            "value_source": "introspect",
            "repeat": None,
        },
    ]


def _msv_graph():
    """合成前端 Graph（整体形态 {"nodes": [...]}），覆盖三分类各一例。"""
    return {
        "nodes": [
            # 全匹配：path 命中 + class 一致 + weight_shapes 正维一致
            {
                "id": "root.0",
                "canonical_id": "decoder.0.self_attn.q_proj",
                "type": "operator",
                "attributes": {"class": "Linear", "operator_id": "linear", "weightMatrices": [{"class": "tp", "out": 4096, "in": 1024}]},
                "weight_shapes": {"weight": [4096, 1024]},
            },
            # class mismatch：path 命中但 class/operator_id 都对不上
            {
                "id": "root.1",
                "canonical_id": "decoder.0.mlp.down_proj",
                "type": "operator",
                "attributes": {"class": "Conv1d", "operator_id": "conv1d", "weightMatrices": [{"class": "tp", "out": 1024, "in": 3584}]},
                "weight_shapes": {"weight": [1024, 3584]},
            },
            # shape mismatch：path/class 命中但正维分歧（4096 vs 2048）
            {
                "id": "root.2",
                "canonical_id": "decoder.0.self_attn.k_proj",
                "type": "operator",
                "attributes": {"class": "Linear", "operator_id": "linear", "weightMatrices": [{"class": "tp", "out": 512, "in": 2048}]},
                "weight_shapes": {"weight": [512, 2048]},
            },
            # only_msv：后端无（P0-2：须带声明才参与对账；且不得命中
            # known_divergences——^lm_head 是 tied 专属登记，这里用假想模块）
            {"id": "root.3", "canonical_id": "decoder.custom_head", "type": "output", "attributes": {"class": "Linear", "weightMatrices": [{"class": "tp", "out": 1, "in": 1}]}},
        ]
    }


def test_diff_module_evidence_classifies_three_way():
    diff = diff_module_evidence(transformers_modules=_backend_modules(), msv_graph=_msv_graph())

    assert diff["only_transformers"] == ["norm"]
    assert diff["only_msv"] == ["decoder.custom_head"]
    assert {(entry["path"], entry["kind"]) for entry in diff["mismatches"]} == {
        ("decoder.mlp.down_proj", "class"),
        ("decoder.self_attn.k_proj", "shape"),
    }
    shape_entry = next(entry for entry in diff["mismatches"] if entry["kind"] == "shape")
    assert shape_entry["transformers"] == {"weight": [512, 1024]}
    assert shape_entry["msv"] == {"weight": [512, 2048]}


def test_diff_module_evidence_clean_match_reports_empty_buckets():
    msv_graph = {
        "nodes": [
            {
                "id": "root.0",
                "canonical_id": "decoder.0.self_attn.q_proj",
                "attributes": {"class": "Linear"},
                "weight_shapes": {"weight": [4096, 1024]},
            },
            # 后缀容忍：前端通用标签是后端专有类名的尾部
            {"id": "root.1", "canonical_id": "norm", "attributes": {"class": "RMSNorm"}},
        ]
    }

    diff = diff_module_evidence(
        transformers_modules=[_backend_modules()[0], _backend_modules()[3]],
        msv_graph=msv_graph,
    )

    assert diff["only_transformers"] == []
    assert diff["only_msv"] == []
    assert diff["mismatches"] == []


def test_diff_module_evidence_skips_placeholder_dims_and_missing_shapes():
    modules = [
        {
            "path": "root.model.layers.0.self_attn.q_proj",
            "class": "Linear",
            "params": None,
            "weight_shapes": {"weight": [4096, 1024]},
            "dtype": None,
            "value_source": "introspect",
            "repeat": None,
        }
    ]
    # -1 占位维不参与比较；weight_shapes 缺失的节点整段跳过 shape 检查
    msv_graph = {
        "nodes": [
            {
                "id": "root.0",
                "canonical_id": "decoder.0.self_attn.q_proj",
                "attributes": {"class": "Linear"},
                "weight_shapes": {"weight": [4096, -1]},
            },
            {"id": "root.1", "canonical_id": "decoder.0.self_attn.o_proj", "attributes": {"class": "Linear"}},
        ]
    }

    diff = diff_module_evidence(transformers_modules=modules, msv_graph=msv_graph)

    assert diff["mismatches"] == []
    assert diff["only_msv"] == ["decoder.self_attn.o_proj"]
    assert diff["only_transformers"] == []


def test_canonical_reconciliation_path_folds_wrappers_and_instances():
    assert canonical_reconciliation_path("root.model.layers.0.self_attn.q_proj") == "decoder.self_attn.q_proj"
    assert canonical_reconciliation_path("model.language_model.embed_tokens") == "embed_tokens"
    assert canonical_reconciliation_path("root.visual.blocks.0.group0") == "vision_tower.blocks"
    assert canonical_reconciliation_path("root.language_model.layers.0.group0.pattern0") == "decoder"
    # 整树节点折叠为空键——调用方据空键排除出 diff
    assert canonical_reconciliation_path("root") == ""
    assert canonical_reconciliation_path("model") == ""


def test_canonical_path_contract_fixture_pairs_reconcile():
    contract = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert contract["pairs"], "契约样例不能为空"
    for pair in contract["pairs"]:
        backend_key = canonical_reconciliation_path(pair["backend"])
        frontend_key = canonical_reconciliation_path(pair["frontend"])
        assert backend_key == frontend_key, (
            f"{pair['backend']} <-> {pair['frontend']}: {backend_key!r} != {frontend_key!r}"
        )
        assert backend_key, f"契约样例不应折叠到空键: {pair['backend']}"
