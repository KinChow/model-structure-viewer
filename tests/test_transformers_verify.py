from model_structure_viewer import service
from model_structure_viewer.schemas import (
    ModelStructure,
    StructureGraph,
    StructureGraphNode,
    VerifyRequest,
)
from model_structure_viewer.settings import AppSettings
from model_structure_viewer.verification.transformers_verify import verify_transformers_structure


def _structure():
    # P7（步骤 7）：graph 是唯一必需载荷——root-only 兼容夹具退役。
    return ModelStructure(
        summary={"strategy": "meta-introspect", "backbone_class": "DemoModel"},
        source={"strategy": "meta-introspect", "backbone_class": "DemoModel"},
        graph=StructureGraph(nodes=[StructureGraphNode(id="root", canonical_id="root", name="DemoModel", type="module")]),
    )


def _graph_structure():
    """带折叠图的结构：root / model / layers(ModuleList, repeat=4) / q_proj / norm。

    norm 节点刻意不带 params/weight_shapes/dtype/value_source，验证 evidence
    的 None 保全（不伪造数值）。
    """
    nodes = [
        StructureGraphNode(id="root", canonical_id="root", type="module", attributes={"class": "DemoModel"}),
        StructureGraphNode(
            id="root.model", canonical_id="root.model", parent_id="root", type="module",
            attributes={"class": "DemoTextModel"},
        ),
        StructureGraphNode(
            id="root.model.layers", canonical_id="root.model.layers", parent_id="root.model",
            type="module-list", attributes={"class": "ModuleList"},
        ),
        StructureGraphNode(
            id="root.model.layers.0", canonical_id="root.model.layers.0", parent_id="root.model.layers",
            type="layer-group", repeat=4, attributes={"class": "DemoDecoderLayer"},
        ),
        StructureGraphNode(
            id="root.model.layers.0.self_attn.q_proj",
            canonical_id="root.model.layers.0.self_attn.q_proj",
            parent_id="root.model.layers.0", type="linear", params=4096,
            weight_shapes={"weight": [64, 64]}, dtype="BF16", value_source="introspect",
            attributes={"class": "Linear"},
        ),
        StructureGraphNode(
            id="root.model.norm", canonical_id="root.model.norm", parent_id="root.model",
            type="normalization", attributes={"class": "DemoRMSNorm"},
        ),
    ]
    return ModelStructure(summary={"strategy": "meta-introspect"}, source={}, graph=StructureGraph(nodes=nodes))


def test_verify_transformers_structure_passes_meta_introspection(monkeypatch):
    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        lambda config, **kwargs: _structure(),
    )

    result = verify_transformers_structure(
        {"model_type": "demo", "architectures": ["DemoModel"]},
        source={"kind": "test", "model_id": "Org/Demo"},
    )

    assert result.ok is True
    assert result.status == "passed"
    assert result.strategy == "transformers-meta"
    assert result.model_id == "Org/Demo"
    assert result.summary["backbone_class"] == "DemoModel"


def test_verify_transformers_structure_failure_returns_error(monkeypatch):
    def fail_meta(config, **kwargs):
        from model_structure_viewer.errors import IntrospectionError

        raise IntrospectionError("AutoModel.from_config failed: unsupported")

    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        fail_meta,
    )

    result = verify_transformers_structure(
        {"model_type": "demo", "architectures": ["DemoModel"], "num_hidden_layers": 2},
        source={"kind": "test", "model_id": "Org/Demo"},
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.strategy == "transformers-meta"
    assert result.diagnostics["failure_kind"] == "model_init_failed"
    assert result.summary.get("strategy") is None


def test_verify_transformers_structure_retries_without_flash_attention(monkeypatch):
    from model_structure_viewer.errors import IntrospectionError

    calls = {"count": 0}

    class FakeConfig:
        pass

    class FakeVisionConfig:
        _attn_implementation = "flash_attention_2"

    def flaky_meta(config, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise IntrospectionError(
                "AutoModel.from_config failed: FlashAttention2 has been toggled on, "
                "but the package for FlashAttention2 doesn't seem to be installed."
            )
        hf_config = FakeConfig()
        hf_config.vision_config = FakeVisionConfig()
        diagnostics = kwargs["config_normalizer"].normalize(hf_config)
        assert hf_config.vision_config._attn_implementation == "sdpa"
        structure = _structure()
        structure.source["diagnostics"] = diagnostics
        return structure

    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        flaky_meta,
    )

    result = verify_transformers_structure(
        {"model_type": "demo", "architectures": ["DemoModel"]},
        source={"kind": "test", "model_id": "Org/Demo"},
    )

    assert calls["count"] == 2
    assert result.ok is True
    assert result.status == "passed"
    assert result.strategy == "transformers-meta"
    assert result.summary["strategy"] == "attention-normalized-transformers-meta"
    assert result.diagnostics["attention_backend_retry"] == "sdpa"


def test_verify_transformers_structure_retries_when_flash_attention_is_unsupported(monkeypatch):
    from model_structure_viewer.errors import IntrospectionError

    calls = {"count": 0}

    class FakeConfig:
        _attn_implementation = "flash_attention_2"

    def flaky_meta(config, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise IntrospectionError(
                "AutoModel.from_config failed: MoonViT3dPretrainedModel "
                "does not support Flash Attention 2 yet."
            )
        hf_config = FakeConfig()
        diagnostics = kwargs["config_normalizer"].normalize(hf_config)
        assert hf_config._attn_implementation == "sdpa"
        structure = _structure()
        structure.source["diagnostics"] = diagnostics
        return structure

    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        flaky_meta,
    )

    result = verify_transformers_structure(
        {"model_type": "demo", "architectures": ["DemoModel"]},
        source={"kind": "test", "model_id": "Org/Demo"},
    )

    assert calls["count"] == 2
    assert result.ok is True
    assert result.diagnostics["attention_backend_retry"] == "sdpa"


def test_verify_transformers_structure_retries_kimi_tie_weights_after_attention_patch(monkeypatch):
    from model_structure_viewer.errors import IntrospectionError

    calls = {"count": 0}

    class FakeConfig:
        _attn_implementation = "flash_attention_2"

    def flaky_meta(config, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise IntrospectionError(
                "AutoModel.from_config failed: MoonViT3dPretrainedModel "
                "does not support Flash Attention 2 yet."
            )
        if calls["count"] == 2:
            raise IntrospectionError(
                "AutoModel.from_config failed: "
                "KimiK25ForConditionalGeneration.tie_weights() got an unexpected keyword argument "
                "'recompute_mapping'"
            )
        hf_config = FakeConfig()
        diagnostics = kwargs["config_normalizer"].normalize(hf_config)
        assert kwargs["runtime_patch"] is not None
        structure = _structure()
        structure.source["diagnostics"] = diagnostics
        return structure

    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        flaky_meta,
    )

    result = verify_transformers_structure(
        {"model_type": "kimi_k25", "architectures": ["KimiK25ForConditionalGeneration"]},
        source={"kind": "test", "model_id": "moonshotai/Kimi-K2.5"},
    )

    assert calls["count"] == 3
    assert result.ok is True
    assert result.summary["strategy"] == "tie-weights-compatible-transformers-meta"
    assert result.diagnostics["attention_backend_retry"] == "sdpa"
    assert result.diagnostics["runtime_patch"] == "kimi_tie_weights_compat"


def test_verify_response_uses_strict_worker_result(monkeypatch):
    def fake_verify_worker(config, **kwargs):
        return {
            "ok": False,
            "status": "failed",
            "strategy": "transformers-meta",
            "model_id": "Org/Demo",
            "source": {"kind": "uploaded config", "model_id": "Org/Demo"},
            "summary": {"model_type": "demo", "architecture": "DemoModel"},
            "diagnostics": {"failure_kind": "model_init_failed"},
            "error": "AutoModel.from_config failed: unsupported",
        }

    monkeypatch.setattr(service, "_run_transformers_verify_worker", fake_verify_worker)
    result = service.verify_structure_response(
        VerifyRequest(
            source="config",
            model_id="Org/Demo",
            config_json={"model_type": "demo", "architectures": ["DemoModel"]},
        ),
        AppSettings(offline=True),
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.strategy == "transformers-meta"
    assert result.diagnostics["failure_kind"] == "model_init_failed"


def test_verify_transformers_structure_carries_per_module_evidence(monkeypatch):
    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        lambda config, **kwargs: _graph_structure(),
    )

    result = verify_transformers_structure(
        {"model_type": "demo", "architectures": ["DemoModel"]},
        source={"kind": "test", "model_id": "Org/Demo"},
    )

    assert result.ok is True
    assert result.evidence is not None
    evidence = result.evidence
    # 整树 summary 语义不变，两态结论落在 evidence.summary
    assert evidence.summary == {"constructed": True, "structurally_consistent": None, "module_count": 6}
    assert evidence.diff.note == "msv_graph not provided"
    assert evidence.diff.only_transformers == []
    assert evidence.diff.only_msv == []
    assert evidence.diff.mismatches == []

    modules = {module["path"]: module for module in evidence.modules}
    q_proj = modules["root.model.layers.0.self_attn.q_proj"]
    assert q_proj == {
        "path": "root.model.layers.0.self_attn.q_proj",
        "class": "Linear",
        "params": 4096,
        "weight_shapes": {"weight": [64, 64]},
        "dtype": "BF16",
        "value_source": "introspect",
        "repeat": None,
        "source_ref": None,
    }
    # 形状类字段无值保持 None（不伪造数值），折叠节点保留 repeat
    norm = modules["root.model.norm"]
    assert norm["params"] is None
    assert norm["weight_shapes"] is None
    assert norm["dtype"] is None
    assert norm["value_source"] is None
    assert modules["root.model.layers.0"]["repeat"] == 4


def test_verify_transformers_structure_reconciliation_consistent(monkeypatch):
    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        lambda config, **kwargs: _graph_structure(),
    )
    msv_graph = {
        "nodes": [
            # 前端模板栈节点：无 class 标签（class 检查 None 不参与）
            {"id": "root.2", "canonical_id": "decoder", "type": "decoder"},
            {"id": "root.2.0", "canonical_id": "decoder.0.self_attn.q_proj", "type": "operator",
             "attributes": {"class": "Linear"}, "weight_shapes": {"weight": [64, 64]}},
            # 后缀容忍：RMSNorm ⊂ DemoRMSNorm
            {"id": "root.3", "canonical_id": "norm", "type": "normalization", "attributes": {"class": "RMSNorm"}},
        ]
    }

    result = verify_transformers_structure(
        {"model_type": "demo", "architectures": ["DemoModel"]},
        source={"kind": "test", "model_id": "Org/Demo"},
        msv_graph=msv_graph,
    )

    assert result.ok is True
    assert result.status == "passed"
    assert result.evidence.diff.note is None
    assert result.evidence.diff.only_transformers == []
    assert result.evidence.diff.only_msv == []
    assert result.evidence.diff.mismatches == []
    assert result.evidence.summary["structurally_consistent"] is True


def test_verify_transformers_structure_constructed_but_structurally_inconsistent(monkeypatch):
    """Task 7.6 合成用例：meta 构造通过但结构不一致——status 保持 passed，
    两态结论落在 evidence.summary.structurally_consistent=False。"""
    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        lambda config, **kwargs: _graph_structure(),
    )
    msv_graph = {
        "nodes": [
            # 前端多了后端没有的模块
            # P0-2：带声明且不命中 known_divergences（^lm_head 是 tied 专属登记），
            # 才能表达"构造通过但结构不一致"——unclassified 只剩未登记分歧。
            {"id": "root.4", "canonical_id": "decoder.custom_head", "type": "output", "attributes": {"class": "Linear", "weightMatrices": [{"class": "tp", "out": 1, "in": 1}]}},
            # path 命中但 weight_shapes 正维分歧
            {"id": "root.5", "canonical_id": "decoder.0.self_attn.q_proj", "type": "operator",
             "attributes": {"class": "Linear", "weightMatrices": [{"class": "tp", "out": 64, "in": 128}]},
             "weight_shapes": {"weight": [64, 128]}},
        ]
    }

    result = verify_transformers_structure(
        {"model_type": "demo", "architectures": ["DemoModel"]},
        source={"kind": "test", "model_id": "Org/Demo"},
        msv_graph=msv_graph,
    )

    assert result.ok is True
    assert result.status == "passed"
    assert result.evidence.diff.only_msv == ["decoder.custom_head"]
    assert [
        (entry.path, entry.kind, entry.transformers, entry.msv)
        for entry in result.evidence.diff.mismatches
    ] == [("decoder.self_attn.q_proj", "shape", {"weight": [64, 64]}, {"weight": [64, 128]})]
    assert result.evidence.summary["structurally_consistent"] is False


def test_verify_transformers_structure_failure_has_no_evidence(monkeypatch):
    def fail_meta(config, **kwargs):
        from model_structure_viewer.errors import IntrospectionError

        raise IntrospectionError("AutoModel.from_config failed: unsupported")

    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        fail_meta,
    )

    result = verify_transformers_structure(
        {"model_type": "demo", "architectures": ["DemoModel"]},
        source={"kind": "test", "model_id": "Org/Demo"},
        msv_graph={"nodes": [{"id": "root.0", "canonical_id": "norm"}]},
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.evidence is None


def test_verify_service_passes_msv_graph_through_to_verification(monkeypatch):
    """service → worker（直连模式）→ verify_transformers_structure 全链接线。"""
    monkeypatch.setenv("MSV_DISABLE_STRUCTURE_WORKER", "1")
    monkeypatch.setattr(
        "model_structure_viewer.structure.recovery.build_from_meta_model",
        lambda config, **kwargs: _graph_structure(),
    )
    msv_graph = {
        "nodes": [
            {"id": "root.0", "canonical_id": "decoder", "type": "decoder"},
            {"id": "root.1", "canonical_id": "decoder.0.self_attn.q_proj", "type": "operator",
             "attributes": {"class": "Linear"}, "weight_shapes": {"weight": [64, 64]}},
            {"id": "root.2", "canonical_id": "norm", "type": "normalization", "attributes": {"class": "RMSNorm"}},
        ]
    }

    result = service.verify_structure_response(
        VerifyRequest(
            source="config",
            model_id="Org/Demo",
            config_json={"model_type": "demo", "architectures": ["DemoModel"]},
            msv_graph=msv_graph,
        ),
        AppSettings(offline=True),
    )

    assert result.ok is True
    assert result.evidence.summary["structurally_consistent"] is True
    assert result.evidence.summary["module_count"] == 6
