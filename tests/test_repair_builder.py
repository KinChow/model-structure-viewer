from model_structure_viewer.errors import IntrospectionError
from model_structure_viewer.schemas import ModelStructure, StructureNode
from model_structure_viewer.structure import builder


CONFIG = {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]}


def _structure(strategy):
    return ModelStructure(
        summary={"strategy": strategy},
        source={"strategy": strategy},
        root=StructureNode(id="root", name="Root", type="model"),
    )


def test_builder_retries_once_after_matching_repair(monkeypatch):
    calls = []

    def fake_meta(
        config,
        *,
        source,
        local_dir,
        config_overrides=None,
        runtime_patch=None,
        config_normalizer=None,
    ):
        calls.append((config, source, config_overrides, runtime_patch, config_normalizer))
        if len(calls) == 1:
            raise IntrospectionError(
                "AutoModel.from_config failed: cannot import name 'is_torch_fx_available'"
            )
        return _structure("meta-introspect")

    monkeypatch.setattr(builder, "build_from_meta_model", fake_meta)

    result = builder.build_model_structure(CONFIG, source={}, local_dir=None)

    assert len(calls) == 2
    assert calls[1][2] == {}
    assert calls[1][3] is not None
    assert calls[1][3].name == "deepseek_torch_fx_compat"
    assert calls[1][4] is None
    assert result.summary["strategy"] == "repaired-meta-introspect"
    assert result.source["strategy"] == "repaired-meta-introspect"
    assert result.source["diagnostics"]["failure_kind"] == "remote_import_compat"
    assert result.source["diagnostics"]["repair_strategy"] == "deepseek_import_compat"
    assert result.source["diagnostics"]["retry_count"] == 1
    assert result.source["diagnostics"]["repair_status"] == "success"


def test_builder_raises_when_no_repair_strategy(monkeypatch):
    def fake_meta(
        config,
        *,
        source,
        local_dir,
        config_overrides=None,
        runtime_patch=None,
        config_normalizer=None,
    ):
        raise IntrospectionError("something unexpected happened")

    monkeypatch.setattr(builder, "build_from_meta_model", fake_meta)

    import pytest

    with pytest.raises(IntrospectionError, match="something unexpected happened"):
        builder.build_model_structure(
            {"model_type": "unknown", "num_hidden_layers": 2},
            source={},
            local_dir=None,
        )


def test_builder_passes_minimax_normalizer_to_repair_retry(monkeypatch):
    calls = []
    config = {
        "model_type": "minimax_m3",
        "architectures": ["MiniMaxM3ForCausalLM"],
        "img_token_compression_config": {"temporal_patch_size": 4},
    }

    def fake_meta(
        config,
        *,
        source,
        local_dir,
        config_overrides=None,
        runtime_patch=None,
        config_normalizer=None,
    ):
        calls.append((config, source, config_overrides, runtime_patch, config_normalizer))
        if len(calls) == 1:
            raise IntrospectionError("AutoModel.from_config failed: object has no attribute 'temporal_patch_size'")
        return _structure("meta-introspect")

    monkeypatch.setattr(builder, "build_from_meta_model", fake_meta)

    result = builder.build_model_structure(config, source={}, local_dir=None)

    assert len(calls) == 2
    assert calls[1][2] == {"temporal_patch_size": 4}
    assert calls[1][3] is None
    assert calls[1][4] is not None
    assert calls[1][4].name == "minimax_config_normalizer"
    assert result.summary["strategy"] == "repaired-meta-introspect"
    assert result.source["diagnostics"]["config_overrides"] == ["temporal_patch_size"]
    assert result.source["diagnostics"]["config_normalizer"] == "minimax_config_normalizer"


def test_builder_reports_minimax_normalizer_diagnostics_after_retry_failure(monkeypatch):
    calls = []
    config = {
        "model_type": "minimax_m3",
        "architectures": ["MiniMaxM3ForCausalLM"],
        "img_token_compression_config": {"temporal_patch_size": 4},
    }

    def fake_meta(
        config,
        *,
        source,
        local_dir,
        config_overrides=None,
        runtime_patch=None,
        config_normalizer=None,
    ):
        calls.append((source, config_overrides, runtime_patch, config_normalizer))
        if len(calls) == 1:
            raise IntrospectionError("AutoModel.from_config failed: object has no attribute 'temporal_patch_size'")
        raise IntrospectionError("AutoModel.from_config failed: object has no attribute 'another_field'")

    monkeypatch.setattr(builder, "build_from_meta_model", fake_meta)

    import pytest

    with pytest.raises(IntrospectionError) as exc_info:
        builder.build_model_structure(config, source={}, local_dir=None)

    assert len(calls) == 2
    assert calls[1][1] == {"temporal_patch_size": 4}
    assert calls[1][2] is None
    assert calls[1][3] is not None
    assert calls[1][3].name == "minimax_config_normalizer"
    message = str(exc_info.value)
    assert "repair_strategy=minimax_config_adapter" in message
    assert "repair_status=failed" in message
    assert "config_normalizer=minimax_config_normalizer" in message


def test_builder_raises_after_repair_retry_failure(monkeypatch):
    calls = []

    def fake_meta(
        config,
        *,
        source,
        local_dir,
        config_overrides=None,
        runtime_patch=None,
        config_normalizer=None,
    ):
        calls.append((source, config_overrides, runtime_patch, config_normalizer))
        raise IntrospectionError(
            "AutoModel.from_config failed: cannot import name 'is_torch_fx_available'"
        )

    monkeypatch.setattr(builder, "build_from_meta_model", fake_meta)

    import pytest

    with pytest.raises(IntrospectionError) as exc_info:
        builder.build_model_structure(CONFIG, source={}, local_dir=None)

    assert len(calls) == 2
    assert calls[1][1] == {}
    assert calls[1][2] is not None
    assert calls[1][2].name == "deepseek_torch_fx_compat"
    assert calls[1][3] is None
    message = str(exc_info.value)
    assert "repair_strategy=deepseek_import_compat" in message
    assert "repair_status=failed" in message
    assert "retry_count=1" in message


def test_builder_retries_attention_and_kimi_tie_weights_compat(monkeypatch):
    calls = []

    def fake_meta(
        config,
        *,
        source,
        local_dir,
        config_overrides=None,
        runtime_patch=None,
        config_normalizer=None,
    ):
        calls.append((source, runtime_patch, config_normalizer))
        if len(calls) == 1:
            raise IntrospectionError(
                "AutoModel.from_config failed: MoonViT3dPretrainedModel does not support Flash Attention 2 yet."
            )
        if len(calls) == 2:
            raise IntrospectionError(
                "AutoModel.from_config failed: "
                "KimiK25ForConditionalGeneration.tie_weights() got an unexpected keyword argument "
                "'recompute_mapping'"
            )
        return _structure("meta-introspect")

    monkeypatch.setattr(builder, "build_from_meta_model", fake_meta)

    result = builder.build_model_structure(
        {"model_type": "kimi_k25", "architectures": ["KimiK25ForConditionalGeneration"]},
        source={"model_id": "moonshotai/Kimi-K2.5"},
        local_dir=None,
    )

    assert len(calls) == 3
    assert calls[1][2] is not None
    assert calls[1][2].name == "attention_implementation_normalizer"
    assert calls[2][1] is not None
    assert calls[2][1].name == "composite_runtime_patch"
    assert result.summary["strategy"] == "repaired-meta-introspect"
    assert result.source["diagnostics"]["attention_backend_retry"] == "sdpa"
    assert result.source["diagnostics"]["runtime_patch"] == "kimi_tie_weights_compat"
    assert result.source["diagnostics"]["retry_status"] == "success"


def test_builder_does_not_short_circuit_glm_moe_dsa(monkeypatch):
    calls = []

    def fake_meta(
        config,
        *,
        source,
        local_dir,
        config_overrides=None,
        runtime_patch=None,
        config_normalizer=None,
    ):
        calls.append((config, source, local_dir))
        return _structure("meta-introspect")

    monkeypatch.setattr(builder, "build_from_meta_model", fake_meta)

    result = builder.build_model_structure(
        {
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "text_config": {
                "num_hidden_layers": 78,
                "hidden_size": 6144,
                "num_attention_heads": 64,
            },
        },
        source={"kind": "test"},
    )

    assert len(calls) == 1
    assert result.summary["strategy"] == "meta-introspect"
