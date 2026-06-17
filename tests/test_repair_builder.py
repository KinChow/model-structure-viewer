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


def test_builder_falls_back_when_no_repair_strategy(monkeypatch):
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

    result = builder.build_model_structure(
        {"model_type": "unknown", "num_hidden_layers": 2},
        source={},
        local_dir=None,
    )

    assert result.summary["strategy"] == "config-fallback"
    assert result.source["strategy"] == "config-fallback"
    assert result.source["diagnostics"]["failure_kind"] == "unknown"
    assert result.source["diagnostics"]["repair_status"] == "not_attempted"


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


def test_builder_preserves_minimax_normalizer_diagnostics_after_retry_failure(monkeypatch):
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

    result = builder.build_model_structure(config, source={}, local_dir=None)

    assert len(calls) == 2
    assert calls[1][1] == {"temporal_patch_size": 4}
    assert calls[1][2] is None
    assert calls[1][3] is not None
    assert calls[1][3].name == "minimax_config_normalizer"
    assert result.summary["strategy"] == "config-fallback"
    assert result.source["diagnostics"]["repair_strategy"] == "minimax_config_adapter"
    assert result.source["diagnostics"]["repair_status"] == "failed"
    assert result.source["diagnostics"]["config_normalizer"] == "minimax_config_normalizer"


def test_builder_falls_back_after_repair_retry_failure(monkeypatch):
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

    result = builder.build_model_structure(CONFIG, source={}, local_dir=None)

    assert len(calls) == 2
    assert calls[1][1] == {}
    assert calls[1][2] is not None
    assert calls[1][2].name == "deepseek_torch_fx_compat"
    assert calls[1][3] is None
    assert result.summary["strategy"] == "config-fallback"
    assert result.source["diagnostics"]["repair_strategy"] == "deepseek_import_compat"
    assert result.source["diagnostics"]["repair_status"] == "failed"
    assert result.source["diagnostics"]["retry_count"] == 1
