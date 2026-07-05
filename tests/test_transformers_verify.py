from model_structure_viewer import service
from model_structure_viewer.schemas import ModelStructure, StructureNode, VerifyRequest
from model_structure_viewer.settings import AppSettings
from model_structure_viewer.verification.transformers_verify import verify_transformers_structure


def _structure():
    return ModelStructure(
        summary={"strategy": "meta-introspect", "backbone_class": "DemoModel"},
        source={"strategy": "meta-introspect", "backbone_class": "DemoModel"},
        root=StructureNode(id="root", name="DemoModel", type="module"),
    )


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
