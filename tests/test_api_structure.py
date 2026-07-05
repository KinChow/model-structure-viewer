"""Route-level tests for the /api/structure endpoint."""
import json

from fastapi.testclient import TestClient

from model_structure_viewer import service
from model_structure_viewer.api import app, get_settings, set_settings
from model_structure_viewer.settings import AppSettings


client = TestClient(app)


def test_structure_api_returns_contract_for_inline_config():
    service.clear_structure_cache()
    config = {
        "model_type": "bert",
        "architectures": ["BertModel"],
        "num_hidden_layers": 1,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 128,
    }

    response = client.post(
        "/api/structure",
        json={
            "source": "config",
            "config_json": config,
            "detail_level": "compressed",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"summary", "source", "root", "extra_config"}
    assert payload["summary"]["strategy"] in {"meta-introspect", "repaired-meta-introspect"}
    assert payload["root"]["children"]


def test_structure_api_returns_error_when_transformers_introspection_fails():
    service.clear_structure_cache()
    config = {
        "model_type": "tiny_decoder",
        "architectures": ["TinyDecoderForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
    }

    response = client.post(
        "/api/structure",
        json={
            "source": "config",
            "config_json": config,
            "detail_level": "compressed",
        },
    )

    assert response.status_code == 500
    payload = response.json()
    assert "AutoConfig.for_model failed" in payload["detail"]


def test_models_api_lists_standalone_configs_and_excludes_inference_configs(tmp_path):
    original_settings = get_settings()
    try:
        model_dir = tmp_path / "Org" / "Model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "standard", "num_hidden_layers": 2, "hidden_size": 64}),
            encoding="utf-8",
        )
        kimi_dir = tmp_path / "kimi"
        kimi_dir.mkdir()
        standalone_path = kimi_dir / "Kimi-K2-Instruct-config.json"
        standalone_path.write_text(
            json.dumps({"model_type": "kimi_k2", "num_hidden_layers": 61, "hidden_size": 7168}),
            encoding="utf-8",
        )
        inference_dir = tmp_path / "Org" / "Model" / "inference"
        inference_dir.mkdir()
        (inference_dir / "config.json").write_text(
            json.dumps({"n_layers": 61, "dim": 7168, "n_heads": 128}),
            encoding="utf-8",
        )
        (kimi_dir / "tokenizer_config.json").write_text(
            json.dumps({"model_type": "tokenizer"}), encoding="utf-8"
        )

        set_settings(AppSettings(model_root=tmp_path, offline=True))
        response = client.get("/api/models")
    finally:
        set_settings(original_settings)

    assert response.status_code == 200
    entries = {entry["model_id"]: entry for entry in response.json()}
    assert set(entries) == {"Org/Model", "kimi/Kimi-K2-Instruct-config"}
    assert entries["Org/Model"]["load_by"] == "model_id"
    assert entries["kimi/Kimi-K2-Instruct-config"]["load_by"] == "config_path"
    assert entries["kimi/Kimi-K2-Instruct-config"]["config_path"] == str(standalone_path)


def test_structure_api_accepts_config_path_for_standalone_config(tmp_path):
    service.clear_structure_cache()
    config_path = tmp_path / "bert" / "tiny-bert-config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertModel"],
                "num_hidden_layers": 1,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )

    response = client.post(
        "/api/structure",
        json={"source": "local", "config_path": str(config_path), "detail_level": "compressed"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"]["kind"] == "local file"
    assert payload["summary"]["model_type"] == "bert"
    assert payload["summary"]["text_layers"] == 1


def test_local_config_api_returns_config_without_building_structure(tmp_path, monkeypatch):
    original_settings = get_settings()
    try:
        model_dir = tmp_path / "Qwen" / "Qwen3.5-0.8B"
        model_dir.mkdir(parents=True)
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 1024,
            "num_attention_heads": 16,
        }
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        def fail_build(*args, **kwargs):
            raise AssertionError("structure builder must not run")

        monkeypatch.setattr(service, "build_structure_response", fail_build, raising=False)
        set_settings(AppSettings(model_root=tmp_path, offline=True))
        response = client.get("/api/local/config?model_id=Qwen/Qwen3.5-0.8B")
    finally:
        set_settings(original_settings)

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert payload["config"] == config
    assert payload["source"]["kind"] == "local cache"


def test_local_config_api_does_not_fetch_remote_code(tmp_path, monkeypatch):
    original_settings = get_settings()
    try:
        model_dir = tmp_path / "deepseek-ai" / "DeepSeek-V3.1"
        model_dir.mkdir(parents=True)
        config = {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "auto_map": {"AutoModel": "modeling_deepseek.DeepseekV3Model"},
            "num_hidden_layers": 2,
            "hidden_size": 1024,
            "num_attention_heads": 16,
        }
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        def fail_remote_fetch(*args, **kwargs):
            raise AssertionError("local config endpoint must not fetch remote code")

        monkeypatch.setattr(
            "model_structure_viewer.resolve.remote_code.RemoteCodeFetcher.ensure_remote_code",
            fail_remote_fetch,
        )
        set_settings(AppSettings(model_root=tmp_path, offline=False, auto_fetch_remote_code=True))
        response = client.get("/api/local/config?model_id=deepseek-ai/DeepSeek-V3.1")
    finally:
        set_settings(original_settings)

    assert response.status_code == 200
    assert response.json()["config"] == config


def test_structure_api_caches_repeated_responses(monkeypatch):
    service.clear_structure_cache()
    calls = {"count": 0}

    def fake_worker(config, **kwargs):
        calls["count"] += 1
        return {
            "ok": True,
            "structure": service.build_model_structure(
                config,
                source=kwargs["source"],
                detail_level=kwargs["detail_level"],
                local_dir=kwargs["local_dir"],
            ).model_dump(mode="json"),
        }

    monkeypatch.setattr(service, "_run_introspection_worker", fake_worker)
    config = {
        "model_type": "bert",
        "architectures": ["BertModel"],
        "num_hidden_layers": 1,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 128,
    }
    request = {
        "source": "config",
        "config_json": config,
        "detail_level": "compressed",
    }

    first = client.post("/api/structure", json=request)
    second = client.post("/api/structure", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert calls["count"] == 1


def test_verify_api_returns_transformers_validation(monkeypatch):
    def fake_verify(payload, settings):
        from model_structure_viewer.schemas import VerifyResponse

        return VerifyResponse(
            ok=True,
            status="passed",
            strategy="transformers-meta",
            model_id=payload.model_id,
            source={"kind": "test"},
            summary={"strategy": "meta-introspect", "backbone_class": "DemoModel"},
            diagnostics={"backbone_class": "DemoModel"},
        )

    monkeypatch.setattr("model_structure_viewer.api.verify_structure_response", fake_verify)

    response = client.post(
        "/api/verify",
        json={
            "source": "config",
            "model_id": "Org/Demo",
            "config_json": {"model_type": "demo", "architectures": ["DemoModel"]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status"] == "passed"
    assert payload["strategy"] == "transformers-meta"
