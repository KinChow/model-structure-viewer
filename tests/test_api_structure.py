"""Route-level tests for the /api/structure endpoint."""
import json

from fastapi.testclient import TestClient

from model_structure_viewer.api import app, get_settings, set_settings
from model_structure_viewer.settings import AppSettings


client = TestClient(app)


def test_structure_api_returns_contract_for_inline_config():
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

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"summary", "source", "root", "extra_config"}
    assert payload["summary"]["strategy"] == "config-fallback"
    assert payload["source"]["strategy"] == payload["summary"]["strategy"]
    assert payload["root"]["children"]


def test_structure_api_fallback_keeps_decoder_only_config_structural():
    config = {
        "model_type": "deepseek_v3",
        "architectures": ["DeepseekV3ForCausalLM"],
        "auto_map": {"AutoModel": "modeling_deepseek.DeepseekV3Model"},
        "num_hidden_layers": 4,
        "hidden_size": 128,
        "num_attention_heads": 8,
        "n_routed_experts": 16,
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
    assert payload["summary"]["strategy"] == "config-fallback"

    children = payload["root"]["children"]
    assert children
    assert [child["name"] for child in children] != ["Configuration"]

    structural = next(child for child in children if child["type"] in {"decoder", "layers"})
    assert structural["confidence"] == "low"
    assert structural["repeat"] == 4 or structural["attributes"].get("num_hidden_layers") == 4
    assert payload["source"]["diagnostics"]["failure_kind"] == "remote_code_unavailable"


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
    config_path = tmp_path / "kimi" / "Kimi-K2-Instruct-config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "model_type": "kimi_k2",
                "architectures": ["DeepseekV3ForCausalLM"],
                "num_hidden_layers": 61,
                "hidden_size": 7168,
                "num_attention_heads": 64,
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
    assert payload["summary"]["model_type"] == "kimi_k2"
    assert payload["summary"]["text_layers"] == 61
