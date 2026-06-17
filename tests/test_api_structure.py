"""Route-level tests for the /api/structure endpoint."""
from fastapi.testclient import TestClient

from model_structure_viewer.api import app


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
