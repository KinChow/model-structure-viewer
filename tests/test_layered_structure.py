from model_structure_viewer import service
from model_structure_viewer.schemas import ModelStructure, StructureNode, StructureRequest
from model_structure_viewer.settings import AppSettings


def _request(config: dict) -> StructureRequest:
    return StructureRequest(
        source="config",
        config_json=config,
        detail_level="compressed",
    )


def test_high_risk_config_skips_introspection_by_generic_budget(monkeypatch):
    service.clear_structure_cache()
    calls = {"count": 0}

    def fake_worker(*args, **kwargs):
        calls["count"] += 1
        return {
            "ok": True,
            "structure": ModelStructure(
                summary={"strategy": "meta-introspect"},
                source={"strategy": "meta-introspect"},
                root=StructureNode(id="root", name="Root", type="model"),
            ).model_dump(mode="json"),
        }

    monkeypatch.setattr(service, "_run_introspection_worker", fake_worker, raising=False)

    structure = service.build_structure_response(
        _request(
            {
                "model_type": "large_moe_candidate",
                "architectures": ["HugeMoEForCausalLM"],
                "num_hidden_layers": 80,
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "n_routed_experts": 128,
            }
        ),
        AppSettings(offline=True),
    )

    assert calls["count"] == 0
    assert structure.summary["strategy"] == "budget-config-fallback"
    diagnostics = structure.source["diagnostics"]
    assert diagnostics["failure_kind"] == "resource_budget_exceeded"
    assert diagnostics["budget"]["layers"] == 80
    assert diagnostics["budget"]["hidden_size"] == 8192
    assert diagnostics["budget"]["experts"] == 128


def test_worker_failure_returns_config_fallback_with_diagnostics(monkeypatch):
    service.clear_structure_cache()

    def fake_worker(*args, **kwargs):
        return {
            "ok": False,
            "failure_kind": "worker_killed",
            "message": "worker exited with code -9",
            "exit_code": -9,
        }

    monkeypatch.setattr(service, "_run_introspection_worker", fake_worker, raising=False)

    structure = service.build_structure_response(
        _request(
            {
                "model_type": "bert",
                "architectures": ["BertModel"],
                "num_hidden_layers": 2,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "intermediate_size": 64,
            }
        ),
        AppSettings(offline=True),
    )

    assert structure.summary["strategy"] == "worker-config-fallback"
    assert structure.summary["fallback_reason"] == "worker exited with code -9"
    assert structure.source["diagnostics"]["failure_kind"] == "worker_killed"
    assert structure.source["diagnostics"]["worker_exit_code"] == -9
