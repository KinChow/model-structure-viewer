import json

from model_structure_viewer import cli
from model_structure_viewer.schemas import VerifyResponse


def test_cli_preserves_offline_environment_without_flag(monkeypatch):
    seen = []
    monkeypatch.setenv("MSV_OFFLINE", "1")
    monkeypatch.setattr(cli, "cmd_list", lambda args, settings: seen.append(settings) or 0)

    assert cli.main(["list"]) == 0
    assert seen[0].offline is True


def test_cli_offline_flag_enables_offline_mode(monkeypatch):
    seen = []
    monkeypatch.setenv("MSV_OFFLINE", "0")
    monkeypatch.setattr(cli, "cmd_list", lambda args, settings: seen.append(settings) or 0)

    assert cli.main(["--offline", "list"]) == 0
    assert seen[0].offline is True


def test_cli_verify_graph_flag_loads_msv_graph(tmp_path, monkeypatch):
    captured = {}

    def fake_verify(payload, settings):
        captured["payload"] = payload
        return VerifyResponse(ok=True, status="passed", strategy="transformers-meta", model_id=payload.model_id)

    monkeypatch.setattr(cli, "verify_structure_response", fake_verify)
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(
        json.dumps({"nodes": [{"id": "root.0", "canonical_id": "norm"}]}),
        encoding="utf-8",
    )
    config_file = tmp_path / "demo-config.json"
    config_file.write_text(json.dumps({"model_type": "demo"}), encoding="utf-8")

    code = cli.main(["verify", "--source", "config", "--config", str(config_file), "--graph", str(graph_file)])

    assert code == 0
    assert captured["payload"].msv_graph == {"nodes": [{"id": "root.0", "canonical_id": "norm"}]}


def test_cli_verify_without_graph_keeps_msv_graph_none(tmp_path, monkeypatch):
    captured = {}

    def fake_verify(payload, settings):
        captured["payload"] = payload
        return VerifyResponse(ok=False, status="failed", strategy="transformers-meta", model_id=payload.model_id)

    monkeypatch.setattr(cli, "verify_structure_response", fake_verify)
    config_file = tmp_path / "demo-config.json"
    config_file.write_text(json.dumps({"model_type": "demo"}), encoding="utf-8")

    code = cli.main(["verify", "--source", "config", "--config", str(config_file)])

    assert code == 1
    assert captured["payload"].msv_graph is None
