import json
from pathlib import Path

import pytest

from model_structure_viewer.errors import RemoteError
from model_structure_viewer.resolve.hf_client import HuggingFaceClient
from model_structure_viewer.resolve.remote_code import RemoteCodeFetcher
from model_structure_viewer.resolver import ModelSourceResolver, SourceResolutionError
from model_structure_viewer.settings import AppSettings

FIXTURE = Path(__file__).parent / "fixtures" / "minimax_m3" / "config.json"
DEEPSEEK_FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v3" / "config.json"


def test_list_and_resolve_local_model(tmp_path):
    model_dir = tmp_path / "MiniMaxAI" / "MiniMax-M3"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    (model_dir / "README.md").write_text("# MiniMax-M3\n", encoding="utf-8")

    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    entries = resolver.list_local_models()
    assert [entry.model_id for entry in entries] == ["MiniMaxAI/MiniMax-M3"]

    resolved = resolver.resolve(source="local", model_id="MiniMaxAI/MiniMax-M3")
    assert resolved.config["model_type"] == "minimax_m3_vl"
    assert resolved.source["kind"] == "local cache"


def test_list_local_models_includes_standalone_model_json_and_skips_non_model_configs(tmp_path):
    standard_dir = tmp_path / "MiniMaxAI" / "MiniMax-M3"
    standard_dir.mkdir(parents=True)
    (standard_dir / "config.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    standalone_dir = tmp_path / "kimi"
    standalone_dir.mkdir()
    kimi_config = {
        "model_type": "kimi_k2",
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_hidden_layers": 61,
        "hidden_size": 7168,
        "num_attention_heads": 64,
    }
    standalone_path = standalone_dir / "Kimi-K2-Instruct-config.json"
    standalone_path.write_text(json.dumps(kimi_config), encoding="utf-8")

    (standalone_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_type": "not_a_model_structure"}), encoding="utf-8"
    )
    (standalone_dir / "generation_config.json").write_text(
        json.dumps({"architectures": ["Ignored"]}), encoding="utf-8"
    )
    (standalone_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}}), encoding="utf-8"
    )
    inference_dir = tmp_path / "deepseek-ai" / "DeepSeek-V4-Pro" / "inference"
    inference_dir.mkdir(parents=True)
    (inference_dir / "config.json").write_text(
        json.dumps({"n_layers": 61, "dim": 7168, "n_heads": 128}), encoding="utf-8"
    )
    hidden_dir = tmp_path / ".venv" / "FakeModel"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "config.json").write_text(json.dumps(kimi_config), encoding="utf-8")

    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    entries = {entry.model_id: entry for entry in resolver.list_local_models()}

    assert set(entries) == {"MiniMaxAI/MiniMax-M3", "kimi/Kimi-K2-Instruct-config"}
    assert entries["MiniMaxAI/MiniMax-M3"].load_by == "model_id"
    assert entries["kimi/Kimi-K2-Instruct-config"].load_by == "config_path"
    assert entries["kimi/Kimi-K2-Instruct-config"].config_path == str(standalone_path)


def test_resolve_config_path_supports_standalone_model_json(tmp_path):
    config_path = tmp_path / "kimi" / "Kimi-K2-Instruct-config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps({"model_type": "kimi_k2", "num_hidden_layers": 61, "hidden_size": 7168}),
        encoding="utf-8",
    )

    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    resolved = resolver.resolve(source="local", config_path=str(config_path), detail_level="compressed")

    assert resolved.config["model_type"] == "kimi_k2"
    assert resolved.source["kind"] == "local file"
    assert resolved.local_dir == config_path.parent


def test_local_model_cache_accepts_string_model_root(tmp_path):
    model_dir = tmp_path / "Org" / "Model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "tiny", "num_hidden_layers": 1, "hidden_size": 8}),
        encoding="utf-8",
    )

    resolver = ModelSourceResolver(AppSettings(model_root=str(tmp_path), offline=True))

    assert [entry.model_id for entry in resolver.list_local_models()] == ["Org/Model"]


def test_auto_offline_fails_when_local_missing(tmp_path):
    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    with pytest.raises(SourceResolutionError):
        resolver.resolve(source="auto", model_id="MiniMaxAI/MiniMax-M3", cache_policy="offline")


def test_config_source_uses_supplied_json():
    config = json.loads(FIXTURE.read_text(encoding="utf-8"))
    resolver = ModelSourceResolver(AppSettings())
    resolved = resolver.resolve(source="config", config_json=config)
    assert resolved.config["model_type"] == "minimax_m3_vl"
    assert resolved.source["kind"] == "uploaded config"


def _stub_resolver(tmp_path: Path, *, offline: bool = False, auto_fetch: bool = True) -> ModelSourceResolver:
    settings = AppSettings(
        model_root=tmp_path,
        offline=offline,
        auto_fetch_remote_code=auto_fetch,
    )
    return ModelSourceResolver(settings)


def _seed_local(tmp_path: Path, model_id: str, config_text: str) -> Path:
    parts = model_id.split("/")
    target = tmp_path.joinpath(*parts)
    target.mkdir(parents=True)
    (target / "config.json").write_text(config_text, encoding="utf-8")
    return target


def test_ensure_remote_code_downloads_auto_map_modules(tmp_path, monkeypatch):
    model_id = "deepseek-ai/DeepSeek-V3"
    local_dir = _seed_local(tmp_path, model_id, DEEPSEEK_FIXTURE.read_text(encoding="utf-8"))
    resolver = _stub_resolver(tmp_path)

    tree = [
        {"path": "config.json"},
        {"path": "modeling_deepseek.py"},
        {"path": "configuration_deepseek.py"},
        {"path": "weights/model.safetensors"},
        {"path": "model-00001.safetensors"},
        {"path": "tokenizer.json"},
    ]
    downloads: list[str] = []

    def fake_tree(self, mid, rev):
        assert mid == model_id
        return tree

    def fake_download_text(self, mid, filename, rev):
        downloads.append(filename)
        return f"# fake {filename}\n"

    monkeypatch.setattr(HuggingFaceClient, "list_tree", fake_tree, raising=True)
    monkeypatch.setattr(HuggingFaceClient, "download_text", fake_download_text, raising=True)

    resolved = resolver.resolve(source="local", model_id=model_id)

    assert (local_dir / "modeling_deepseek.py").exists()
    assert (local_dir / "configuration_deepseek.py").exists()
    assert not (local_dir / "model-00001.safetensors").exists()
    assert "weights/model.safetensors" not in downloads
    assert "tokenizer.json" not in downloads
    info = resolved.source.get("remote_code_fetch")
    assert info is not None
    assert sorted(info["fetched"]) == ["configuration_deepseek.py", "modeling_deepseek.py"]
    assert info["errors"] == []


def test_ensure_remote_code_skips_existing_files(tmp_path, monkeypatch):
    model_id = "deepseek-ai/DeepSeek-V3"
    local_dir = _seed_local(tmp_path, model_id, DEEPSEEK_FIXTURE.read_text(encoding="utf-8"))
    (local_dir / "modeling_deepseek.py").write_text("# already here\n", encoding="utf-8")
    resolver = _stub_resolver(tmp_path)

    tree = [
        {"path": "config.json"},
        {"path": "modeling_deepseek.py"},
        {"path": "configuration_deepseek.py"},
    ]
    downloads: list[str] = []

    monkeypatch.setattr(HuggingFaceClient, "list_tree", lambda self, mid, rev: tree)

    def fake_download_text(self, mid, filename, rev):
        downloads.append(filename)
        return f"# fetched {filename}"

    monkeypatch.setattr(HuggingFaceClient, "download_text", fake_download_text)

    resolver.resolve(source="local", model_id=model_id)
    assert downloads == ["configuration_deepseek.py"]
    assert (local_dir / "modeling_deepseek.py").read_text(encoding="utf-8") == "# already here\n"


def test_ensure_remote_code_records_errors_and_continues(tmp_path, monkeypatch):
    model_id = "deepseek-ai/DeepSeek-V3"
    local_dir = _seed_local(tmp_path, model_id, DEEPSEEK_FIXTURE.read_text(encoding="utf-8"))
    resolver = _stub_resolver(tmp_path)

    monkeypatch.setattr(
        HuggingFaceClient,
        "list_tree",
        lambda self, mid, rev: [
            {"path": "modeling_deepseek.py"},
            {"path": "configuration_deepseek.py"},
        ],
    )

    def fake_download_text(self, mid, filename, rev):
        if filename == "modeling_deepseek.py":
            raise RemoteError("HTTP 404")
        return "# ok"

    monkeypatch.setattr(HuggingFaceClient, "download_text", fake_download_text)

    resolved = resolver.resolve(source="local", model_id=model_id)
    info = resolved.source["remote_code_fetch"]
    assert info["fetched"] == ["configuration_deepseek.py"]
    assert info["errors"] == [{"file": "modeling_deepseek.py", "reason": "HTTP 404"}]
    assert (local_dir / "configuration_deepseek.py").exists()
    assert not (local_dir / "modeling_deepseek.py").exists()


def test_ensure_remote_code_disabled_when_flag_false(tmp_path, monkeypatch):
    model_id = "deepseek-ai/DeepSeek-V3"
    _seed_local(tmp_path, model_id, DEEPSEEK_FIXTURE.read_text(encoding="utf-8"))
    resolver = _stub_resolver(tmp_path, auto_fetch=False)

    def boom(*a, **kw):
        raise AssertionError("network must not be touched")

    monkeypatch.setattr(HuggingFaceClient, "list_tree", boom)
    monkeypatch.setattr(HuggingFaceClient, "download_text", boom)

    resolved = resolver.resolve(source="local", model_id=model_id)
    assert "remote_code_fetch" not in resolved.source


def test_ensure_remote_code_disabled_in_offline(tmp_path, monkeypatch):
    model_id = "deepseek-ai/DeepSeek-V3"
    _seed_local(tmp_path, model_id, DEEPSEEK_FIXTURE.read_text(encoding="utf-8"))
    resolver = _stub_resolver(tmp_path, offline=True)

    def boom(*a, **kw):
        raise AssertionError("offline must not call HF")

    monkeypatch.setattr(HuggingFaceClient, "list_tree", boom)
    monkeypatch.setattr(HuggingFaceClient, "download_text", boom)

    resolved = resolver.resolve(source="local", model_id=model_id)
    assert "remote_code_fetch" not in resolved.source


def test_source_config_does_not_trigger_remote_fetch(monkeypatch, tmp_path):
    config = json.loads(DEEPSEEK_FIXTURE.read_text(encoding="utf-8"))
    resolver = _stub_resolver(tmp_path)

    def boom(*a, **kw):
        raise AssertionError("source=config must not fetch")

    monkeypatch.setattr(HuggingFaceClient, "list_tree", boom)
    monkeypatch.setattr(HuggingFaceClient, "download_text", boom)

    resolved = resolver.resolve(source="config", config_json=config)
    assert resolved.local_dir is None
    assert "remote_code_fetch" not in resolved.source


def test_auto_map_modules_handles_list_and_invalid():
    config = {
        "auto_map": {
            "AutoConfig": "configuration_x.XConfig",
            "AutoModel": ["modeling_x.XModel", "modeling_extra.XModelHelper"],
            "Bad": 123,
            "Empty": "",
        }
    }
    assert RemoteCodeFetcher._auto_map_modules(config) == [
        "configuration_x",
        "modeling_extra",
        "modeling_x",
    ]
