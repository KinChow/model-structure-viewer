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


def test_resolve_config_path_accepts_model_directory(tmp_path):
    model_dir = tmp_path / "deepseek-ai" / "DeepSeek-V3.1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v3", "num_hidden_layers": 2}),
        encoding="utf-8",
    )

    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    resolved = resolver.resolve(source="local", config_path=str(model_dir), detail_level="compressed")

    assert resolved.config["model_type"] == "deepseek_v3"
    assert resolved.source["kind"] == "local directory"
    assert resolved.source["config_path"] == str(model_dir / "config.json")
    assert resolved.local_dir == model_dir


def test_local_model_cache_accepts_string_model_root(tmp_path):
    model_dir = tmp_path / "Org" / "Model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "tiny", "num_hidden_layers": 1, "hidden_size": 8}),
        encoding="utf-8",
    )

    resolver = ModelSourceResolver(AppSettings(model_root=str(tmp_path), offline=True))

    assert [entry.model_id for entry in resolver.list_local_models()] == ["Org/Model"]


def test_model_id_cannot_escape_model_root(tmp_path):
    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    with pytest.raises(SourceResolutionError, match="path traversal"):
        resolver.local_config_path("../outside")


def test_modelscope_client_uses_models_prefix():
    client = HuggingFaceClient("https://www.modelscope.cn")
    assert client._resolve_url("Org/Model", "config.json", "master") == "https://www.modelscope.cn/models/Org/Model/resolve/master/config.json"


def test_hf_client_resolves_huggingface_revision(monkeypatch):
    client = HuggingFaceClient("https://huggingface.co")
    monkeypatch.setattr(client, "_http_json", lambda url, log_errors=False: {"sha": "commit-a"})

    assert client.resolve_revision("Org/Model", "main") == "commit-a"


def test_hf_client_resolves_modelscope_config_revision(monkeypatch):
    client = HuggingFaceClient("https://www.modelscope.cn")
    monkeypatch.setattr(
        client,
        "_http_json",
        lambda url, log_errors=False: {
            "Data": {"Files": [{"Path": "config.json", "Revision": "commit-b"}]},
        },
    )

    assert client.resolve_revision("Org/Model", "master") == "commit-b"


def test_auto_offline_fails_when_local_missing(tmp_path):
    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    with pytest.raises(SourceResolutionError):
        resolver.resolve(source="auto", model_id="MissingOrg/MissingModel", cache_policy="offline")


def test_builtin_source_reads_repo_models_without_model_root(tmp_path):
    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    resolved = resolver.resolve(source="builtin", model_id="Qwen/Qwen3.5-0.8B")
    assert resolved.config["model_type"]
    assert resolved.source["kind"] == "built-in config"
    assert resolved.local_dir is not None
    assert str(resolved.local_dir).endswith("models/Qwen/Qwen3.5-0.8B")


def test_auto_prefers_builtin_before_local_model_root(tmp_path):
    local_dir = tmp_path / "Qwen" / "Qwen3.5-0.8B"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text(
        json.dumps({"model_type": "local_shadow", "num_hidden_layers": 1, "hidden_size": 8}),
        encoding="utf-8",
    )
    resolver = ModelSourceResolver(AppSettings(model_root=tmp_path, offline=True))
    resolved = resolver.resolve(source="auto", model_id="Qwen/Qwen3.5-0.8B")
    assert resolved.source["kind"] == "built-in config"
    assert resolved.config["model_type"] != "local_shadow"


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


def test_hf_source_does_not_download_remote_code_when_disabled(tmp_path, monkeypatch):
    resolver = _stub_resolver(tmp_path, auto_fetch=False)
    monkeypatch.setattr(
        HuggingFaceClient,
        "download_json",
        lambda self, model_id, filename, revision: {
            "model_type": "custom",
            "auto_map": {"AutoModel": "modeling_untrusted.UntrustedModel"},
        },
    )
    monkeypatch.setattr(HuggingFaceClient, "resolve_revision", lambda *args: "commit-a")

    def reject_code_fetch(*args, **kwargs):
        raise AssertionError("remote code must not be listed or downloaded")

    monkeypatch.setattr(HuggingFaceClient, "list_tree", reject_code_fetch)
    monkeypatch.setattr(HuggingFaceClient, "download_text", reject_code_fetch)

    resolved = resolver.resolve(
        source="hf",
        model_id="Org/Model",
        cache_policy="refresh",
    )

    assert (resolved.local_dir / "config.json").exists()
    assert list(resolved.local_dir.glob("*.py")) == []
    assert "remote_code_fetch" not in resolved.source


def test_hf_cache_isolated_by_endpoint_and_revision(tmp_path, monkeypatch):
    downloads = []

    def configure(resolver, marker):
        monkeypatch.setattr(resolver._hf, "resolve_revision", lambda model_id, revision: f"{marker}-{revision}")

        def download(model_id, filename, revision):
            downloads.append((marker, revision))
            return {"model_type": "tiny", "marker": marker, "revision": revision}

        monkeypatch.setattr(resolver._hf, "download_json", download)

    hf = ModelSourceResolver(AppSettings(model_root=tmp_path, hf_endpoint="https://huggingface.co", auto_fetch_remote_code=False))
    ms = ModelSourceResolver(AppSettings(model_root=tmp_path, hf_endpoint="https://www.modelscope.cn", auto_fetch_remote_code=False))
    configure(hf, "hf")
    configure(ms, "ms")

    hf_a = hf.resolve(source="hf", model_id="Org/Model", revision="rev-a", cache_policy="refresh")
    hf_b = hf.resolve(source="hf", model_id="Org/Model", revision="rev-b", cache_policy="refresh")
    ms_a = ms.resolve(source="hf", model_id="Org/Model", revision="rev-a", cache_policy="refresh")

    monkeypatch.setattr(hf._hf, "download_json", lambda *args: (_ for _ in ()).throw(AssertionError("cache miss")))
    cached_a = hf.resolve(source="hf", model_id="Org/Model", revision="rev-a", cache_policy="prefer-local")

    assert hf_a.local_dir != hf_b.local_dir
    assert hf_a.local_dir != ms_a.local_dir
    assert cached_a.config == hf_a.config
    assert cached_a.source["kind"] == "hf cache"
    assert cached_a.source["resolved_revision"] == "hf-rev-a"
    assert downloads == [("hf", "hf-rev-a"), ("hf", "hf-rev-b"), ("ms", "ms-rev-a")]


def test_remote_snapshot_remains_visible_in_local_model_list(tmp_path, monkeypatch):
    resolver = _stub_resolver(tmp_path, auto_fetch=False)
    monkeypatch.setattr(resolver._hf, "resolve_revision", lambda model_id, revision: "commit-a")
    monkeypatch.setattr(
        resolver._hf,
        "download_json",
        lambda model_id, filename, revision: {
            "model_type": "tiny",
            "num_hidden_layers": 1,
            "hidden_size": 8,
        },
    )
    resolved = resolver.resolve(source="hf", model_id="Org/Model", cache_policy="refresh")

    entries = resolver.list_local_models()

    assert len(entries) == 1
    assert entries[0].model_id == "Org/Model"
    assert entries[0].config_path == str(resolved.local_dir / "config.json")
    assert entries[0].load_by == "config_path"


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


def test_hf_list_tree_is_best_effort_and_silent(monkeypatch):
    calls = []

    def fake_request(self, url, *, context, log_errors=True):
        calls.append(log_errors)
        raise RemoteError("network reset")

    monkeypatch.setattr(HuggingFaceClient, "_request_text", fake_request)

    assert HuggingFaceClient("https://example.test").list_tree("Org/Model", "main") == []
    assert calls == [False]


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
