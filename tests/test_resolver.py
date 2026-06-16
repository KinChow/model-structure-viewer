import json
from pathlib import Path

import pytest

from model_structure_viewer.resolver import ModelSourceResolver, SourceResolutionError
from model_structure_viewer.settings import AppSettings

FIXTURE = Path(__file__).parent / "fixtures" / "minimax_m3" / "config.json"


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
