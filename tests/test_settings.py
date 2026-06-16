"""Tests for AppSettings env loading and override semantics."""
import os
from pathlib import Path

import pytest

from model_structure_viewer.settings import (
    AppSettings,
    DEFAULT_AUTO_FETCH_REMOTE_CODE,
    DEFAULT_CACHE_POLICY,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_MODEL_ROOT,
)


_ENV_KEYS = (
    "MODEL_ROOT",
    "HF_ENDPOINT",
    "CACHE_POLICY",
    "MSV_OFFLINE",
    "MSV_AUTO_FETCH_REMOTE_CODE",
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_from_env_defaults(clean_env):
    settings = AppSettings.from_env()
    assert settings.model_root == DEFAULT_MODEL_ROOT
    assert settings.hf_endpoint == DEFAULT_HF_ENDPOINT
    assert settings.cache_policy == DEFAULT_CACHE_POLICY
    assert settings.offline is False
    assert settings.auto_fetch_remote_code == DEFAULT_AUTO_FETCH_REMOTE_CODE


def test_from_env_reads_overrides(clean_env, tmp_path):
    clean_env.setenv("MODEL_ROOT", str(tmp_path))
    clean_env.setenv("HF_ENDPOINT", "https://hf-mirror.example.com/")
    clean_env.setenv("CACHE_POLICY", "remote-only")
    clean_env.setenv("MSV_OFFLINE", "true")
    clean_env.setenv("MSV_AUTO_FETCH_REMOTE_CODE", "0")
    settings = AppSettings.from_env()
    assert settings.model_root == tmp_path
    assert settings.hf_endpoint == "https://hf-mirror.example.com"  # trailing slash stripped
    assert settings.cache_policy == "remote-only"
    assert settings.offline is True
    assert settings.auto_fetch_remote_code is False


def test_from_env_offline_truthy_variants(clean_env):
    for value in ("1", "true", "Yes", "ON"):
        clean_env.setenv("MSV_OFFLINE", value)
        assert AppSettings.from_env().offline is True
    clean_env.setenv("MSV_OFFLINE", "no")
    assert AppSettings.from_env().offline is False


def test_with_overrides_only_changes_provided_fields():
    base = AppSettings(
        model_root=Path("/a"),
        hf_endpoint="https://x",
        cache_policy="prefer-local",
        offline=False,
        auto_fetch_remote_code=True,
    )
    out = base.with_overrides(offline=True)
    assert out.offline is True
    assert out.model_root == base.model_root
    assert out.hf_endpoint == base.hf_endpoint
    assert out.cache_policy == base.cache_policy
    assert out.auto_fetch_remote_code is True
    # Original is untouched (dataclasses.replace returns a new instance).
    assert base.offline is False


def test_with_overrides_no_args_returns_equivalent():
    base = AppSettings()
    out = base.with_overrides()
    assert out == base
    assert out is not base


def test_with_overrides_strips_endpoint_trailing_slash():
    out = AppSettings().with_overrides(hf_endpoint="https://hf.example.com/")
    assert out.hf_endpoint == "https://hf.example.com"


def test_with_overrides_expands_user_in_model_root():
    out = AppSettings().with_overrides(model_root="~/models")
    assert "~" not in str(out.model_root)


def test_with_overrides_accepts_explicit_false_for_auto_fetch():
    base = AppSettings(auto_fetch_remote_code=True)
    out = base.with_overrides(auto_fetch_remote_code=False)
    assert out.auto_fetch_remote_code is False


def test_default_auto_fetch_remote_code_is_true():
    assert AppSettings().auto_fetch_remote_code is True
