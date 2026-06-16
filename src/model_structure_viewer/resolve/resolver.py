"""Orchestrator that wires :class:`HuggingFaceClient`, :class:`LocalModelCache`,
and :class:`RemoteCodeFetcher` into the public ``ModelSourceResolver`` API.

The single responsibility of this module is **routing**: pick the right source
(uploaded JSON / local file / local cache / HF remote) and stitch together
local-cache writes plus best-effort remote-code fetching. All HTTP lives in
``hf_client``; all filesystem reads live in ``local_cache``; this file is the
seam that decides which to call.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..errors import ConfigError, NotFoundError, RemoteError
from ..schemas import HfSearchResult, ModelEntry
from ..settings import AppSettings
from .hf_client import HuggingFaceClient
from .local_cache import LocalModelCache, ResolvedConfig
from .remote_code import RemoteCodeFetcher

_LOG = logging.getLogger(__name__)


class ModelSourceResolver:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or AppSettings.from_env()
        self._cache = LocalModelCache(self.settings.model_root)
        self._hf = HuggingFaceClient(self.settings.hf_endpoint)
        self._remote_code = RemoteCodeFetcher(self._hf)

    # ---- listing / paths -----------------------------------------------------------
    def list_local_models(self) -> list[ModelEntry]:
        return self._cache.list_local_models()

    def local_config_path(self, model_id: str) -> Path:
        return self._cache.local_config_path(model_id)

    # ---- main entry ----------------------------------------------------------------
    def resolve(
        self,
        *,
        source: str = "auto",
        model_id: str | None = None,
        config_path: str | None = None,
        config_json: dict[str, Any] | None = None,
        revision: str = "main",
        cache_policy: str = "prefer-local",
        detail_level: str = "compressed",
    ) -> ResolvedConfig:
        _LOG.info(
            "resolve start source=%s model_id=%s cache_policy=%s offline=%s",
            source,
            model_id,
            cache_policy,
            self.settings.offline,
        )

        if source == "config":
            if config_json is None:
                raise ConfigError("source=config requires config_json.")
            return ResolvedConfig(
                config=config_json,
                source={"kind": "uploaded config", "detail_level": detail_level},
            )

        if config_path:
            resolved = self._cache.resolve_config_path(config_path, detail_level)
            self._maybe_fetch_remote_code(resolved, model_id=model_id, revision=revision)
            return resolved

        if not model_id:
            raise ConfigError("model_id, config_path, or config_json is required.")

        effective_policy = "offline" if self.settings.offline else cache_policy
        if source == "local":
            resolved = self._cache.resolve_local_model(model_id, detail_level)
        elif source == "hf":
            if effective_policy == "offline":
                raise ConfigError("source=hf cannot run with offline cache policy.")
            resolved = self._resolve_hf_model(model_id, revision, cache_policy, detail_level)
        elif source == "auto":
            resolved = self._resolve_auto(
                model_id, revision, cache_policy, effective_policy, detail_level
            )
        else:
            raise ConfigError(f"Unsupported source: {source}")

        self._maybe_fetch_remote_code(resolved, model_id=model_id, revision=revision)
        return resolved

    # ---- HF helpers ----------------------------------------------------------------
    def search_hf_models(self, query: str, limit: int = 10) -> list[HfSearchResult]:
        if self.settings.offline:
            raise ConfigError("HF search is disabled in offline mode.")
        payload = self._hf.search_models(query, limit)
        results: list[HfSearchResult] = []
        for item in payload:
            model_id = item.get("modelId") or item.get("id")
            if not model_id:
                continue
            results.append(
                HfSearchResult(
                    model_id=model_id,
                    pipeline_tag=item.get("pipeline_tag"),
                    tags=item.get("tags") or [],
                    downloads=item.get("downloads"),
                    likes=item.get("likes"),
                )
            )
        return results

    def get_remote_config(self, model_id: str, revision: str = "main") -> dict[str, Any]:
        if self.settings.offline:
            raise ConfigError("HF config lookup is disabled in offline mode.")
        return self._hf.download_json(model_id, "config.json", revision)

    # ---- internal: source routing --------------------------------------------------
    def _resolve_auto(
        self,
        model_id: str,
        revision: str,
        cache_policy: str,
        effective_policy: str,
        detail_level: str,
    ) -> ResolvedConfig:
        local: ResolvedConfig | None = None
        if effective_policy != "refresh":
            local = self._cache.try_local_model(model_id, detail_level)
        if local is not None:
            _LOG.info("auto resolved %s via local cache", model_id)
            return local
        if effective_policy == "offline":
            expected = self._cache.local_config_path(model_id)
            raise NotFoundError(f"Local config not found and offline is enabled: {expected}")
        _LOG.info("auto falling back to HF for %s (policy=%s)", model_id, cache_policy)
        return self._resolve_hf_model(model_id, revision, cache_policy, detail_level)

    def _resolve_hf_model(
        self,
        model_id: str,
        revision: str,
        cache_policy: str,
        detail_level: str,
    ) -> ResolvedConfig:
        cache_dir = self._cache.local_config_path(model_id).parent
        config_path = cache_dir / "config.json"
        if cache_policy == "prefer-local" and config_path.exists():
            return self._cache.resolve_local_model(model_id, detail_level)

        cache_dir.mkdir(parents=True, exist_ok=True)
        config = self._hf.download_json(model_id, "config.json", revision)
        self._cache.write_json(config_path, config)
        self._remote_code.cache_metadata_files(model_id, revision, cache_dir)
        return ResolvedConfig(
            config=config,
            source={
                "kind": "hf remote",
                "model_id": model_id,
                "revision": revision,
                "hf_endpoint": self.settings.hf_endpoint,
                "cache_path": str(config_path),
                "detail_level": detail_level,
            },
            local_dir=cache_dir,
        )

    def _maybe_fetch_remote_code(
        self,
        resolved: ResolvedConfig,
        *,
        model_id: str | None,
        revision: str,
    ) -> None:
        if resolved.local_dir is None or model_id is None:
            return
        if self.settings.offline or not self.settings.auto_fetch_remote_code:
            return
        info = self._remote_code.ensure_remote_code(
            model_id=model_id,
            revision=revision,
            local_dir=resolved.local_dir,
            config=resolved.config,
        )
        if info.get("fetched") or info.get("errors"):
            resolved.source["remote_code_fetch"] = info

    # ---- backwards-compat: callers that imported ensure_remote_code directly -------
    def ensure_remote_code(
        self,
        *,
        model_id: str,
        revision: str,
        local_dir: Path,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return self._remote_code.ensure_remote_code(
            model_id=model_id,
            revision=revision,
            local_dir=local_dir,
            config=config,
        )


__all__ = ["ModelSourceResolver", "ResolvedConfig", "RemoteError"]
