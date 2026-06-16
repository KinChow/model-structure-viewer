from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_MODEL_ROOT = Path("/Users/zhouzijian01/Desktop/workspace/models")
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_CACHE_POLICY = "prefer-local"
DEFAULT_AUTO_FETCH_REMOTE_CODE = True


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class AppSettings:
    model_root: Path = DEFAULT_MODEL_ROOT
    hf_endpoint: str = DEFAULT_HF_ENDPOINT
    cache_policy: str = DEFAULT_CACHE_POLICY
    offline: bool = False
    auto_fetch_remote_code: bool = DEFAULT_AUTO_FETCH_REMOTE_CODE

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            model_root=Path(os.environ.get("MODEL_ROOT", str(DEFAULT_MODEL_ROOT))).expanduser(),
            hf_endpoint=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/"),
            cache_policy=os.environ.get("CACHE_POLICY", DEFAULT_CACHE_POLICY),
            offline=_parse_bool(os.environ.get("MSV_OFFLINE", "0")),
            auto_fetch_remote_code=_parse_bool(
                os.environ.get("MSV_AUTO_FETCH_REMOTE_CODE", "1")
            ),
        )

    def with_overrides(
        self,
        *,
        model_root: str | Path | None = None,
        hf_endpoint: str | None = None,
        cache_policy: str | None = None,
        offline: bool | None = None,
        auto_fetch_remote_code: bool | None = None,
    ) -> "AppSettings":
        changes: dict[str, object] = {}
        if model_root is not None:
            changes["model_root"] = Path(model_root).expanduser()
        if hf_endpoint:
            changes["hf_endpoint"] = hf_endpoint.rstrip("/")
        if cache_policy:
            changes["cache_policy"] = cache_policy
        if offline is not None:
            changes["offline"] = offline
        if auto_fetch_remote_code is not None:
            changes["auto_fetch_remote_code"] = auto_fetch_remote_code
        return replace(self, **changes)
