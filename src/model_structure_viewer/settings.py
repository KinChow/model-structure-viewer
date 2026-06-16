from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_ROOT = Path("/Users/zhouzijian01/Desktop/workspace/models")
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_CACHE_POLICY = "prefer-local"


@dataclass
class AppSettings:
    model_root: Path = DEFAULT_MODEL_ROOT
    hf_endpoint: str = DEFAULT_HF_ENDPOINT
    cache_policy: str = DEFAULT_CACHE_POLICY
    offline: bool = False

    @classmethod
    def from_env(cls) -> "AppSettings":
        offline_raw = os.environ.get("MSV_OFFLINE", "0").lower()
        return cls(
            model_root=Path(os.environ.get("MODEL_ROOT", str(DEFAULT_MODEL_ROOT))).expanduser(),
            hf_endpoint=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/"),
            cache_policy=os.environ.get("CACHE_POLICY", DEFAULT_CACHE_POLICY),
            offline=offline_raw in {"1", "true", "yes", "on"},
        )

    def with_overrides(
        self,
        *,
        model_root: str | Path | None = None,
        hf_endpoint: str | None = None,
        cache_policy: str | None = None,
        offline: bool | None = None,
    ) -> "AppSettings":
        return AppSettings(
            model_root=Path(model_root).expanduser() if model_root is not None else self.model_root,
            hf_endpoint=(hf_endpoint.rstrip("/") if hf_endpoint else self.hf_endpoint),
            cache_policy=cache_policy or self.cache_policy,
            offline=self.offline if offline is None else offline,
        )
