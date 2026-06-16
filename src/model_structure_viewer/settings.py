from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL_ROOT = Path("/Users/zhouzijian01/Desktop/workspace/models")
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_CACHE_POLICY = "prefer-local"
DEFAULT_AUTO_FETCH_REMOTE_CODE = True


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


# Each entry maps an override field to (skip_predicate, transform).
# ``skip_predicate(value)`` returns True when the override should be ignored,
# preserving the historical semantics:
#   * ``None`` is always treated as "no change"
#   * string fields additionally treat ``""`` as "no change"
#   * bool / Path / non-string values only skip on ``None``
_skip_none: Callable[[Any], bool] = lambda v: v is None
_skip_falsy_str: Callable[[Any], bool] = lambda v: not v  # None or ""

_OVERRIDE_TRANSFORMS: dict[str, tuple[Callable[[Any], bool], Callable[[Any], Any]]] = {
    "model_root": (_skip_none, lambda v: Path(v).expanduser()),
    "hf_endpoint": (_skip_falsy_str, lambda v: v.rstrip("/")),
    "cache_policy": (_skip_falsy_str, lambda v: v),
    "offline": (_skip_none, lambda v: v),
    "auto_fetch_remote_code": (_skip_none, lambda v: v),
}


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
        candidates = {
            "model_root": model_root,
            "hf_endpoint": hf_endpoint,
            "cache_policy": cache_policy,
            "offline": offline,
            "auto_fetch_remote_code": auto_fetch_remote_code,
        }
        changes: dict[str, object] = {}
        for field, value in candidates.items():
            skip, transform = _OVERRIDE_TRANSFORMS[field]
            if skip(value):
                continue
            changes[field] = transform(value)
        return replace(self, **changes)
