"""Top-level dispatcher: try meta-model introspection, fallback to config recursion."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..schemas import ModelStructure
from .fallback import build_from_config
from .introspect import IntrospectionError, build_from_meta_model

_LOG = logging.getLogger(__name__)


def build_model_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
    detail_level: str = "compressed",
    local_dir: Path | str | None = None,
) -> ModelStructure:
    """Build a ModelStructure from a config dict using meta introspection when possible."""
    base_source = dict(source or {})
    base_source.setdefault("detail_level", detail_level)

    local_path = _coerce_path(local_dir)

    try:
        return build_from_meta_model(config, source=base_source, local_dir=local_path)
    except IntrospectionError as exc:
        _LOG.info("Falling back to config-only structure: %s", exc)
        return build_from_config(config, source=base_source, fallback_reason=str(exc))
    except Exception as exc:  # noqa: BLE001  - safety net, never crash the API
        _LOG.warning("Unexpected introspection failure: %s", exc)
        return build_from_config(
            config,
            source=base_source,
            fallback_reason=f"unexpected: {type(exc).__name__}: {exc}",
        )


def _coerce_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None
