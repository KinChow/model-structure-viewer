"""Top-level dispatcher for Transformers meta-model introspection."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..errors import IntrospectionError
from ..schemas import ModelStructure
from .recovery import MetaRecoveryError, build_meta_model_with_recovery

_LOG = logging.getLogger(__name__)


def build_model_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
    detail_level: str = "compressed",
    local_dir: Path | str | None = None,
) -> ModelStructure:
    """Build a ModelStructure from a config dict using Transformers meta introspection."""
    base_source = dict(source or {})
    base_source.setdefault("detail_level", detail_level)

    try:
        return build_meta_model_with_recovery(
            config,
            source=base_source,
            local_dir=local_dir,
        ).structure
    except MetaRecoveryError:
        raise
    except IntrospectionError:
        raise
    except Exception as exc:  # noqa: BLE001  - normalize third-party failures
        _LOG.warning("Unexpected introspection failure: %s", exc)
        raise IntrospectionError(f"Unexpected introspection failure: {type(exc).__name__}: {exc}") from exc
