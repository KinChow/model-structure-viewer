"""Top-level dispatcher: try meta-model introspection, fallback to config recursion."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..schemas import ModelStructure
from .fallback import build_from_config
from .introspect import IntrospectionError, build_from_meta_model
from .repair import RepairContext, classify_introspection_error
from .repair.context import RepairResult
from .repair.runner import try_repair

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
        return _recover_from_introspection_error(
            config,
            source=base_source,
            local_dir=local_path,
            error=exc,
        )
    except Exception as exc:  # noqa: BLE001  - safety net, never crash the API
        _LOG.warning("Unexpected introspection failure: %s", exc)
        diagnostics = {
            "failure_kind": "unknown",
            "repair_status": "not_attempted",
            "error_type": type(exc).__name__,
        }
        return build_from_config(
            config,
            source=_with_diagnostics(base_source, diagnostics),
            fallback_reason=f"unexpected: {type(exc).__name__}: {exc}",
        )


def _recover_from_introspection_error(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    error: IntrospectionError,
) -> ModelStructure:
    failure_kind = classify_introspection_error(error)
    context = RepairContext(
        config=config,
        source=source,
        local_dir=local_dir,
        failure_kind=failure_kind,
        original_error=str(error),
    )
    repair_result = try_repair(context)
    if repair_result is None or repair_result.diagnostics.get("repair_status") == "skipped":
        diagnostics = {
            "failure_kind": failure_kind.value,
            "repair_status": "not_attempted" if repair_result is None else "skipped",
        }
        if repair_result is not None:
            diagnostics.update(repair_result.diagnostics)
        _LOG.info("Falling back to config-only structure: %s", error)
        return build_from_config(
            config,
            source=_with_diagnostics(source, diagnostics),
            fallback_reason=str(error),
        )

    try:
        repaired_source = _with_diagnostics(
            source,
            {
                "failure_kind": failure_kind.value,
                **repair_result.diagnostics,
                "retry_count": 1,
            },
        )
        structure = build_from_meta_model(
            repair_result.config,
            source=repaired_source,
            local_dir=repair_result.local_dir,
            config_overrides=repair_result.config_overrides,
            runtime_patch=repair_result.runtime_patch,
            config_normalizer=repair_result.config_normalizer,
        )
        return _mark_repaired(structure, repair_result, failure_kind)
    except IntrospectionError as retry_exc:
        diagnostics = {
            "failure_kind": failure_kind.value,
            **repair_result.diagnostics,
            "repair_status": "failed",
            "retry_count": 1,
        }
        _LOG.info("Repair retry failed; falling back to config-only structure: %s", retry_exc)
        return build_from_config(
            config,
            source=_with_diagnostics(source, diagnostics),
            fallback_reason=str(retry_exc),
        )


def _mark_repaired(
    structure: ModelStructure,
    repair_result: RepairResult,
    failure_kind: Any,
) -> ModelStructure:
    diagnostics = dict(structure.source.get("diagnostics") or {})
    diagnostics.update(repair_result.diagnostics)
    diagnostics.update(
        {
            "failure_kind": failure_kind.value,
            "repair_strategy": repair_result.strategy_name,
            "repair_status": "success",
            "retry_count": 1,
        }
    )
    structure.summary["strategy"] = "repaired-meta-introspect"
    structure.source["strategy"] = "repaired-meta-introspect"
    structure.source["diagnostics"] = diagnostics
    return structure


def _with_diagnostics(source: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(source)
    merged = dict(enriched.get("diagnostics") or {})
    merged.update(diagnostics)
    enriched["diagnostics"] = merged
    return enriched


def _coerce_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None
