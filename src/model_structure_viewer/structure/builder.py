"""Top-level dispatcher for Transformers meta-model introspection."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..schemas import ModelStructure
from .introspect import IntrospectionError, build_from_meta_model
from .repair import RepairContext, classify_introspection_error
from .repair.compat import (
    AttentionImplementationNormalizer,
    CompositeConfigNormalizer,
    CompositeRuntimePatch,
    KimiTieWeightsCompatPatch,
    is_flash_attention2_unavailable,
    is_kimi_tie_weights_signature_error,
)
from .repair.context import RepairResult
from .repair.runtime import ConfigNormalizer, RuntimePatch
from .repair.runner import try_repair

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

    local_path = _coerce_path(local_dir)

    try:
        return build_from_meta_model(config, source=base_source, local_dir=local_path)
    except IntrospectionError as exc:
        repaired = _recover_with_repair(
            config,
            source=base_source,
            local_dir=local_path,
            error=exc,
        )
        if repaired is not None:
            return repaired
        compatible = _recover_with_runtime_compat(
            config,
            source=base_source,
            local_dir=local_path,
            error=exc,
            diagnostics={
                "failure_kind": classify_introspection_error(exc).value,
                "repair_status": "not_attempted",
            },
        )
        if compatible is not None:
            return compatible
        raise exc
    except Exception as exc:  # noqa: BLE001  - normalize third-party failures
        _LOG.warning("Unexpected introspection failure: %s", exc)
        raise IntrospectionError(f"Unexpected introspection failure: {type(exc).__name__}: {exc}") from exc


def _recover_with_repair(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    error: IntrospectionError,
) -> ModelStructure | None:
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
        return None

    retry_source = _with_diagnostics(
        source,
        {
            "failure_kind": failure_kind.value,
            **repair_result.diagnostics,
            "retry_count": 1,
        },
    )
    try:
        structure = build_from_meta_model(
            repair_result.config,
            source=retry_source,
            local_dir=repair_result.local_dir,
            config_overrides=repair_result.config_overrides,
            runtime_patch=repair_result.runtime_patch,
            config_normalizer=repair_result.config_normalizer,
        )
        return _mark_repaired(structure, repair_result, failure_kind)
    except IntrospectionError as retry_exc:
        compatible = _recover_with_runtime_compat(
            repair_result.config,
            source=source,
            local_dir=repair_result.local_dir,
            error=retry_exc,
            diagnostics={
                "failure_kind": failure_kind.value,
                **repair_result.diagnostics,
                "repair_status": "failed",
                "retry_count": 1,
            },
            config_overrides=repair_result.config_overrides,
            runtime_patch=repair_result.runtime_patch,
            config_normalizer=repair_result.config_normalizer,
        )
        if compatible is not None:
            return compatible
        retry_exc.args = (
            _format_retry_failure_message(
                retry_exc,
                failure_kind=failure_kind.value,
                diagnostics=retry_source.get("diagnostics") or {},
            ),
        )
        raise retry_exc


def _recover_with_runtime_compat(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    error: IntrospectionError,
    diagnostics: dict[str, Any],
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
) -> ModelStructure | None:
    attention_normalized = _try_attention_normalized_meta(
        config,
        source=source,
        local_dir=local_dir,
        original_error=error,
        diagnostics=diagnostics,
        config_overrides=config_overrides,
        runtime_patch=runtime_patch,
        config_normalizer=(
            CompositeConfigNormalizer(config_normalizer, AttentionImplementationNormalizer("sdpa"))
            if config_normalizer is not None
            else None
        ),
    )
    if attention_normalized is not None:
        return attention_normalized

    return _try_kimi_tie_weights_compat_meta(
        config,
        source=source,
        local_dir=local_dir,
        original_error=error,
        diagnostics=diagnostics,
        config_overrides=config_overrides,
        runtime_patch=runtime_patch,
        config_normalizer=config_normalizer,
    )


def _try_attention_normalized_meta(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    original_error: IntrospectionError,
    diagnostics: dict[str, Any],
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
) -> ModelStructure | None:
    if not is_flash_attention2_unavailable(original_error):
        return None
    normalizer = config_normalizer or AttentionImplementationNormalizer("sdpa")
    retry_count = max(1, int(diagnostics.get("retry_count") or 0) + 1)
    retry_source = _with_diagnostics(
        source,
        {
            **diagnostics,
            "attention_backend_retry": "sdpa",
            "retry_count": retry_count,
        },
    )
    try:
        structure = build_from_meta_model(
            config,
            source=retry_source,
            local_dir=local_dir,
            config_overrides=config_overrides,
            runtime_patch=runtime_patch,
            config_normalizer=normalizer,
        )
        return _mark_runtime_compat(
            structure,
            strategy="repaired-meta-introspect",
            diagnostics={
                **diagnostics,
                "attention_backend_retry": "sdpa",
                "repair_status": diagnostics.get("repair_status", "not_attempted"),
                "retry_status": "success",
                "retry_count": retry_count,
            },
        )
    except IntrospectionError as retry_exc:
        return _try_kimi_tie_weights_compat_meta(
            config,
            source=source,
            local_dir=local_dir,
            original_error=retry_exc,
            diagnostics={
                **diagnostics,
                "attention_backend_retry": "sdpa",
                "retry_status": "failed",
                "retry_count": retry_count,
            },
            config_overrides=config_overrides,
            runtime_patch=CompositeRuntimePatch(runtime_patch, KimiTieWeightsCompatPatch()),
            config_normalizer=normalizer,
        )


def _try_kimi_tie_weights_compat_meta(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    original_error: IntrospectionError,
    diagnostics: dict[str, Any],
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
) -> ModelStructure | None:
    if not is_kimi_tie_weights_signature_error(original_error):
        return None
    retry_count = max(1, int(diagnostics.get("retry_count") or 0) + 1)
    retry_source = _with_diagnostics(
        source,
        {
            **diagnostics,
            "runtime_patch": "kimi_tie_weights_compat",
            "retry_count": retry_count,
        },
    )
    structure = build_from_meta_model(
        config,
        source=retry_source,
        local_dir=local_dir,
        config_overrides=config_overrides,
        runtime_patch=runtime_patch or KimiTieWeightsCompatPatch(),
        config_normalizer=config_normalizer,
    )
    return _mark_runtime_compat(
        structure,
        strategy="repaired-meta-introspect",
        diagnostics={
            **diagnostics,
            "runtime_patch": "kimi_tie_weights_compat",
            "retry_status": "success",
            "retry_count": retry_count,
        },
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


def _mark_runtime_compat(
    structure: ModelStructure,
    *,
    strategy: str,
    diagnostics: dict[str, Any],
) -> ModelStructure:
    merged = dict(structure.source.get("diagnostics") or {})
    merged.update(diagnostics)
    structure.summary["strategy"] = strategy
    structure.source["strategy"] = strategy
    structure.source["diagnostics"] = merged
    return structure


def _with_diagnostics(source: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(source)
    merged = dict(enriched.get("diagnostics") or {})
    merged.update(diagnostics)
    enriched["diagnostics"] = merged
    return enriched


def _format_retry_failure_message(
    error: IntrospectionError,
    *,
    failure_kind: str,
    diagnostics: dict[str, Any],
) -> str:
    parts = [
        str(error),
        f"failure_kind={failure_kind}",
        f"repair_strategy={diagnostics.get('repair_strategy', 'unknown')}",
        "repair_status=failed",
        f"retry_count={diagnostics.get('retry_count', 1)}",
    ]
    if diagnostics.get("config_normalizer"):
        parts.append(f"config_normalizer={diagnostics['config_normalizer']}")
    if diagnostics.get("runtime_patch"):
        parts.append(f"runtime_patch={diagnostics['runtime_patch']}")
    return "; ".join(parts)


def _coerce_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None
