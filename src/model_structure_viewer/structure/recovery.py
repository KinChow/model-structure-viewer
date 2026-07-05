from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..errors import IntrospectionError
from ..schemas import ModelStructure
from .introspect import build_from_meta_model
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

RecoveryKind = Literal["none", "repair", "attention", "kimi", "repair_attention", "repair_kimi"]


@dataclass(frozen=True)
class MetaRecoveryOutcome:
    structure: ModelStructure
    recovery_kind: RecoveryKind = "none"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class MetaRecoveryError(IntrospectionError):
    def __init__(self, message: str, *, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def build_meta_model_with_recovery(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | str | None = None,
) -> MetaRecoveryOutcome:
    base_source = dict(source)
    local_path = coerce_existing_path(local_dir)
    try:
        structure = build_from_meta_model(config, source=base_source, local_dir=local_path)
        return MetaRecoveryOutcome(structure=structure)
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
            recovery_prefix=None,
        )
        if compatible is not None:
            return compatible
        failure_kind = classify_introspection_error(exc).value
        raise MetaRecoveryError(
            str(exc),
            diagnostics={
                "failure_kind": failure_kind,
                "repair_status": "not_attempted",
            },
        ) from exc


def coerce_existing_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _recover_with_repair(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    error: IntrospectionError,
) -> MetaRecoveryOutcome | None:
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
        return _mark_repaired(structure, repair_result, failure_kind.value)
    except IntrospectionError as retry_exc:
        diagnostics = {
            "failure_kind": failure_kind.value,
            **repair_result.diagnostics,
            "repair_status": "failed",
            "retry_count": 1,
        }
        compatible = _recover_with_runtime_compat(
            repair_result.config,
            source=source,
            local_dir=repair_result.local_dir,
            error=retry_exc,
            diagnostics=diagnostics,
            config_overrides=repair_result.config_overrides,
            runtime_patch=repair_result.runtime_patch,
            config_normalizer=repair_result.config_normalizer,
            recovery_prefix="repair",
        )
        if compatible is not None:
            return compatible
        raise MetaRecoveryError(
            _format_retry_failure_message(retry_exc, diagnostics=diagnostics),
            diagnostics=diagnostics,
        ) from retry_exc


def _recover_with_runtime_compat(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    error: IntrospectionError,
    diagnostics: dict[str, Any],
    recovery_prefix: Literal["repair"] | None,
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
) -> MetaRecoveryOutcome | None:
    attention_normalized = _try_attention_normalized_meta(
        config,
        source=source,
        local_dir=local_dir,
        original_error=error,
        diagnostics=diagnostics,
        recovery_prefix=recovery_prefix,
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
        recovery_prefix=recovery_prefix,
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
    recovery_prefix: Literal["repair"] | None,
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
) -> MetaRecoveryOutcome | None:
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
            recovery_kind=_prefixed_kind(recovery_prefix, "attention"),
            diagnostics={
                **diagnostics,
                "attention_backend_retry": "sdpa",
                "repair_status": diagnostics.get("repair_status", "not_attempted"),
                "retry_status": "success",
                "retry_count": retry_count,
            },
        )
    except IntrospectionError as retry_exc:
        if is_kimi_tie_weights_signature_error(retry_exc):
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
                recovery_prefix=recovery_prefix,
                config_overrides=config_overrides,
                runtime_patch=CompositeRuntimePatch(runtime_patch, KimiTieWeightsCompatPatch()),
                config_normalizer=normalizer,
            )
        failure_kind = classify_introspection_error(retry_exc).value
        failed_diagnostics = {
            **diagnostics,
            "failure_kind": failure_kind,
            "attention_backend_retry": "sdpa",
            "retry_status": "failed",
            "retry_count": retry_count,
        }
        raise MetaRecoveryError(str(retry_exc), diagnostics=failed_diagnostics) from retry_exc


def _try_kimi_tie_weights_compat_meta(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    original_error: IntrospectionError,
    diagnostics: dict[str, Any],
    recovery_prefix: Literal["repair"] | None,
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
) -> MetaRecoveryOutcome | None:
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
    try:
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
            recovery_kind=_prefixed_kind(recovery_prefix, "kimi"),
            diagnostics={
                **diagnostics,
                "runtime_patch": "kimi_tie_weights_compat",
                "retry_status": "success",
                "retry_count": retry_count,
            },
        )
    except IntrospectionError as retry_exc:
        failure_kind = classify_introspection_error(retry_exc).value
        failed_diagnostics = {
            **diagnostics,
            "failure_kind": failure_kind,
            "runtime_patch": "kimi_tie_weights_compat",
            "retry_status": "failed",
            "retry_count": retry_count,
        }
        raise MetaRecoveryError(str(retry_exc), diagnostics=failed_diagnostics) from retry_exc


def _mark_repaired(
    structure: ModelStructure,
    repair_result: RepairResult,
    failure_kind: str,
) -> MetaRecoveryOutcome:
    diagnostics = dict(structure.source.get("diagnostics") or {})
    diagnostics.update(repair_result.diagnostics)
    diagnostics.update(
        {
            "failure_kind": failure_kind,
            "repair_strategy": repair_result.strategy_name,
            "repair_status": "success",
            "retry_count": 1,
        }
    )
    structure.summary["strategy"] = "repaired-meta-introspect"
    structure.source["strategy"] = "repaired-meta-introspect"
    structure.source["diagnostics"] = diagnostics
    return MetaRecoveryOutcome(structure=structure, recovery_kind="repair", diagnostics=diagnostics)


def _mark_runtime_compat(
    structure: ModelStructure,
    *,
    recovery_kind: RecoveryKind,
    diagnostics: dict[str, Any],
) -> MetaRecoveryOutcome:
    merged = dict(structure.source.get("diagnostics") or {})
    merged.update(diagnostics)
    structure.summary["strategy"] = "repaired-meta-introspect"
    structure.source["strategy"] = "repaired-meta-introspect"
    structure.source["diagnostics"] = merged
    return MetaRecoveryOutcome(structure=structure, recovery_kind=recovery_kind, diagnostics=merged)


def _with_diagnostics(source: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(source)
    merged = dict(enriched.get("diagnostics") or {})
    merged.update(diagnostics)
    enriched["diagnostics"] = merged
    return enriched


def _prefixed_kind(prefix: Literal["repair"] | None, kind: Literal["attention", "kimi"]) -> RecoveryKind:
    if prefix == "repair":
        return f"repair_{kind}"
    return kind


def _format_retry_failure_message(
    error: IntrospectionError,
    *,
    diagnostics: dict[str, Any],
) -> str:
    parts = [
        str(error),
        f"failure_kind={diagnostics.get('failure_kind', 'unknown')}",
        f"repair_strategy={diagnostics.get('repair_strategy', 'unknown')}",
        "repair_status=failed",
        f"retry_count={diagnostics.get('retry_count', 1)}",
    ]
    if diagnostics.get("config_normalizer"):
        parts.append(f"config_normalizer={diagnostics['config_normalizer']}")
    if diagnostics.get("runtime_patch"):
        parts.append(f"runtime_patch={diagnostics['runtime_patch']}")
    return "; ".join(parts)
