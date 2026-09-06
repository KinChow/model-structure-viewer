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
    KimiRemoteCodeCompatPatch,
    is_kimi_output_recorder_import_error,
    is_flash_attention2_unavailable,
    is_kimi_tie_weights_signature_error,
)
from .repair.context import RepairResult
from .repair.runtime import ConfigNormalizer, RuntimePatch
from .repair.runner import try_repair

RecoveryKind = Literal[
    "none", "repair", "attention", "kimi", "kimi_remote_code", "repair_attention", "repair_kimi",
    "repair_kimi_remote_code",
]


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
    detail_level: str = "compressed",
) -> MetaRecoveryOutcome:
    base_source = dict(source)
    local_path = coerce_existing_path(local_dir)
    try:
        structure = _invoke_meta_builder(
            config,
            source=base_source,
            local_dir=local_path,
            detail_level=detail_level,
        )
        return MetaRecoveryOutcome(structure=structure)
    except IntrospectionError as exc:
        repaired = _recover_with_repair(
            config,
            source=base_source,
            local_dir=local_path,
            error=exc,
            detail_level=detail_level,
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
            detail_level=detail_level,
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
    detail_level: str,
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
        structure = _invoke_meta_builder(
            repair_result.config,
            source=retry_source,
            local_dir=repair_result.local_dir,
            config_overrides=repair_result.config_overrides,
            runtime_patch=repair_result.runtime_patch,
            config_normalizer=repair_result.config_normalizer,
            detail_level=detail_level,
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
            detail_level=detail_level,
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
    detail_level: str = "compressed",
) -> MetaRecoveryOutcome | None:
    current_error = error
    current_patch = runtime_patch
    current_normalizer = config_normalizer
    current_diagnostics = dict(diagnostics)
    applied: list[str] = []

    while True:
        adapter = next(
            (
                candidate
                for candidate in _runtime_compat_adapters()
                if candidate[0] not in applied and candidate[1](current_error)
            ),
            None,
        )
        if adapter is None:
            if not applied:
                return None
            current_diagnostics.update(
                {
                    "failure_kind": classify_introspection_error(current_error).value,
                    "retry_status": "failed",
                }
            )
            raise MetaRecoveryError(str(current_error), diagnostics=current_diagnostics) from current_error

        name, _predicate, recovery_kind = adapter
        applied.append(name)
        if name == "attention_backend_sdpa":
            current_normalizer = _compose_normalizers(
                current_normalizer,
                AttentionImplementationNormalizer("sdpa"),
            )
            current_diagnostics["attention_backend_retry"] = "sdpa"
        elif name == "kimi_tie_weights_compat":
            current_patch = _compose_runtime_patches(current_patch, KimiTieWeightsCompatPatch())
            current_diagnostics["runtime_patch"] = KimiTieWeightsCompatPatch.name
        elif name == "kimi_remote_code_compat":
            current_patch = _compose_runtime_patches(current_patch, KimiRemoteCodeCompatPatch())
            current_diagnostics["runtime_patch"] = KimiRemoteCodeCompatPatch.name

        retry_count = int(current_diagnostics.get("retry_count") or 0) + 1
        current_diagnostics.update(
            {
                "applied_compatibility": list(applied),
                "retry_count": retry_count,
            }
        )
        retry_source = _with_diagnostics(source, current_diagnostics)
        try:
            structure = _invoke_meta_builder(
                config,
                source=retry_source,
                local_dir=local_dir,
                config_overrides=config_overrides,
                runtime_patch=current_patch,
                config_normalizer=current_normalizer,
                detail_level=detail_level,
            )
            return _mark_runtime_compat(
                structure,
                recovery_kind=_prefixed_kind(recovery_prefix, recovery_kind),
                diagnostics={**current_diagnostics, "retry_status": "success"},
            )
        except IntrospectionError as retry_error:
            current_error = retry_error
            current_diagnostics.update(
                {
                    "failure_kind": classify_introspection_error(retry_error).value,
                    "retry_status": "failed",
                }
            )


def _runtime_compat_adapters():
    """Ordered, composable compatibility adapters."""
    return (
        ("attention_backend_sdpa", is_flash_attention2_unavailable, "attention"),
        ("kimi_tie_weights_compat", is_kimi_tie_weights_signature_error, "kimi"),
        ("kimi_remote_code_compat", is_kimi_output_recorder_import_error, "kimi_remote_code"),
    )


def _compose_runtime_patches(*patches: RuntimePatch | None) -> RuntimePatch | None:
    active = [patch for patch in patches if patch is not None]
    if len(active) <= 1:
        return active[0] if active else None
    return CompositeRuntimePatch(*active)


def _compose_normalizers(*normalizers: ConfigNormalizer | None) -> ConfigNormalizer | None:
    active = [normalizer for normalizer in normalizers if normalizer is not None]
    if len(active) <= 1:
        return active[0] if active else None
    return CompositeConfigNormalizer(*active)


def _invoke_meta_builder(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    detail_level: str,
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
) -> ModelStructure:
    kwargs = {
        "source": source,
        "local_dir": local_dir,
        "config_overrides": config_overrides,
        "runtime_patch": runtime_patch,
        "config_normalizer": config_normalizer,
    }
    if detail_level == "expanded":
        kwargs["collapse_repeated"] = False
    return build_from_meta_model(config, **kwargs)


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


def _prefixed_kind(
    prefix: Literal["repair"] | None,
    kind: Literal["attention", "kimi", "kimi_remote_code"],
) -> RecoveryKind:
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
