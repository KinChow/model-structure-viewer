from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import IntrospectionError
from ..schemas import VerifyResponse
from ..structure import builder
from ..structure.introspect import build_from_meta_model
from ..structure.repair import RepairContext, classify_introspection_error
from ..structure.repair.compat import (
    AttentionImplementationNormalizer,
    CompositeConfigNormalizer,
    CompositeRuntimePatch,
    KimiTieWeightsCompatPatch,
    is_flash_attention2_unavailable,
    is_kimi_tie_weights_signature_error,
)
from ..structure.repair.runtime import ConfigNormalizer, RuntimePatch
from ..structure.repair.runner import try_repair


def verify_transformers_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | str | None = None,
) -> VerifyResponse:
    """Validate that Transformers can build the model on meta tensors.

    This is intentionally strict. It never returns config-derived structure for
    failed model construction, because that would hide unsupported Transformers
    behavior.
    """
    source_info = dict(source)
    local_path = builder._coerce_path(local_dir)
    try:
        structure = build_from_meta_model(config, source=source_info, local_dir=local_path)
        return _passed(structure.summary, source_info)
    except IntrospectionError as exc:
        repaired = _try_repaired_meta(config, source=source_info, local_dir=local_path, error=exc)
        if repaired is not None:
            return repaired
        failure_kind = classify_introspection_error(exc)
        tie_weights_normalized = _try_kimi_tie_weights_compat_meta(
            config,
            source=source_info,
            local_dir=local_path,
            original_error=exc,
            diagnostics={
                "failure_kind": failure_kind.value,
                "repair_status": "not_attempted",
                "retry_reason": "tie_weights_signature_compat",
            },
        )
        if tie_weights_normalized is not None:
            return tie_weights_normalized
        attention_normalized = _try_attention_normalized_meta(
            config,
            source=source_info,
            local_dir=local_path,
            original_error=exc,
            diagnostics={
                "failure_kind": failure_kind.value,
                "repair_status": "not_attempted",
                "retry_reason": "flash_attention_2_unavailable",
            },
        )
        if attention_normalized is not None:
            return attention_normalized
        return VerifyResponse(
            ok=False,
            status="failed",
            source=source_info,
            model_id=source_info.get("model_id"),
            summary=_minimal_summary(config),
            diagnostics={
                "failure_kind": failure_kind.value,
                "repair_status": "not_attempted",
            },
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - verification reports errors instead of raising
        return VerifyResponse(
            ok=False,
            status="failed",
            source=source_info,
            model_id=source_info.get("model_id"),
            summary=_minimal_summary(config),
            diagnostics={
                "failure_kind": "unknown",
                "error_type": type(exc).__name__,
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def _try_repaired_meta(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    error: IntrospectionError,
) -> VerifyResponse | None:
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
    try:
        structure = build_from_meta_model(
            repair_result.config,
            source={
                **source,
                "diagnostics": {
                    "failure_kind": failure_kind.value,
                    **repair_result.diagnostics,
                    "retry_count": 1,
                },
            },
            local_dir=repair_result.local_dir,
            config_overrides=repair_result.config_overrides,
            runtime_patch=repair_result.runtime_patch,
            config_normalizer=repair_result.config_normalizer,
        )
        summary = dict(structure.summary)
        summary["strategy"] = "repaired-transformers-meta"
        diagnostics = dict(structure.source.get("diagnostics") or {})
        diagnostics.update(
            {
                "failure_kind": failure_kind.value,
                **repair_result.diagnostics,
                "repair_status": "success",
                "retry_count": 1,
            }
        )
        return VerifyResponse(
            ok=True,
            status="passed",
            strategy="transformers-meta",
            model_id=source.get("model_id"),
            source=source,
            summary=summary,
            diagnostics=diagnostics,
        )
    except IntrospectionError as retry_exc:
        tie_weights_normalized = _try_kimi_tie_weights_compat_meta(
            repair_result.config,
            source=source,
            local_dir=repair_result.local_dir,
            original_error=retry_exc,
            diagnostics={
                "failure_kind": failure_kind.value,
                **repair_result.diagnostics,
                "repair_status": "failed",
                "retry_count": 1,
                "retry_reason": "tie_weights_signature_compat",
            },
            config_overrides=repair_result.config_overrides,
            runtime_patch=CompositeRuntimePatch(
                repair_result.runtime_patch,
                KimiTieWeightsCompatPatch(),
            ),
            config_normalizer=repair_result.config_normalizer,
        )
        if tie_weights_normalized is not None:
            return tie_weights_normalized
        attention_normalized = _try_attention_normalized_meta(
            repair_result.config,
            source=source,
            local_dir=repair_result.local_dir,
            original_error=retry_exc,
            diagnostics={
                "failure_kind": failure_kind.value,
                **repair_result.diagnostics,
                "repair_status": "failed",
                "retry_count": 1,
                "retry_reason": "flash_attention_2_unavailable",
            },
            config_overrides=repair_result.config_overrides,
            runtime_patch=repair_result.runtime_patch,
            config_normalizer=CompositeConfigNormalizer(
                repair_result.config_normalizer,
                AttentionImplementationNormalizer("sdpa"),
            ),
        )
        if attention_normalized is not None:
            return attention_normalized
        return VerifyResponse(
            ok=False,
            status="failed",
            source=source,
            model_id=source.get("model_id"),
            summary=_minimal_summary(config),
            diagnostics={
                "failure_kind": failure_kind.value,
                **repair_result.diagnostics,
                "repair_status": "failed",
                "retry_count": 1,
            },
            error=str(retry_exc),
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
) -> VerifyResponse | None:
    if not is_kimi_tie_weights_signature_error(original_error):
        return None
    try:
        structure = build_from_meta_model(
            config,
            source={
                **source,
                "diagnostics": {
                    **diagnostics,
                    "runtime_patch": "kimi_tie_weights_compat",
                    "retry_count": max(1, int(diagnostics.get("retry_count") or 0) + 1),
                },
            },
            local_dir=local_dir,
            config_overrides=config_overrides,
            runtime_patch=runtime_patch or KimiTieWeightsCompatPatch(),
            config_normalizer=config_normalizer,
        )
        summary = dict(structure.summary)
        summary["strategy"] = "tie-weights-compatible-transformers-meta"
        result_diagnostics = dict(structure.source.get("diagnostics") or {})
        result_diagnostics.update(
            {
                **diagnostics,
                "runtime_patch": "kimi_tie_weights_compat",
                "retry_status": "success",
                "retry_count": max(1, int(diagnostics.get("retry_count") or 0) + 1),
            }
        )
        return VerifyResponse(
            ok=True,
            status="passed",
            strategy="transformers-meta",
            model_id=source.get("model_id"),
            source=source,
            summary=summary,
            diagnostics=result_diagnostics,
        )
    except IntrospectionError as retry_exc:
        return _retry_failed_response(
            config,
            source=source,
            diagnostics=diagnostics,
            retry_exc=retry_exc,
            retry_key="runtime_patch",
            retry_value="kimi_tie_weights_compat",
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
) -> VerifyResponse | None:
    if not is_flash_attention2_unavailable(original_error):
        return None
    normalizer = config_normalizer or AttentionImplementationNormalizer("sdpa")
    try:
        structure = build_from_meta_model(
            config,
            source={
                **source,
                "diagnostics": {
                    **diagnostics,
                    "attention_backend_retry": "sdpa",
                    "retry_count": max(1, int(diagnostics.get("retry_count") or 0) + 1),
                },
            },
            local_dir=local_dir,
            config_overrides=config_overrides,
            runtime_patch=runtime_patch,
            config_normalizer=normalizer,
        )
        summary = dict(structure.summary)
        summary["strategy"] = "attention-normalized-transformers-meta"
        result_diagnostics = dict(structure.source.get("diagnostics") or {})
        result_diagnostics.update(
            {
                **diagnostics,
                "attention_backend_retry": "sdpa",
                "repair_status": diagnostics.get("repair_status", "not_attempted"),
                "retry_status": "success",
                "retry_count": max(1, int(diagnostics.get("retry_count") or 0) + 1),
            }
        )
        return VerifyResponse(
            ok=True,
            status="passed",
            strategy="transformers-meta",
            model_id=source.get("model_id"),
            source=source,
            summary=summary,
            diagnostics=result_diagnostics,
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
                    "retry_count": max(1, int(diagnostics.get("retry_count") or 0) + 1),
                    "retry_reason": "tie_weights_signature_compat",
                },
                config_overrides=config_overrides,
                runtime_patch=CompositeRuntimePatch(runtime_patch, KimiTieWeightsCompatPatch()),
                config_normalizer=normalizer,
            )
        return _retry_failed_response(
            config,
            source=source,
            diagnostics=diagnostics,
            retry_exc=retry_exc,
            retry_key="attention_backend_retry",
            retry_value="sdpa",
        )


def _retry_failed_response(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    diagnostics: dict[str, Any],
    retry_exc: IntrospectionError,
    retry_key: str,
    retry_value: Any,
) -> VerifyResponse:
    failure_kind = classify_introspection_error(retry_exc)
    return VerifyResponse(
        ok=False,
        status="failed",
        strategy="transformers-meta",
        model_id=source.get("model_id"),
        source=source,
        summary=_minimal_summary(config),
        diagnostics={
            **diagnostics,
            "failure_kind": failure_kind.value,
            retry_key: retry_value,
            "retry_status": "failed",
            "retry_count": max(1, int(diagnostics.get("retry_count") or 0) + 1),
        },
        error=str(retry_exc),
    )


def _passed(summary: dict[str, Any], source: dict[str, Any]) -> VerifyResponse:
    return VerifyResponse(
        ok=True,
        status="passed",
        strategy="transformers-meta",
        model_id=source.get("model_id"),
        source=source,
        summary=summary,
        diagnostics={"backbone_class": summary.get("backbone_class")},
    )


def _minimal_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": config.get("model_type"),
        "architecture": _architecture(config),
    }


def _architecture(config: dict[str, Any]) -> Any:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return architectures[0]
    return None
