from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import VerifyResponse
from ..structure.recovery import MetaRecoveryError, MetaRecoveryOutcome, build_meta_model_with_recovery


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
    try:
        outcome = build_meta_model_with_recovery(config, source=source_info, local_dir=local_dir)
        return _passed(outcome, source_info)
    except MetaRecoveryError as exc:
        return VerifyResponse(
            ok=False,
            status="failed",
            strategy="transformers-meta",
            source=source_info,
            model_id=source_info.get("model_id"),
            summary=_minimal_summary(config),
            diagnostics=exc.diagnostics,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - verification reports errors instead of raising
        return VerifyResponse(
            ok=False,
            status="failed",
            strategy="transformers-meta",
            source=source_info,
            model_id=source_info.get("model_id"),
            summary=_minimal_summary(config),
            diagnostics={
                "failure_kind": "unknown",
                "error_type": type(exc).__name__,
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def _passed(outcome: MetaRecoveryOutcome, source: dict[str, Any]) -> VerifyResponse:
    summary = dict(outcome.structure.summary)
    diagnostics = dict(outcome.diagnostics or outcome.structure.source.get("diagnostics") or {})
    if outcome.recovery_kind == "none":
        diagnostics.setdefault("backbone_class", summary.get("backbone_class"))
    else:
        summary["strategy"] = _verify_summary_strategy(outcome.recovery_kind)
    return VerifyResponse(
        ok=True,
        status="passed",
        strategy="transformers-meta",
        model_id=source.get("model_id"),
        source=source,
        summary=summary,
        diagnostics=diagnostics,
    )


def _verify_summary_strategy(recovery_kind: str) -> str:
    if recovery_kind == "repair":
        return "repaired-transformers-meta"
    if recovery_kind in {"attention", "repair_attention"}:
        return "attention-normalized-transformers-meta"
    if recovery_kind in {"kimi", "repair_kimi"}:
        return "tie-weights-compatible-transformers-meta"
    return "transformers-meta"


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
