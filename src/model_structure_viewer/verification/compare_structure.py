from __future__ import annotations

from typing import Any


def compare_structure_summary(
    *,
    predicted: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare the high-level summary fields that gate catalog verification."""
    predicted_summary = predicted.get("summary") or {}
    reference_summary = reference.get("summary") or {}
    errors: list[str] = []
    warnings: list[str] = []

    for key in ("canonical_architecture", "text_layers"):
        predicted_value = predicted_summary.get(key)
        reference_value = reference_summary.get(key)
        if predicted_value != reference_value:
            errors.append(f"{key} mismatch: predicted={predicted_value!r} reference={reference_value!r}")

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
    }
