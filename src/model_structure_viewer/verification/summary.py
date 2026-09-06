from __future__ import annotations

from typing import Any


def minimal_summary(config: dict[str, Any]) -> dict[str, Any]:
    architectures = config.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
    return {
        "model_type": config.get("model_type"),
        "architecture": architecture,
    }
