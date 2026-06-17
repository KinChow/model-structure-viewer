from __future__ import annotations

from typing import Any

from ..context import RepairContext, RepairResult
from ..errors import IntrospectionFailureKind


class MiniMaxConfigAdapterStrategy:
    name = "minimax_config_adapter"

    def matches(self, context: RepairContext) -> bool:
        return (
            context.failure_kind == IntrospectionFailureKind.CONFIG_FIELD_MISSING
            and "temporal_patch_size" in context.original_error
            and _is_minimax(context)
        )

    def apply(self, context: RepairContext) -> RepairResult:
        patched = dict(context.config)
        temporal_patch_size = _find_nested_value(context.config, "temporal_patch_size")
        if temporal_patch_size is None:
            return RepairResult(
                config=patched,
                local_dir=context.local_dir,
                strategy_name=self.name,
                diagnostics={
                    "repair_strategy": self.name,
                    "repair_status": "skipped",
                    "reason": "temporal_patch_size_not_found",
                },
            )
        patched["temporal_patch_size"] = temporal_patch_size
        return RepairResult(
            config=patched,
            local_dir=context.local_dir,
            strategy_name=self.name,
            diagnostics={
                "repair_strategy": self.name,
                "repair_status": "prepared",
                "patched_fields": ["temporal_patch_size"],
                "config_overrides": ["temporal_patch_size"],
            },
            config_overrides={"temporal_patch_size": temporal_patch_size},
        )


def _is_minimax(context: RepairContext) -> bool:
    values = [str(context.config.get("model_type", "")), str(context.source.get("model_id", ""))]
    values.extend(str(item) for item in context.config.get("architectures") or [])
    return any("minimax" in value.lower() for value in values)


def _find_nested_value(value: Any, key: str) -> Any:
    if not isinstance(value, dict):
        return None
    if key in value:
        return value[key]
    for nested in value.values():
        found = _find_nested_value(nested, key)
        if found is not None:
            return found
    return None
