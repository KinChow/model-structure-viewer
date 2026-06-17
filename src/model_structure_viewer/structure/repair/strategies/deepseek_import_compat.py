from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from ..context import RepairContext, RepairResult
from ..errors import IntrospectionFailureKind


class DeepSeekTorchFxCompatPatch:
    name = "deepseek_torch_fx_compat"

    @contextmanager
    def activate(self) -> Iterator[None]:
        import transformers.utils.import_utils as import_utils

        existed = hasattr(import_utils, "is_torch_fx_available")
        previous = getattr(import_utils, "is_torch_fx_available", None)
        if not existed:
            import_utils.is_torch_fx_available = lambda: False
        try:
            yield
        finally:
            if existed:
                import_utils.is_torch_fx_available = previous
            elif hasattr(import_utils, "is_torch_fx_available"):
                delattr(import_utils, "is_torch_fx_available")


class DeepSeekImportCompatStrategy:
    name = "deepseek_import_compat"

    def matches(self, context: RepairContext) -> bool:
        return (
            context.failure_kind == IntrospectionFailureKind.REMOTE_IMPORT_COMPAT
            and "is_torch_fx_available" in context.original_error
            and _is_deepseek(context)
        )

    def apply(self, context: RepairContext) -> RepairResult:
        runtime_patch = DeepSeekTorchFxCompatPatch()
        return RepairResult(
            config=dict(context.config),
            local_dir=context.local_dir,
            strategy_name=self.name,
            diagnostics={
                "repair_strategy": self.name,
                "repair_status": "prepared",
                "compat_symbol": "is_torch_fx_available",
                "runtime_patch": runtime_patch.name,
            },
            runtime_patch=runtime_patch,
        )


def _is_deepseek(context: RepairContext) -> bool:
    values = [str(context.config.get("model_type", "")), str(context.source.get("model_id", ""))]
    values.extend(str(item) for item in context.config.get("architectures") or [])
    return any("deepseek" in value.lower() for value in values)
