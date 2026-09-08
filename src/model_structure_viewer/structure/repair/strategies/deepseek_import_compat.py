"""Repair strategy: shim the removed ``is_torch_fx_available`` symbol. ALIVE.

Trigger condition (verified end-to-end with the bundled DeepSeek-V3.1 fixture
under transformers 5.16.1, ``repair_status=success``): remote
``modeling_deepseek.py`` starts with ``from transformers.utils.import_utils
import is_torch_fx_available``, but modern transformers no longer exports that
symbol, so the import fails with ``cannot import name 'is_torch_fx_available'``
-> classifier maps it to ``REMOTE_IMPORT_COMPAT`` -> this strategy matches any
DeepSeek-bearing context (model_type / architectures / ``auto_map`` referencing
``modeling_deepseek`` / a local ``modeling_deepseek.py``) and injects a no-op
shim for the introspection retry.

Bundled models that hit this import: deepseek-ai/DeepSeek-R1, DeepSeek-V3.1,
moonshotai Kimi-K2-Base / K2-Instruct / K2-Instruct-0905 / K2-Thinking / K2.5 /
K2.6 (K2.7-Code vendors its own fallback and does not need the shim).

Consumer chain: service.py + verification/transformers_verify.py ->
structure/builder.py -> recovery._recover_with_repair -> repair.runner.try_repair
-> repair.registry (STRATEGIES wired in repair/strategies/__init__.py).
Do not remove while any bundled model ships a pre-5.x ``modeling_deepseek.py``.
"""
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
            and (
                _is_deepseek(context)
                or _uses_deepseek_remote_code(context)
                or _has_deepseek_remote_file(context)
            )
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


def _uses_deepseek_remote_code(context: RepairContext) -> bool:
    auto_map = context.config.get("auto_map") or {}
    if not isinstance(auto_map, dict):
        return False
    values: list[str] = []
    for value in auto_map.values():
        values.extend(value if isinstance(value, list) else [value])
    return any(isinstance(value, str) and "modeling_deepseek" in value for value in values)


def _has_deepseek_remote_file(context: RepairContext) -> bool:
    return context.local_dir is not None and (context.local_dir / "modeling_deepseek.py").exists()
