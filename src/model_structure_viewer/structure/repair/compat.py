from __future__ import annotations

from typing import Any

from .runtime import RuntimePatch


def is_flash_attention2_unavailable(error: BaseException) -> bool:
    message = str(error).lower()
    normalized = message.replace(" ", "")
    mentions_flash_attention2 = "flashattention2" in normalized or "flash_attention_2" in normalized
    return mentions_flash_attention2 and (
        "not installed" in message
        or "doesn't seem to be installed" in message
        or "flash_attn" in message
        or "does not support" in message
        or "not support" in message
    )


def is_kimi_tie_weights_signature_error(error: BaseException) -> bool:
    message = str(error)
    return "KimiK25ForConditionalGeneration.tie_weights()" in message and "unexpected keyword argument" in message


class KimiTieWeightsCompatPatch:
    name = "kimi_tie_weights_compat"

    def activate(self):
        return _KimiTieWeightsCompatContext()


class _KimiTieWeightsCompatContext:
    def __init__(self):
        self._original_init_weights = None

    def __enter__(self):
        from transformers.modeling_utils import PreTrainedModel

        self._original_init_weights = PreTrainedModel.init_weights

        def compatible_init_weights(model_self):
            try:
                return self._original_init_weights(model_self)
            except TypeError as exc:
                if not is_kimi_tie_weights_signature_error(exc):
                    raise
                if type(model_self).__name__ != "KimiK25ForConditionalGeneration":
                    raise
                return model_self.tie_weights()

        PreTrainedModel.init_weights = compatible_init_weights
        return None

    def __exit__(self, exc_type, exc, tb):
        if self._original_init_weights is not None:
            from transformers.modeling_utils import PreTrainedModel

            PreTrainedModel.init_weights = self._original_init_weights
            self._original_init_weights = None
        return False


class CompositeRuntimePatch:
    name = "composite_runtime_patch"

    def __init__(self, *patches: RuntimePatch | None):
        self.patches = [patch for patch in patches if patch is not None]

    def activate(self):
        return _CompositeRuntimePatchContext([patch.activate() for patch in self.patches])


class _CompositeRuntimePatchContext:
    def __init__(self, contexts: list[Any]):
        self.contexts = contexts

    def __enter__(self):
        for context in self.contexts:
            context.__enter__()
        return None

    def __exit__(self, exc_type, exc, tb):
        suppress = False
        for context in reversed(self.contexts):
            suppress = bool(context.__exit__(exc_type, exc, tb)) or suppress
        return suppress


class AttentionImplementationNormalizer:
    name = "attention_implementation_normalizer"

    def __init__(self, implementation: str):
        self.implementation = implementation

    def normalize(self, hf_config: Any) -> dict[str, Any]:
        normalized_targets: list[str] = []
        previous_values: dict[str, Any] = {}
        self._normalize_config_object(hf_config, "root", normalized_targets, previous_values, set())
        return {
            "config_normalizer": self.name,
            "attention_backend": self.implementation,
            "normalized_targets": normalized_targets,
            "previous_attention_backends": previous_values,
        }

    def _normalize_config_object(
        self,
        value: Any,
        path: str,
        normalized_targets: list[str],
        previous_values: dict[str, Any],
        seen: set[int],
    ) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        for attr in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
            if hasattr(value, attr):
                previous = getattr(value, attr)
                if _looks_like_flash_attention(previous):
                    setattr(value, attr, self.implementation)
                    target = f"{path}.{attr}"
                    normalized_targets.append(target)
                    previous_values[target] = previous

        for nested_name in (
            "text_config",
            "vision_config",
            "audio_config",
            "encoder_config",
            "decoder_config",
            "llm_config",
        ):
            if hasattr(value, nested_name):
                self._normalize_config_object(
                    getattr(value, nested_name),
                    f"{path}.{nested_name}",
                    normalized_targets,
                    previous_values,
                    seen,
                )


class CompositeConfigNormalizer:
    name = "composite_config_normalizer"

    def __init__(self, *normalizers: Any):
        self.normalizers = [normalizer for normalizer in normalizers if normalizer is not None]

    def normalize(self, hf_config: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {"config_normalizer": self.name}
        applied: list[str] = []
        for normalizer in self.normalizers:
            diagnostics = normalizer.normalize(hf_config)
            applied.append(getattr(normalizer, "name", type(normalizer).__name__))
            for key, value in diagnostics.items():
                if key == "config_normalizer":
                    continue
                if key in merged and isinstance(merged[key], list) and isinstance(value, list):
                    merged[key].extend(value)
                elif key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
        merged["applied_config_normalizers"] = applied
        return merged


def _looks_like_flash_attention(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "flash_attention_2"
