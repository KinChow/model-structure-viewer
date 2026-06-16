"""Shared summary extraction for both introspection and fallback paths."""
from __future__ import annotations

from typing import Any

_TOP_KEYS = (
    "model_type",
    "torch_dtype",
    "tie_word_embeddings",
    "max_position_embeddings",
)

_TEXT_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "vocab_size",
    "num_local_experts",
    "num_experts_per_tok",
    "n_routed_experts",
)

_VISION_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "patch_size",
    "image_size",
)


def extract_summary(
    config: dict[str, Any],
    *,
    model_family: str | None = None,
    confidence: str = "high",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat summary dict from a (possibly nested) HF config dict."""
    summary: dict[str, Any] = {}
    if model_family:
        summary["model_family"] = model_family

    for key in _TOP_KEYS:
        _put(summary, key, config.get(key))

    architecture = _first(config.get("architectures"))
    _put(summary, "architecture", architecture)

    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    if isinstance(text_cfg, dict):
        for key in _TEXT_KEYS:
            target = "text_layers" if key == "num_hidden_layers" else key
            _put(summary, target, text_cfg.get(key))

    vision_cfg = config.get("vision_config")
    if isinstance(vision_cfg, dict):
        for key in _VISION_KEYS:
            target = f"vision_{key}" if key != "num_hidden_layers" else "vision_layers"
            _put(summary, target, vision_cfg.get(key))

    if extra:
        for key, value in extra.items():
            _put(summary, key, value)

    summary["confidence"] = confidence
    return summary


def infer_model_family(config: dict[str, Any]) -> str | None:
    """Best-effort human-friendly family label derived from config metadata."""
    architecture = _first(config.get("architectures"))
    if architecture:
        return architecture
    model_type = config.get("model_type")
    if model_type:
        return str(model_type)
    return None


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is None or key in target:
        return
    target[key] = value


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return None
