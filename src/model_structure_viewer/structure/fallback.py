"""Config-only fallback path (Plan C). Used when meta introspection fails."""
from __future__ import annotations

from typing import Any

from ..schemas import ModelStructure, StructureNode
from .summary import extract_summary, infer_model_family

_MAX_DEPTH = 8

_KNOWN_NESTED = {
    "text_config": ("text", "decoder"),
    "vision_config": ("vision", "vision-encoder"),
    "audio_config": ("audio", "audio-encoder"),
    "speech_config": ("speech", "audio-encoder"),
    "encoder_config": ("encoder", "encoder"),
    "decoder_config": ("decoder", "decoder"),
}

_HIGHLIGHT_KEYS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "tie_word_embeddings",
    "hidden_act",
    "patch_size",
    "image_size",
)


def build_from_config(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    fallback_reason: str | None = None,
) -> ModelStructure:
    """Build a low-confidence ModelStructure purely from a config dict."""
    family = infer_model_family(config) or "generic"
    children = _build_children(config, depth=1)
    if not children:
        children.append(_leaf("config", "Configuration", "config", config))

    root = StructureNode(
        id="model",
        name=family,
        type="generic-model",
        attributes=_pick(config),
        confidence="low",
        source_fields=["model_type", "architectures"],
        children=children,
    )

    enriched_source = dict(source)
    if fallback_reason:
        enriched_source["fallback_reason"] = fallback_reason
    enriched_source.setdefault("strategy", "config-fallback")

    return ModelStructure(
        summary=extract_summary(
            config,
            model_family=family,
            confidence="low",
            extra={"note": "Config-derived structure; introspection unavailable."},
        ),
        source=enriched_source,
        root=root,
        extra_config=_extra_config(config),
    )


def _build_children(config: dict[str, Any], *, depth: int) -> list[StructureNode]:
    if depth > _MAX_DEPTH:
        return []
    children: list[StructureNode] = []
    for key, (node_id, node_type) in _KNOWN_NESTED.items():
        nested = config.get(key)
        if isinstance(nested, dict):
            children.append(
                StructureNode(
                    id=node_id,
                    name=key.replace("_", " ").title(),
                    type=node_type,
                    repeat=nested.get("num_hidden_layers") or nested.get("num_layers"),
                    attributes=_pick(nested),
                    source_fields=[key],
                    confidence="low",
                    children=_build_children(nested, depth=depth + 1),
                )
            )
    return children


def _pick(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in _HIGHLIGHT_KEYS if config.get(key) is not None}


def _leaf(node_id: str, name: str, node_type: str, config: dict[str, Any]) -> StructureNode:
    return StructureNode(
        id=node_id,
        name=name,
        type=node_type,
        attributes=_pick(config),
        confidence="low",
    )


def _extra_config(config: dict[str, Any]) -> dict[str, Any]:
    skip = set(_KNOWN_NESTED) | {
        "model_type",
        "architectures",
        "auto_map",
        "torch_dtype",
        "transformers_version",
    }
    return {key: value for key, value in config.items() if key not in skip}
