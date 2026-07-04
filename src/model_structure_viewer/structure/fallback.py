"""Config-only fallback path (Plan C). Used when meta introspection fails."""
from __future__ import annotations

import logging
from typing import Any

from ..schemas import ModelStructure, StructureNode
from .keys import FALLBACK_HIGHLIGHT_KEYS, make_extra_config
from .summary import extract_summary, infer_model_family

_LOG = logging.getLogger(__name__)

_MAX_NESTED_CONFIG_DEPTH = 8
_LAYER_COUNT_KEYS = ("num_hidden_layers", "num_layers", "n_layer", "n_layers")

_KNOWN_NESTED = {
    "text_config": ("text", "decoder"),
    "vision_config": ("vision", "vision-encoder"),
    "audio_config": ("audio", "audio-encoder"),
    "speech_config": ("speech", "audio-encoder"),
    "encoder_config": ("encoder", "encoder"),
    "decoder_config": ("decoder", "decoder"),
}


def build_from_config(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    fallback_reason: str | None = None,
    fallback_strategy: str = "config-fallback",
    diagnostics: dict[str, Any] | None = None,
) -> ModelStructure:
    """Build a low-confidence ModelStructure purely from a config dict."""
    family = infer_model_family(config) or "generic"
    children = _build_children(config, depth=1)
    if not children:
        decoder = _top_level_decoder(config)
        if decoder is not None:
            children.append(decoder)
        else:
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
    if diagnostics:
        merged_diagnostics = dict(enriched_source.get("diagnostics") or {})
        merged_diagnostics.update(diagnostics)
        enriched_source["diagnostics"] = merged_diagnostics
    enriched_source["strategy"] = fallback_strategy

    if fallback_reason:
        _LOG.info("Config-fallback structure built (reason=%s)", fallback_reason)

    return ModelStructure(
        summary=extract_summary(
            config,
            model_family=family,
            confidence="low",
            extra={"note": "Config-derived structure; introspection unavailable."},
            strategy=fallback_strategy,
            fallback_reason=fallback_reason,
        ),
        source=enriched_source,
        root=root,
        extra_config=make_extra_config(config),
    )


def _build_children(config: dict[str, Any], *, depth: int) -> list[StructureNode]:
    if depth > _MAX_NESTED_CONFIG_DEPTH:
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
                    children=_nested_children(nested, depth=depth + 1),
                )
            )
    return children


def _nested_children(config: dict[str, Any], *, depth: int) -> list[StructureNode]:
    children = _build_children(config, depth=depth)
    decoder = _top_level_decoder(config)
    if decoder is not None:
        children.append(decoder)
    return children


def _top_level_decoder(config: dict[str, Any]) -> StructureNode | None:
    layer_count = next((config[key] for key in _LAYER_COUNT_KEYS if config.get(key) is not None), None)
    if layer_count is None:
        return None
    return StructureNode(
        id="decoder",
        name="Decoder Layers",
        type="decoder",
        repeat=layer_count,
        attributes=_pick(config),
        source_fields=[key for key in _LAYER_COUNT_KEYS if config.get(key) is not None],
        confidence="low",
        children=[_decoder_layer_placeholder(config, layer_count)],
    )


def _decoder_layer_placeholder(config: dict[str, Any], layer_count: Any) -> StructureNode:
    layer_count_int = layer_count if isinstance(layer_count, int) and layer_count > 0 else 1
    class_name = _decoder_layer_class(config)
    return StructureNode(
        id="decoder.0",
        name=f"0 ({class_name}) x{layer_count_int}",
        type="layer-group",
        repeat=layer_count_int,
        attributes={**_pick(config), "class": class_name, "range": f"0..{layer_count_int - 1}"},
        source_fields=[key for key in _LAYER_COUNT_KEYS if config.get(key) is not None],
        confidence="low",
    )


def _decoder_layer_class(config: dict[str, Any]) -> str:
    architectures = config.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
    if isinstance(architecture, str) and architecture.endswith("ForCausalLM"):
        return f"{architecture.removesuffix('ForCausalLM')}DecoderLayer"
    model_type = str(config.get("model_type") or "config")
    parts = [part for part in model_type.replace("-", "_").split("_") if part]
    prefix = "".join(part.capitalize() for part in parts) or "Config"
    return f"{prefix}DecoderLayer"


def _pick(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in FALLBACK_HIGHLIGHT_KEYS if config.get(key) is not None}


def _leaf(node_id: str, name: str, node_type: str, config: dict[str, Any]) -> StructureNode:
    return StructureNode(
        id=node_id,
        name=name,
        type=node_type,
        attributes=_pick(config),
        confidence="low",
    )
