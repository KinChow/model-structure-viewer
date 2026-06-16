"""Map nn.Module class names to semantic node types and extract attributes."""
from __future__ import annotations

import re
from typing import Any

# Ordered: more specific patterns first. Each tuple is (keywords, node_type).
_TYPE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ModuleList", "ModuleDict", "Sequential"), "module-list"),
    (("PatchEmbed",), "embedding"),
    (("LMHead",), "lm-head"),
    (("MoE", "SparseMoeBlock", "Experts"), "moe"),
    (("VisionTower", "VisionModel", "VisionTransformer", "ClipVisionModel"), "vision-encoder"),
    (("Projector", "Connector", "MultiModalAdapter"), "projector"),
    (("Attention", "Attn"), "attention"),
    (("MLP", "FeedForward", "FFN", "GLU"), "mlp"),
    (("Embedding",), "embedding"),
    (("RMSNorm", "LayerNorm", "GroupNorm"), "normalization"),
    (("Decoder",), "decoder"),
    (("Encoder",), "encoder"),
    (("Conv2d", "Conv1d", "Conv3d"), "conv"),
    (("Linear",), "linear"),
    (("Dropout",), "dropout"),
)

_ATTENTION_KIND = (
    ("MLA", "Multi-head Latent Attention"),
    ("Sparse", "Sparse Attention"),
    ("Flash", "Flash Attention"),
    ("GQA", "Grouped Query Attention"),
    ("MQA", "Multi-Query Attention"),
)

_ATTR_WHITELIST = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_heads",
    "head_dim",
    "intermediate_size",
    "num_local_experts",
    "num_experts_per_tok",
    "n_routed_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    "vocab_size",
    "embedding_dim",
    "embed_dim",
    "patch_size",
    "image_size",
    "rope_theta",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)

_KEYWORD_TO_FAMILY = (
    ("attention", "attention"),
    ("attn", "attention"),
    ("mlp", "mlp"),
    ("feedforward", "mlp"),
    ("moe", "moe"),
    ("embed", "embedding"),
    ("norm", "normalization"),
    ("vision", "vision"),
    ("projector", "projector"),
    ("connector", "projector"),
    ("head", "output"),
)


def classify(module: Any) -> str:
    """Return a semantic node type for a torch nn.Module."""
    class_name = type(module).__name__
    for keywords, node_type in _TYPE_RULES:
        if any(keyword in class_name for keyword in keywords):
            return node_type
    return "module"


def normalized_class(module: Any) -> str:
    """A canonical name used for isomorphism comparison."""
    return type(module).__name__


def attention_kind(class_name: str) -> str | None:
    for keyword, label in _ATTENTION_KIND:
        if keyword in class_name:
            return label
    return None


def extract_attributes(module: Any) -> dict[str, Any]:
    """Surface meaningful scalar attributes from a module / its config."""
    attrs: dict[str, Any] = {}
    config = getattr(module, "config", None)
    if config is not None:
        for key in _ATTR_WHITELIST:
            value = getattr(config, key, None)
            if _is_scalar(value):
                attrs[key] = value
    for key in _ATTR_WHITELIST:
        if key in attrs:
            continue
        value = getattr(module, key, None)
        if _is_scalar(value):
            attrs[key] = value

    class_name = type(module).__name__
    kind = attention_kind(class_name)
    if kind:
        attrs.setdefault("kind", kind)
    return attrs


def display_name(attribute_name: str, module: Any) -> str:
    """Pretty name combining the attribute path and the module class."""
    class_name = type(module).__name__
    if not attribute_name:
        return class_name
    pretty_attr = re.sub(r"[._]", " ", attribute_name).strip()
    return f"{pretty_attr} ({class_name})" if pretty_attr else class_name


def family(node_type: str, class_name: str) -> str:
    """Coarse family used by the front-end to colorize nodes."""
    lowered = class_name.lower()
    for keyword, fam in _KEYWORD_TO_FAMILY:
        if keyword in lowered:
            return fam
    return node_type


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool))
