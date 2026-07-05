"""Centralized field-name tables shared across the structure pipeline.

Keeping these in one place avoids drift between the introspection path,
the frontend structure generator, and the summary aggregator. Each tuple has
a single, well-defined consumer documented below.
"""
from __future__ import annotations

# Used by `summary.extract_summary` for top-level scalar fields.
SUMMARY_TOP_KEYS: tuple[str, ...] = (
    "model_type",
    "torch_dtype",
    "tie_word_embeddings",
    "max_position_embeddings",
)

# Text-tower fields surfaced in the summary panel.
SUMMARY_TEXT_KEYS: tuple[str, ...] = (
    "hidden_size",
    "dim",
    "num_hidden_layers",
    "num_layers",
    "n_layers",
    "num_attention_heads",
    "n_heads",
    "num_key_value_heads",
    "intermediate_size",
    "inter_dim",
    "max_position_embeddings",
    "vocab_size",
    "num_local_experts",
    "num_experts_per_tok",
    "n_routed_experts",
)

# Vision-tower fields surfaced in the summary panel.
SUMMARY_VISION_KEYS: tuple[str, ...] = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "patch_size",
    "image_size",
)

# Used by `semantics.extract_attributes` to grab scalar attributes from
# a live nn.Module / module.config during meta introspection.
MODULE_ATTR_WHITELIST: tuple[str, ...] = (
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

# Keys that should NOT bleed into `extra_config` (they are either trivial
# metadata or already represented elsewhere in the structure tree).
EXTRA_CONFIG_SKIP: frozenset[str] = frozenset({
    "model_type",
    "architectures",
    "auto_map",
    "torch_dtype",
    "transformers_version",
    "text_config",
    "vision_config",
})


def make_extra_config(
    config: dict,
    *,
    extra_skip: tuple[str, ...] = (),
) -> dict:
    """Return a config copy with metadata / nested towers stripped."""
    skip = EXTRA_CONFIG_SKIP | set(extra_skip)
    return {key: value for key, value in config.items() if key not in skip}
