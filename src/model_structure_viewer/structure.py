from __future__ import annotations

from typing import Any

from .schemas import ModelStructure, StructureNode


def build_model_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
    detail_level: str = "compressed",
) -> ModelStructure:
    if is_minimax_m3(config):
        return build_minimax_m3_structure(config, source=source or {}, detail_level=detail_level)
    if is_deepseek_v3(config):
        return build_deepseek_v3_structure(config, source=source or {}, detail_level=detail_level)
    return build_generic_structure(config, source=source or {}, detail_level=detail_level)


def is_minimax_m3(config: dict[str, Any]) -> bool:
    architectures = config.get("architectures") or []
    return config.get("model_type") == "minimax_m3_vl" or "MiniMaxM3SparseForConditionalGeneration" in architectures


def is_deepseek_v3(config: dict[str, Any]) -> bool:
    architectures = config.get("architectures") or []
    return config.get("model_type") == "deepseek_v3" or "DeepseekV3ForCausalLM" in architectures


def build_deepseek_v3_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    detail_level: str,
) -> ModelStructure:
    architecture = _first(config.get("architectures"))
    num_layers = int(config.get("num_hidden_layers") or 0)
    dense_layers = int(config.get("first_k_dense_replace") or 0)
    moe_start = dense_layers
    moe_count = max(num_layers - dense_layers, 0)
    root = StructureNode(
        id="deepseek_v3",
        name="DeepseekV3Model",
        type="decoder-only-model",
        attributes=_clean_attrs(
            {
                "model_type": config.get("model_type"),
                "architecture": architecture,
                "torch_dtype": config.get("torch_dtype"),
                "hidden_size": config.get("hidden_size"),
                "num_hidden_layers": num_layers,
                "max_position_embeddings": config.get("max_position_embeddings"),
            }
        ),
        source_fields=["model_type", "architectures", "hidden_size", "num_hidden_layers"],
        children=[
            StructureNode(
                id="deepseek.embedding",
                name="Embedding (embed_tokens)",
                type="embedding",
                attributes=_clean_attrs(
                    {
                        "vocab_size": config.get("vocab_size"),
                        "hidden_size": config.get("hidden_size"),
                        "shape": _shape(config.get("vocab_size"), config.get("hidden_size")),
                    }
                ),
                source_fields=["vocab_size", "hidden_size"],
            ),
            StructureNode(
                id="deepseek.layers",
                name="ModuleList (layers)",
                type="module-list",
                repeat=num_layers or None,
                attributes=_clean_attrs({"num_hidden_layers": num_layers}),
                source_fields=["num_hidden_layers"],
                children=_deepseek_layer_groups(config, dense_layers, moe_start, moe_count),
            ),
            StructureNode(
                id="deepseek.output",
                name="MTP and LM Head",
                type="output-heads",
                attributes=_clean_attrs(
                    {
                        "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
                        "vocab_size": config.get("vocab_size"),
                        "tie_word_embeddings": config.get("tie_word_embeddings"),
                    }
                ),
                source_fields=["num_nextn_predict_layers", "vocab_size", "tie_word_embeddings"],
                children=[
                    StructureNode(
                        id="deepseek.output.mtp",
                        name="Next-token Prediction Layers",
                        type="mtp",
                        repeat=config.get("num_nextn_predict_layers"),
                        source_fields=["num_nextn_predict_layers"],
                    ),
                    StructureNode(
                        id="deepseek.output.lm_head",
                        name="Language Modeling Head",
                        type="lm-head",
                        attributes=_clean_attrs(
                            {
                                "vocab_size": config.get("vocab_size"),
                                "hidden_size": config.get("hidden_size"),
                                "shape": _shape(config.get("hidden_size"), config.get("vocab_size")),
                            }
                        ),
                        source_fields=["vocab_size", "hidden_size"],
                    ),
                ],
            ),
        ],
    )
    return ModelStructure(
        summary=_clean_attrs(
            {
                "model_family": "DeepSeek-V3",
                "model_type": config.get("model_type"),
                "architecture": architecture,
                "pipeline": "text-generation",
                "text_layers": num_layers,
                "hidden_size": config.get("hidden_size"),
                "num_attention_heads": config.get("num_attention_heads"),
                "num_key_value_heads": config.get("num_key_value_heads"),
                "max_position_embeddings": config.get("max_position_embeddings"),
                "num_local_experts": config.get("n_routed_experts"),
                "num_experts_per_tok": config.get("num_experts_per_tok"),
                "dense_layers": dense_layers,
                "moe_layers": moe_count,
            }
        ),
        source=source,
        root=root,
        extra_config=_extra_config(config, known=set()),
    )


def _deepseek_layer_groups(
    config: dict[str, Any],
    dense_layers: int,
    moe_start: int,
    moe_count: int,
) -> list[StructureNode]:
    groups: list[StructureNode] = []
    if dense_layers > 0:
        groups.append(
            StructureNode(
                id="deepseek.layers.dense",
                name=f"DeepseekV3DecoderLayer 0-{dense_layers - 1}",
                type="layer-group",
                repeat=dense_layers,
                attributes=_clean_attrs({"range": f"0-{dense_layers - 1}", "moe": False}),
                source_fields=["first_k_dense_replace"],
                children=_deepseek_layer_children(config, is_moe=False),
            )
        )
    if moe_count > 0:
        groups.append(
            StructureNode(
                id="deepseek.layers.moe",
                name=f"DeepseekV3DecoderLayer {moe_start}-{moe_start + moe_count - 1}",
                type="layer-group",
                repeat=moe_count,
                attributes=_clean_attrs(
                    {
                        "range": f"{moe_start}-{moe_start + moe_count - 1}",
                        "moe": True,
                        "moe_layer_freq": config.get("moe_layer_freq"),
                    }
                ),
                source_fields=["num_hidden_layers", "first_k_dense_replace", "moe_layer_freq"],
                children=_deepseek_layer_children(config, is_moe=True),
            )
        )
    return groups


def _deepseek_layer_children(config: dict[str, Any], *, is_moe: bool) -> list[StructureNode]:
    ffn = (
        StructureNode(
            id="deepseek.layer.moe",
            name="DeepseekV3MoE",
            type="moe",
            attributes=_clean_attrs(
                {
                    "routed_experts": config.get("n_routed_experts"),
                    "experts_per_token": config.get("num_experts_per_tok"),
                    "shared_experts": config.get("n_shared_experts"),
                    "expert_intermediate_size": config.get("moe_intermediate_size"),
                    "topk_group": config.get("topk_group"),
                    "n_group": config.get("n_group"),
                    "topk_method": config.get("topk_method"),
                    "scoring_func": config.get("scoring_func"),
                    "routed_scaling_factor": config.get("routed_scaling_factor"),
                }
            ),
            source_fields=[
                "n_routed_experts",
                "num_experts_per_tok",
                "n_shared_experts",
                "moe_intermediate_size",
                "topk_method",
            ],
        )
        if is_moe
        else StructureNode(
            id="deepseek.layer.mlp",
            name="DeepseekV3MLP",
            type="mlp",
            attributes=_clean_attrs(
                {
                    "intermediate_size": config.get("intermediate_size"),
                    "activation": config.get("hidden_act"),
                }
            ),
            source_fields=["intermediate_size", "hidden_act"],
        )
    )
    return [
        StructureNode(
            id="deepseek.layer.attention",
            name="DeepseekV3 MLA Attention",
            type="attention",
            attributes=_clean_attrs(
                {
                    "num_attention_heads": config.get("num_attention_heads"),
                    "num_key_value_heads": config.get("num_key_value_heads"),
                    "q_lora_rank": config.get("q_lora_rank"),
                    "kv_lora_rank": config.get("kv_lora_rank"),
                    "qk_nope_head_dim": config.get("qk_nope_head_dim"),
                    "qk_rope_head_dim": config.get("qk_rope_head_dim"),
                    "v_head_dim": config.get("v_head_dim"),
                    "attention_bias": config.get("attention_bias"),
                    "attention_dropout": config.get("attention_dropout"),
                }
            ),
            source_fields=[
                "num_attention_heads",
                "num_key_value_heads",
                "q_lora_rank",
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
            ],
        ),
        ffn,
        StructureNode(
            id="deepseek.layer.norms",
            name="RMSNorms",
            type="normalization",
            attributes=_clean_attrs({"rms_norm_eps": config.get("rms_norm_eps")}),
            source_fields=["rms_norm_eps"],
        ),
    ]


def build_minimax_m3_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    detail_level: str,
) -> ModelStructure:
    text = config.get("text_config") or {}
    vision = config.get("vision_config") or {}
    sparse = text.get("sparse_attention_config") or {}
    architecture = _first(config.get("architectures"))

    root = StructureNode(
        id="minimax_m3",
        name="MiniMaxM3 VL Wrapper",
        type="vl-wrapper",
        attributes=_clean_attrs(
            {
                "model_type": config.get("model_type"),
                "architecture": architecture,
                "torch_dtype": config.get("torch_dtype"),
                "image_token_index": config.get("image_token_index"),
                "video_token_index": config.get("video_token_index"),
                "image_seq_length": config.get("image_seq_length"),
                "process_image_mode": config.get("process_image_mode"),
                "vision_feature_layer": config.get("vision_feature_layer"),
                "vision_feature_select_strategy": config.get("vision_feature_select_strategy"),
            }
        ),
        source_fields=[
            "model_type",
            "architectures",
            "image_token_index",
            "video_token_index",
            "image_seq_length",
            "process_image_mode",
        ],
        children=[
            _minimax_vision_node(vision),
            _minimax_projector_node(config, vision, text),
            _minimax_text_node(text, sparse, detail_level),
            _minimax_output_node(text),
        ],
    )

    summary = _clean_attrs(
        {
            "model_family": "MiniMax-M3",
            "model_type": config.get("model_type"),
            "architecture": architecture,
            "pipeline": "image-text-to-text",
            "text_layers": text.get("num_hidden_layers"),
            "vision_layers": vision.get("num_hidden_layers"),
            "hidden_size": text.get("hidden_size"),
            "vision_hidden_size": vision.get("hidden_size"),
            "num_attention_heads": text.get("num_attention_heads"),
            "num_key_value_heads": text.get("num_key_value_heads"),
            "max_position_embeddings": text.get("max_position_embeddings"),
            "num_local_experts": text.get("num_local_experts"),
            "num_experts_per_tok": text.get("num_experts_per_tok"),
            "activated_parameters_note": "Config-derived structure only; weights are not loaded.",
        }
    )
    return ModelStructure(
        summary=summary,
        source=source,
        root=root,
        extra_config=_extra_config(config, known={"text_config", "vision_config"}),
    )


def _minimax_vision_node(vision: dict[str, Any]) -> StructureNode:
    compression = vision.get("img_token_compression_config") or {}
    return StructureNode(
        id="vision",
        name="Vision Tower",
        type="vision-encoder",
        attributes=_clean_attrs(
            {
                "model_type": vision.get("model_type", "clip_vision_model"),
                "hidden_size": vision.get("hidden_size"),
                "num_hidden_layers": vision.get("num_hidden_layers"),
                "num_attention_heads": vision.get("num_attention_heads"),
                "intermediate_size": vision.get("intermediate_size"),
                "patch_size": vision.get("patch_size"),
                "image_size": vision.get("image_size"),
                "projection_dim": vision.get("projection_dim"),
                "position_embedding_type": vision.get("position_embedding_type"),
                "rope_mode": vision.get("rope_mode"),
                "vision_segment_max_frames": vision.get("vision_segment_max_frames"),
            }
        ),
        source_fields=["vision_config"],
        children=[
            StructureNode(
                id="vision.patch_embedding",
                name="Patch Embedding",
                type="embedding",
                attributes=_clean_attrs(
                    {
                        "num_channels": vision.get("num_channels"),
                        "patch_size": vision.get("patch_size"),
                        "image_size": vision.get("image_size"),
                    }
                ),
                source_fields=["vision_config.patch_size", "vision_config.image_size"],
            ),
            StructureNode(
                id="vision.blocks",
                name="Vision Transformer Blocks",
                type="layer-group",
                repeat=vision.get("num_hidden_layers"),
                attributes=_clean_attrs(
                    {
                        "hidden_size": vision.get("hidden_size"),
                        "attention_heads": vision.get("num_attention_heads"),
                        "mlp_intermediate_size": vision.get("intermediate_size"),
                        "activation": vision.get("hidden_act"),
                    }
                ),
                source_fields=["vision_config.num_hidden_layers"],
                children=[
                    StructureNode(
                        id="vision.blocks.attention",
                        name="Self Attention",
                        type="attention",
                        attributes=_clean_attrs(
                            {
                                "heads": vision.get("num_attention_heads"),
                                "position_embedding_type": vision.get("position_embedding_type"),
                                "rope_mode": vision.get("rope_mode"),
                                "rope_theta": vision.get("rope_theta"),
                            }
                        ),
                        source_fields=["vision_config.num_attention_heads", "vision_config.rope_mode"],
                    ),
                    StructureNode(
                        id="vision.blocks.mlp",
                        name="Vision MLP",
                        type="mlp",
                        attributes=_clean_attrs(
                            {
                                "intermediate_size": vision.get("intermediate_size"),
                                "activation": vision.get("hidden_act"),
                            }
                        ),
                        source_fields=["vision_config.intermediate_size", "vision_config.hidden_act"],
                    ),
                ],
            ),
            StructureNode(
                id="vision.token_compression",
                name="Image Token Compression",
                type="token-compression",
                attributes=_clean_attrs(
                    {
                        "method": compression.get("image_token_compression_method"),
                        "spatial_merge_size": compression.get("spatial_merge_size"),
                        "temporal_patch_size": compression.get("temporal_patch_size"),
                    }
                ),
                source_fields=["vision_config.img_token_compression_config"],
            ),
        ],
    )


def _minimax_projector_node(
    config: dict[str, Any],
    vision: dict[str, Any],
    text: dict[str, Any],
) -> StructureNode:
    return StructureNode(
        id="projector",
        name="Multimodal Projector",
        type="projector",
        attributes=_clean_attrs(
            {
                "input_dim": vision.get("projection_dim") or vision.get("hidden_size"),
                "output_dim": text.get("hidden_size"),
                "projector_hidden_size": config.get("projector_hidden_size"),
                "activation": config.get("projector_hidden_act"),
                "bias": config.get("multimodal_projector_bias"),
                "vision_feature_layer": config.get("vision_feature_layer"),
                "vision_feature_select_strategy": config.get("vision_feature_select_strategy"),
            }
        ),
        source_fields=[
            "projector_hidden_size",
            "projector_hidden_act",
            "multimodal_projector_bias",
            "vision_config.projection_dim",
            "text_config.hidden_size",
        ],
    )


def _minimax_text_node(
    text: dict[str, Any],
    sparse: dict[str, Any],
    detail_level: str,
) -> StructureNode:
    children = [
        StructureNode(
            id="text.embedding",
            name="Token Embedding",
            type="embedding",
            attributes=_clean_attrs(
                {
                    "vocab_size": text.get("vocab_size"),
                    "hidden_size": text.get("hidden_size"),
                    "tie_word_embeddings": text.get("tie_word_embeddings"),
                }
            ),
            source_fields=["text_config.vocab_size", "text_config.hidden_size"],
        )
    ]
    layer_groups = _layer_groups(text, sparse)
    if detail_level == "expanded":
        children.extend(_expanded_layers(layer_groups))
    else:
        children.extend(_compressed_layer_nodes(layer_groups, text, sparse))

    return StructureNode(
        id="text",
        name="Text Backbone",
        type="decoder",
        attributes=_clean_attrs(
            {
                "architecture": _first(text.get("architectures")),
                "hidden_size": text.get("hidden_size"),
                "num_hidden_layers": text.get("num_hidden_layers"),
                "num_attention_heads": text.get("num_attention_heads"),
                "num_key_value_heads": text.get("num_key_value_heads"),
                "head_dim": text.get("head_dim"),
                "max_position_embeddings": text.get("max_position_embeddings"),
                "rope_theta": text.get("rope_theta"),
                "rotary_dim": text.get("rotary_dim"),
                "partial_rotary_factor": text.get("partial_rotary_factor"),
                "qk_norm_type": text.get("qk_norm_type"),
                "use_qk_norm": text.get("use_qk_norm"),
                "normalization": "Gemma RMSNorm" if text.get("use_gemma_norm") else "RMSNorm",
            }
        ),
        source_fields=["text_config"],
        children=children,
    )


def _compressed_layer_nodes(
    groups: list[dict[str, Any]],
    text: dict[str, Any],
    sparse: dict[str, Any],
) -> list[StructureNode]:
    nodes: list[StructureNode] = []
    for index, group in enumerate(groups):
        prefix = "Dense Warmup Layers" if not group["moe"] else "Sparse MoE Layers"
        label = f"{prefix} {group['start']}-{group['end']}"
        nodes.append(
            StructureNode(
                id=f"text.layers.group{index}",
                name=label,
                type="layer-group",
                repeat=group["count"],
                attributes=_clean_attrs(
                    {
                        "range": f"{group['start']}-{group['end']}",
                        "moe": group["moe"],
                        "sparse_attention": group["sparse_attention"],
                        "hidden_size": text.get("hidden_size"),
                    }
                ),
                source_fields=["text_config.moe_layer_freq", "text_config.sparse_attention_config.sparse_attention_freq"],
                children=_layer_children(group["moe"], group["sparse_attention"], text, sparse),
            )
        )
    return nodes


def _expanded_layers(groups: list[dict[str, Any]]) -> list[StructureNode]:
    nodes: list[StructureNode] = []
    for group in groups:
        for layer_index in range(group["start"], group["end"] + 1):
            nodes.append(
                StructureNode(
                    id=f"text.layers.{layer_index}",
                    name=f"Text Layer {layer_index}",
                    type="decoder-layer",
                    attributes={"moe": group["moe"], "sparse_attention": group["sparse_attention"]},
                    source_fields=["text_config.moe_layer_freq", "text_config.sparse_attention_config.sparse_attention_freq"],
                )
            )
    return nodes


def _layer_children(
    is_moe: bool,
    is_sparse_attention: bool,
    text: dict[str, Any],
    sparse: dict[str, Any],
) -> list[StructureNode]:
    attention_attrs = {
        "kind": "MiniMax Sparse Attention" if is_sparse_attention else "GQA attention",
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "block_size": sparse.get("sparse_block_size") if is_sparse_attention else None,
        "topk_blocks": sparse.get("sparse_topk_blocks") if is_sparse_attention else None,
        "index_heads": sparse.get("sparse_num_index_heads") if is_sparse_attention else None,
        "index_dim": sparse.get("sparse_index_dim") if is_sparse_attention else None,
        "score_type": sparse.get("sparse_score_type") if is_sparse_attention else None,
    }
    ffn_attrs = (
        {
            "num_local_experts": text.get("num_local_experts"),
            "num_experts_per_token": text.get("num_experts_per_tok"),
            "shared_experts": text.get("n_shared_experts"),
            "expert_intermediate_size": text.get("intermediate_size"),
            "shared_intermediate_size": text.get("shared_intermediate_size"),
            "scoring_func": text.get("scoring_func"),
            "use_routing_bias": text.get("use_routing_bias"),
            "routed_scaling_factor": text.get("routed_scaling_factor"),
            "activation": text.get("hidden_act"),
        }
        if is_moe
        else {
            "intermediate_size": text.get("dense_intermediate_size") or text.get("intermediate_size"),
            "activation": text.get("hidden_act"),
        }
    )
    return [
        StructureNode(
            id="text.layer.attention",
            name="MiniMax Sparse Attention" if is_sparse_attention else "Grouped Query Attention",
            type="attention",
            attributes=_clean_attrs(attention_attrs),
            source_fields=["text_config.sparse_attention_config", "text_config.num_key_value_heads"],
        ),
        StructureNode(
            id="text.layer.ffn",
            name="MoE Feed Forward" if is_moe else "Dense Feed Forward",
            type="moe" if is_moe else "mlp",
            attributes=_clean_attrs(ffn_attrs),
            source_fields=["text_config.num_local_experts", "text_config.num_experts_per_tok", "text_config.dense_intermediate_size"],
        ),
        StructureNode(
            id="text.layer.norms",
            name="Layer Norms",
            type="normalization",
            attributes=_clean_attrs(
                {
                    "rms_norm_eps": text.get("rms_norm_eps"),
                    "use_gemma_norm": text.get("use_gemma_norm"),
                    "use_qk_norm": text.get("use_qk_norm"),
                    "qk_norm_type": text.get("qk_norm_type"),
                }
            ),
            source_fields=["text_config.rms_norm_eps", "text_config.use_qk_norm"],
        ),
    ]


def _minimax_output_node(text: dict[str, Any]) -> StructureNode:
    return StructureNode(
        id="output",
        name="MTP and Output Heads",
        type="output-heads",
        attributes=_clean_attrs(
            {
                "num_mtp_modules": text.get("num_mtp_modules"),
                "num_nextn_predict_layers": text.get("num_nextn_predict_layers"),
                "vocab_size": text.get("vocab_size"),
                "tie_word_embeddings": text.get("tie_word_embeddings"),
            }
        ),
        source_fields=["text_config.num_mtp_modules", "text_config.num_nextn_predict_layers"],
        children=[
            StructureNode(
                id="output.mtp",
                name="Multi-Token Prediction Modules",
                type="mtp",
                repeat=text.get("num_mtp_modules"),
                attributes=_clean_attrs({"nextn_predict_layers": text.get("num_nextn_predict_layers")}),
                source_fields=["text_config.num_mtp_modules"],
            ),
            StructureNode(
                id="output.lm_head",
                name="Language Modeling Head",
                type="lm-head",
                attributes=_clean_attrs({"vocab_size": text.get("vocab_size"), "hidden_size": text.get("hidden_size")}),
                source_fields=["text_config.vocab_size"],
            ),
        ],
    )


def _layer_groups(text: dict[str, Any], sparse: dict[str, Any]) -> list[dict[str, Any]]:
    num_layers = int(text.get("num_hidden_layers") or 0)
    moe_freq = _normalized_flags(text.get("moe_layer_freq"), num_layers, default=bool(text.get("num_local_experts")))
    sparse_freq = _normalized_flags(sparse.get("sparse_attention_freq"), num_layers, default=bool(sparse.get("use_sparse_attention")))
    groups: list[dict[str, Any]] = []
    start = 0
    while start < num_layers:
        moe = moe_freq[start]
        is_sparse = sparse_freq[start]
        end = start
        while end + 1 < num_layers and moe_freq[end + 1] == moe and sparse_freq[end + 1] == is_sparse:
            end += 1
        groups.append(
            {
                "start": start,
                "end": end,
                "count": end - start + 1,
                "moe": moe,
                "sparse_attention": is_sparse,
            }
        )
        start = end + 1
    return groups


def _normalized_flags(value: Any, length: int, *, default: bool) -> list[bool]:
    if length <= 0:
        return []
    if isinstance(value, list):
        flags = [bool(item) for item in value[:length]]
        if len(flags) < length:
            flags.extend([default] * (length - len(flags)))
        return flags
    return [default] * length


def build_generic_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    detail_level: str = "compressed",
) -> ModelStructure:
    model_type = config.get("model_type", "unknown")
    architectures = config.get("architectures") or []
    root_children: list[StructureNode] = []

    if isinstance(config.get("vision_config"), dict):
        vision = config["vision_config"]
        root_children.append(
            StructureNode(
                id="vision",
                name="Vision Config",
                type="vision-encoder",
                repeat=vision.get("num_hidden_layers"),
                attributes=_standard_config_attrs(vision),
                source_fields=["vision_config"],
                confidence="medium",
            )
        )

    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    if isinstance(text_config, dict):
        root_children.append(
            StructureNode(
                id="text",
                name="Text Transformer",
                type="transformer",
                repeat=text_config.get("num_hidden_layers") or text_config.get("num_layers"),
                attributes=_standard_config_attrs(text_config),
                source_fields=["text_config" if text_config is not config else "config"],
                confidence="medium" if text_config is not config else "low",
            )
        )

    if not root_children:
        root_children.append(
            StructureNode(
                id="config",
                name="Configuration",
                type="config",
                attributes=_standard_config_attrs(config),
                confidence="low",
            )
        )

    root = StructureNode(
        id="model",
        name=str(model_type),
        type="generic-model",
        attributes=_clean_attrs({"model_type": model_type, "architectures": architectures}),
        confidence="low",
        children=root_children,
    )
    return ModelStructure(
        summary=_clean_attrs(
            {
                "model_family": "generic",
                "model_type": model_type,
                "architecture": _first(architectures),
                "confidence": "low",
                "note": "Generic config-derived structure; add a template for higher fidelity.",
            }
        ),
        source=source,
        root=root,
        extra_config=_extra_config(config, known={"text_config", "vision_config"}),
    )


def _standard_config_attrs(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
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
    ]
    return _clean_attrs({key: config.get(key) for key in keys})


def _extra_config(config: dict[str, Any], *, known: set[str]) -> dict[str, Any]:
    common = {
        "model_type",
        "architectures",
        "auto_map",
        "torch_dtype",
        "transformers_version",
    }
    return {key: value for key, value in config.items() if key not in known | common}


def _clean_attrs(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return None


def _shape(*dims: Any) -> str | None:
    if any(dim is None for dim in dims):
        return None
    return " x ".join(str(dim) for dim in dims)
