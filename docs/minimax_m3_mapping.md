# MiniMax-M3 Mapping

Model: `MiniMaxAI/MiniMax-M3`

MVP recognition:

- `model_type == "minimax_m3_vl"`
- or `architectures` contains `MiniMaxM3SparseForConditionalGeneration`

## Top Level

| Structure Node | Source Fields |
|---|---|
| MiniMaxM3 VL Wrapper | `model_type`, `architectures`, `image_token_index`, `video_token_index`, `image_seq_length`, `process_image_mode` |
| Pipeline summary | Model card `pipeline_tag=image-text-to-text`; config-driven output marks this as image-text-to-text |

## Vision Tower

| Structure Node | Source Fields |
|---|---|
| Vision Tower | `vision_config.model_type`, `vision_config.num_hidden_layers`, `vision_config.hidden_size` |
| Patch Embedding | `vision_config.patch_size`, `vision_config.image_size`, `vision_config.num_channels` |
| Vision Transformer Blocks | `vision_config.num_hidden_layers=32`, `vision_config.num_attention_heads=16`, `vision_config.intermediate_size=5120` |
| 3D RoPE | `vision_config.position_embedding_type`, `vision_config.rope_mode`, `vision_config.rope_theta` |
| Image Token Compression | `vision_config.img_token_compression_config.image_token_compression_method`, `spatial_merge_size`, `temporal_patch_size` |

## Projector

| Structure Node | Source Fields |
|---|---|
| Multimodal Projector | `vision_config.projection_dim`, `text_config.hidden_size`, `projector_hidden_size`, `projector_hidden_act`, `multimodal_projector_bias` |
| Feature selection | `vision_feature_layer`, `vision_feature_select_strategy` |

## Text Backbone

| Structure Node | Source Fields |
|---|---|
| Text Backbone | `text_config.architectures`, `text_config.num_hidden_layers=60`, `text_config.hidden_size=6144` |
| Attention shape | `text_config.num_attention_heads=64`, `text_config.num_key_value_heads=4`, `text_config.head_dim=128` |
| Context | `text_config.max_position_embeddings=1048576` |
| RoPE | `text_config.rope_theta`, `text_config.rotary_dim`, `text_config.partial_rotary_factor` |
| Norms | `text_config.rms_norm_eps`, `text_config.use_gemma_norm`, `text_config.use_qk_norm`, `text_config.qk_norm_type` |

## Layer Groups

The MVP defaults to compressed layer groups.

| Structure Node | Source Fields |
|---|---|
| Dense Warmup Layers 0-2 | `text_config.moe_layer_freq[0:3] == 0`, `text_config.sparse_attention_config.sparse_attention_freq[0:3] == 0` |
| Sparse MoE Layers 3-59 | `text_config.moe_layer_freq[3:] == 1`, `text_config.sparse_attention_config.sparse_attention_freq[3:] == 1` |

## MoE

| Structure Node | Source Fields |
|---|---|
| MoE Feed Forward | `text_config.num_local_experts=128`, `text_config.num_experts_per_tok=4`, `text_config.n_shared_experts=1` |
| Router | `text_config.scoring_func=sigmoid`, `text_config.use_routing_bias=true`, `text_config.routed_scaling_factor=2.0` |
| Expert MLP | `text_config.intermediate_size`, `text_config.shared_intermediate_size`, `text_config.hidden_act` |

## MiniMax Sparse Attention

| Structure Node | Source Fields |
|---|---|
| MiniMax Sparse Attention | `text_config.sparse_attention_config.use_sparse_attention=true` |
| Sparse index | `sparse_index_dim=128`, `sparse_num_index_heads=4` |
| Sparse blocks | `sparse_block_size=128`, `sparse_topk_blocks=16`, `sparse_local_block`, `sparse_init_block` |

## Output Heads

| Structure Node | Source Fields |
|---|---|
| Multi-Token Prediction Modules | `text_config.num_mtp_modules=7`, `text_config.num_nextn_predict_layers=1` |
| LM Head | `text_config.vocab_size`, `text_config.tie_word_embeddings` |
