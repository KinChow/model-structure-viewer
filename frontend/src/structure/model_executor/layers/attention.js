import { moduleSpec, withShapeDims } from "./base.js";
import { attentionOperatorSpecs, deepseekV4AttentionOperatorSpecs, linearAttentionOperatorSpecs, minimaxDenseAttentionOperatorSpecs, minimaxM2AttentionOperatorSpecs, minimaxSparseAttentionOperatorSpecs, mlaAttentionOperatorSpecs, qsaAttentionOperatorSpecs, qwen35FullAttentionOperatorSpecs } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function attentionModule(id, normalized, attentionKind, layerIndex = 0) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const displayName = attentionKind === "qwen35_full" ? "Qwen3.5 Full Attention" : `${attentionKind.toUpperCase()} Attention`;
  const children = attentionKind === "linear"
    ? linearAttentionOperatorSpecs(id, normalized)
    : attentionKind === "sparse" && normalized.modelType === "minimax_m3_vl"
      ? minimaxSparseAttentionOperatorSpecs(id, normalized, layerIndex)
    : attentionKind === "gqa" && normalized.modelType === "minimax_m3_vl"
      ? minimaxDenseAttentionOperatorSpecs(id, normalized)
    : attentionKind === "gqa" && normalized.modelType === "minimax_m2"
      ? minimaxM2AttentionOperatorSpecs(id, normalized)
    : attentionKind === "gqa" && normalized.modelType === "glm4_moe"
      ? minimaxM2AttentionOperatorSpecs(id, normalized, "glm4_moe")
    : attentionKind === "qwen35_full"
      ? qwen35FullAttentionOperatorSpecs(id, normalized)
    : attentionKind === "qsa"
      ? qsaAttentionOperatorSpecs(id, normalized, layerIndex)
    : attentionKind === "dsv4"
      ? deepseekV4AttentionOperatorSpecs(id, normalized, layerIndex)
    : attentionKind === "mla"
      ? mlaAttentionOperatorSpecs(id, normalized)
      : attentionOperatorSpecs(id, attentionKind, normalized);
  const declaredEdges = attentionKind === "gqa" && normalized.modelType === "minimax_m2"
    ? [["qkv_proj", "qkv_split"], ["qkv_split", "q_norm"], ["qkv_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["rope", "scores"], ["scores", "softmax"], ["softmax", "context"], ["context", "o_proj"]]
    : attentionKind === "gqa" && normalized.modelType === "minimax_m3_vl"
      ? [["qkv_index_proj", "qkv_index_split"], ["qkv_index_split", "q_norm"], ["qkv_index_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["rope", "scores"], ["scores", "softmax"], ["softmax", "context"], ["context", "o_proj"]]
      : attentionKind === "sparse" && normalized.modelType === "minimax_m3_vl"
        ? [["qkv_index_proj", "qkv_index_split"], ["qkv_index_split", "q_norm"], ["qkv_index_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["qkv_index_split", "index_q_norm"], ["qkv_index_split", "index_k_norm"], ["index_q_norm", "index_rope"], ["index_k_norm", "index_rope"], ["index_rope", "indexer"], ["rope", "sparse_attention"], ["indexer", "sparse_attention"], ["sparse_attention", "o_proj"]]
        : attentionKind === "linear" && (String(normalized.modelType).startsWith("qwen3_") || normalized.modelType === "qwen4_exp")
          ? [["qkv_projection", "qkvz_split"], ["qkvz_split", "short_conv"], ["beta_projection", "state_update"], ["decay_projection", "state_update"], ["short_conv", "state_update"], ["state_update", "output_gate_norm"], ["output_gate_norm", "out_proj"]]
          : attentionKind === "linear" && ["kimi_k3", "glm5_next"].includes(normalized.modelType)
            ? [["qkv_projection", "short_conv"], ["beta_projection", "state_update"], ["decay_projection", "state_update"], ["short_conv", "state_update"], ["state_update", "output_gate_norm"], ["output_gate_norm", "out_proj"]]
          : attentionKind === "qwen35_full"
            ? [["qkv_gate_proj", "qkv_gate_split"], ["qkv_gate_split", "q_norm"], ["qkv_gate_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["rope", "scores"], ["scores", "softmax"], ["softmax", "context"], ["context", "output_gate"], ["output_gate", "o_proj"]]
            : attentionKind === "qsa" && ["deepseek_v32", "glm_moe_dsa"].includes(normalized.modelType)
              ? [["q_a_proj", "q_a_norm"], ["q_a_norm", "q_b_proj"], ["kv_a_proj", "kv_split"], ["kv_split", "kv_a_norm"], ["kv_a_norm", "kv_b_proj"], ["q_b_proj", "rope"], ["kv_b_proj", "rope"], ["q_a_norm", "q_proj"], ["q_proj", "indexer"], ["wk_weights_proj", "k_norm"], ["k_norm", "indexer"], ["indexer", "sparse_attention"], ["rope", "sparse_attention"], ["sparse_attention", "o_proj"]]
              : attentionKind === "qsa" && ["glm5_next", "qwen4_exp"].includes(normalized.modelType)
                ? [["qkv_proj", "q_norm"], ["qkv_proj", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["indexer", "sparse_attention"], ["rope", "sparse_attention"], ["sparse_attention", "out_proj"]]
                : attentionKind === "dsv4" && Number(normalized.compressRatios?.[0]) > 0
                  ? [["fused_wqa_wkv", "qkv_split"], ["qkv_split", "q_norm"], ["qkv_split", "kv_norm"], ["q_norm", "q_proj"], ["q_proj", "rope"], ["kv_norm", "rope"], ["compressor", "attention"], ["rope", "attention"], ["attention", "inverse_rope"], ["inverse_rope", "wo_a"], ["wo_a", "wo_b"]]
                  : attentionKind === "dsv4"
                    ? [["fused_wqa_wkv", "qkv_split"], ["qkv_split", "q_norm"], ["qkv_split", "kv_norm"], ["q_norm", "q_proj"], ["q_proj", "rope"], ["kv_norm", "rope"], ["rope", "attention"], ["attention", "inverse_rope"], ["inverse_rope", "wo_a"], ["wo_a", "wo_b"]]
        : attentionKind === "gqa" && !["minimax_m3_vl", "minimax_m2", "glm4_moe"].includes(normalized.modelType)
    ? [["q_proj", "rope"], ["k_proj", "rope"], ["rope", "scores"], ["scores", "softmax"], ["softmax", "context"], ["v_proj", "context"], ["context", "o_proj"]]
    : attentionKind === "mla"
      ? [["q_a_proj", "q_a_norm"], ["q_a_norm", "q_b_proj"], ["kv_a_proj", "kv_split"], ["kv_split", "kv_a_norm"], ["kv_a_norm", "kv_b_proj"], ["q_b_proj", "rope"], ["kv_b_proj", "rope"], ["rope", "scores"], ["scores", "softmax"], ["softmax", "context"], ["context", "o_proj"]]
      : undefined;
  return withShapeDims(moduleSpec(
    id,
    displayName,
    "attention",
    {
      class: `${attentionKind.toUpperCase()}Attention`,
      attention_kind: attentionKind,
      model_variant: normalized.modelType,
      hidden_size: normalized.hiddenSize,
      num_attention_heads: normalized.attentionHeads,
      num_key_value_heads: normalized.kvHeads,
      compress_ratio: attentionKind === "dsv4" ? normalized.compressRatios?.[layerIndex] : undefined,
      attention_variant: attentionKind === "dsv4" ? "deepseek_v4" : undefined,
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        query_shape: shapes.attentionQuery,
        key_shape: shapes.attentionKey,
        value_shape: shapes.attentionValue,
      }),
      dataflow_edges: declaredEdges,
    },
    children,
  ), dims.hidden, dims.hidden);
}
