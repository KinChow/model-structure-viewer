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
  const declaredEdges = attentionKind === "gqa" && !["minimax_m3_vl", "minimax_m2", "glm4_moe"].includes(normalized.modelType)
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
