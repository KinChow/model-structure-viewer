import { moduleSpec, withShapeDims } from "./base.js";
import { attentionOperatorSpecs, deepseekV4AttentionOperatorSpecs, linearAttentionOperatorSpecs, mlaAttentionOperatorSpecs, qsaAttentionOperatorSpecs, qwen35FullAttentionOperatorSpecs } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function attentionModule(id, normalized, attentionKind, layerIndex = 0) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const displayName = attentionKind === "qwen35_full" ? "Qwen3.5 Full Attention" : `${attentionKind.toUpperCase()} Attention`;
  return withShapeDims(moduleSpec(
    id,
    displayName,
    "attention",
    {
      class: `${attentionKind.toUpperCase()}Attention`,
      attention_kind: attentionKind,
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
    },
    attentionKind === "linear"
      ? linearAttentionOperatorSpecs(id, normalized)
      : attentionKind === "qwen35_full"
        ? qwen35FullAttentionOperatorSpecs(id, normalized)
      : attentionKind === "qsa"
        ? qsaAttentionOperatorSpecs(id, normalized)
      : attentionKind === "dsv4"
        ? deepseekV4AttentionOperatorSpecs(id, normalized, layerIndex)
        : attentionKind === "mla"
        ? mlaAttentionOperatorSpecs(id, normalized)
        : attentionOperatorSpecs(id, attentionKind, normalized),
  ), dims.hidden, dims.hidden);
}
