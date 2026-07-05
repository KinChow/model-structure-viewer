import { moduleSpec } from "./base.js";
import { attentionOperatorSpecs } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function attentionModule(id, normalized, attentionKind) {
  const shapes = tensorShapes(normalized);
  return moduleSpec(
    id,
    `${attentionKind.toUpperCase()} Attention`,
    "attention",
    {
      class: `${attentionKind.toUpperCase()}Attention`,
      attention_kind: attentionKind,
      hidden_size: normalized.hiddenSize,
      num_attention_heads: normalized.attentionHeads,
      num_key_value_heads: normalized.kvHeads,
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        query_shape: shapes.attentionQuery,
        key_shape: shapes.attentionKey,
        value_shape: shapes.attentionValue,
      }),
    },
    attentionOperatorSpecs(id, attentionKind, normalized),
  );
}
