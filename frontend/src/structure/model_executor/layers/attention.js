import { moduleSpec } from "./base.js";
import { attentionOperatorSpecs } from "../ops/index.js";

export function attentionModule(id, normalized, attentionKind) {
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
    },
    attentionOperatorSpecs(id, attentionKind),
  );
}
