import { moduleSpec } from "./base.js";
import { attentionModule } from "./attention.js";
import { mlpModule } from "./mlp.js";
import { moeModule } from "./moe.js";
import { rmsNormModule } from "./norm.js";

export function decoderLayerModule(id, normalized, { layerKind, attentionKind }) {
  return moduleSpec(
    id,
    "DecoderLayer",
    "decoder",
    { class: "DecoderLayer", layer_kind: layerKind },
    [
      rmsNormModule(`${id}.input_layernorm`, "input layernorm"),
      attentionModule(`${id}.self_attn`, normalized, attentionKind),
      rmsNormModule(`${id}.post_attention_layernorm`, "post attention layernorm"),
      layerKind === "moe" ? moeModule(`${id}.moe`, normalized) : mlpModule(`${id}.mlp`, normalized),
    ],
  );
}
