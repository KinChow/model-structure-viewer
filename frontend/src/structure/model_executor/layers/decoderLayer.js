import { moduleSpec } from "./base.js";
import { attentionModule } from "./attention.js";
import { mlpModule } from "./mlp.js";
import { moeModule } from "./moe.js";
import { rmsNormModule } from "./norm.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function decoderLayerModule(id, normalized, { layerKind, attentionKind }) {
  const shapes = tensorShapes(normalized);
  return moduleSpec(
    id,
    "DecoderLayer",
    "decoder",
    { class: "DecoderLayer", layer_kind: layerKind, ...shapeFlow(shapes.hidden, shapes.hidden) },
    [
      rmsNormModule(`${id}.input_layernorm`, "input layernorm", normalized),
      attentionModule(`${id}.self_attn`, normalized, attentionKind),
      rmsNormModule(`${id}.post_attention_layernorm`, "post attention layernorm", normalized),
      layerKind === "moe" ? moeModule(`${id}.moe`, normalized) : mlpModule(`${id}.mlp`, normalized),
    ],
  );
}
