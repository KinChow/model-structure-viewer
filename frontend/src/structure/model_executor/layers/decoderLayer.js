import { moduleSpec, withShapeDims } from "./base.js";
import { attentionModule } from "./attention.js";
import { mlpModule } from "./mlp.js";
import { moeModule } from "./moe.js";
import { rmsNormModule } from "./norm.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";
import { attentionResidualModule } from "./residual.js";
import { hyperConnectionModule, pleModule } from "./hybrid.js";

export function decoderLayerModule(id, normalized, { layerKind, attentionKind, layerIndex = 0 }) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    id,
    "DecoderLayer",
    "decoder",
    { class: "DecoderLayer", layer_kind: layerKind, ...shapeFlow(shapes.hidden, shapes.hidden) },
    [
      rmsNormModule(`${id}.input_layernorm`, "input layernorm", normalized),
      attentionModule(`${id}.self_attn`, normalized, attentionKind),
      rmsNormModule(`${id}.post_attention_layernorm`, "post attention layernorm", normalized),
      layerKind === "moe" ? moeModule(`${id}.moe`, normalized) : mlpModule(`${id}.mlp`, normalized),
      ...(normalized.pleLayerIds?.includes(layerIndex + 1) ? [pleModule(`${id}.ple`, normalized)] : []),
      ...(normalized.hyperConnectionCount ? [
        hyperConnectionModule(`${id}.attn_hyper_connection`, normalized),
        hyperConnectionModule(`${id}.mlp_hyper_connection`, normalized),
      ] : []),
      ...(normalized.attnResBlockSize ? [attentionResidualModule(`${id}.attn_residual`, normalized)] : []),
    ],
  ), dims.hidden, dims.hidden);
}
