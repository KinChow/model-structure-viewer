import { moduleSpec, withShapeDims } from "./base.js";
import { attentionModule } from "./attention.js";
import { mlpModule } from "./mlp.js";
import { moeModule } from "./moe.js";
import { rmsNormModule } from "./norm.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";
import { attentionResidualModule } from "./residual.js";
import { hyperConnectionModule, multiHyperConnectionModule, pleModule } from "./hybrid.js";

export function decoderLayerModule(id, normalized, { layerKind, attentionKind, layerIndex = 0 }) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const isMhc = normalized.multiHyperConnection;
  const isQwen4Exp = normalized.modelType === "qwen4_exp";
  const isLastLayer = isMhc && layerIndex === (normalized.layers || 0) - 1;
  const children = isMhc ? [
    multiHyperConnectionModule(`${id}.mhc_attn_pre`, normalized, "pre"),
    attentionModule(`${id}.self_attn`, normalized, attentionKind),
    multiHyperConnectionModule(`${id}.mhc_ffn_pre`, normalized, "fused_post_pre"),
    layerKind === "moe" ? moeModule(`${id}.moe`, normalized) : mlpModule(`${id}.mlp`, normalized),
    ...(isLastLayer ? [
      multiHyperConnectionModule(`${id}.mhc_final_post`, normalized, "post"),
      multiHyperConnectionModule(`${id}.mhc_contract`, normalized, "contract"),
    ] : []),
  ] : isQwen4Exp ? [
    ...(normalized.pleLayerIds?.includes(layerIndex + 1) ? [pleModule(`${id}.ple`, normalized)] : []),
    hyperConnectionModule(`${id}.attn_hyper_connection`, normalized, "attn_mix"),
    attentionModule(`${id}.self_attn`, normalized, attentionKind),
    hyperConnectionModule(`${id}.mlp_hyper_connection`, normalized, "mlp_combine_mix"),
    layerKind === "moe" ? moeModule(`${id}.moe`, normalized) : mlpModule(`${id}.mlp`, normalized),
  ] : [
    rmsNormModule(`${id}.input_layernorm`, "input layernorm", normalized),
    attentionModule(`${id}.self_attn`, normalized, attentionKind),
    rmsNormModule(`${id}.post_attention_layernorm`, "post attention layernorm", normalized),
    layerKind === "moe" ? moeModule(`${id}.moe`, normalized) : mlpModule(`${id}.mlp`, normalized),
    ...(normalized.pleLayerIds?.includes(layerIndex + 1) ? [pleModule(`${id}.ple`, normalized)] : []),
    ...(normalized.hyperConnectionCount ? [
      hyperConnectionModule(`${id}.attn_hyper_connection`, normalized),
      hyperConnectionModule(`${id}.mlp_hyper_connection`, normalized),
    ] : []),
    ...(normalized.attnResBlockSize ? [attentionResidualModule(`${id}.attn_residual`, normalized, { layerIndex })] : []),
  ];
  return withShapeDims(moduleSpec(
    id,
    "DecoderLayer",
    "decoder",
    { class: "DecoderLayer", layer_kind: layerKind, ...shapeFlow(shapes.hidden, shapes.hidden) },
    children,
  ), dims.hidden, dims.hidden);
}
