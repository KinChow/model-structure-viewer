import { moduleSpec, withShapeDims } from "./base.js";
import { attentionModule } from "./attention.js";
import { mlpModule } from "./mlp.js";
import { moeModule } from "./moe.js";
import { rmsNormModule } from "./norm.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { attentionResidualModule } from "./residual.js";
import { engramModule, hyperConnectionModule, multiHyperConnectionModule, pleModule } from "./hybrid.js";
import { layerInSpec, residualAddSpec } from "../operators/ops/index.js";
import { hfAttentionAttr, hfFfnAttr, hfNamedClass, recipeValue } from "../archs/index.js";

function decoderLayerEdges({ isMhc, layerMix, isLastLayer, hasPle, hasHyper, hasAttnRes, hasEngram, attnAttr, ffnAttr }) {
  const ffn = ffnAttr;
  if (isMhc) {
    return [
      ...(hasEngram ? [["engram", "mhc_attn_pre"]] : []),
      ["mhc_attn_pre", attnAttr],
      [attnAttr, "attn_residual_add"],
      ["mhc_attn_pre", "attn_residual_add"],
      ["attn_residual_add", "mhc_ffn_pre"],
      ["mhc_ffn_pre", ffn],
      [ffn, "ffn_residual_add"],
      ["attn_residual_add", "ffn_residual_add"],
      ...(isLastLayer ? [["ffn_residual_add", "mhc_final_post"], ["mhc_final_post", "mhc_contract"]] : []),
    ];
  }
  if (layerMix === "hyper_connection") {
    const edges = [];
    if (hasPle) edges.push(["ple", "attn_hyper_connection"]);
    edges.push(
      ["attn_hyper_connection", attnAttr],
      [attnAttr, "attn_residual_add"],
      ["attn_hyper_connection", "attn_residual_add"],
      ["attn_residual_add", "mlp_hyper_connection"],
      ["mlp_hyper_connection", ffn],
      [ffn, "ffn_residual_add"],
      ["attn_residual_add", "ffn_residual_add"],
    );
    return edges;
  }
  const edges = [
    ["layer_in", "input_layernorm"],
    ["input_layernorm", attnAttr],
    [attnAttr, "attn_residual_add"],
    ["layer_in", "attn_residual_add"],
    ["attn_residual_add", "post_attention_layernorm"],
    ["post_attention_layernorm", ffn],
    [ffn, "ffn_residual_add"],
    ["attn_residual_add", "ffn_residual_add"],
  ];
  let prev = "ffn_residual_add";
  if (hasPle) {
    edges.push([prev, "ple"]);
    prev = "ple";
  }
  if (hasHyper) {
    edges.push([prev, "attn_hyper_connection"], ["attn_hyper_connection", "mlp_hyper_connection"]);
    prev = "mlp_hyper_connection";
  }
  if (hasAttnRes) edges.push([prev, "attn_residual"]);
  return edges;
}

export function decoderLayerModule(id, normalized, { layerKind, attentionKind, layerIndex = 0, forceLastMhc = false }) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const isMhc = normalized.multiHyperConnection;
  const layerMix = recipeValue(normalized, "layerMix");
  const isLastLayer = forceLastMhc || (isMhc && layerIndex === (normalized.layers || 0) - 1);
  const hasPle = Boolean(normalized.pleLayerIds?.includes(layerIndex + 1));
  // Engram 挂在 engramLayerIds 命中层的入口（0-indexed 主干层号，见 normalize 注释）。
  const hasEngram = Boolean(normalized.engramLayerIds?.includes(layerIndex));
  const hasHyper = Boolean(normalized.hyperConnectionCount);
  const hasAttnRes = Boolean(normalized.attnResBlockSize);
  const attnAttr = hfAttentionAttr(normalized, attentionKind);
  const ffnAttr = hfFfnAttr(normalized, layerKind);
  const ffn = layerKind === "moe"
    ? moeModule(`${id}.${ffnAttr}`, normalized, { layerIndex })
    : mlpModule(`${id}.${ffnAttr}`, normalized.denseIntermediateSize
      ? { ...normalized, intermediateSize: normalized.denseIntermediateSize }
      : normalized);
  const children = isMhc ? [
    ...(hasEngram ? [engramModule(`${id}.engram`, normalized, { layerIndex })] : []),
    multiHyperConnectionModule(`${id}.mhc_attn_pre`, normalized, "pre"),
    attentionModule(`${id}.${attnAttr}`, normalized, attentionKind, layerIndex),
    residualAddSpec(`${id}.attn_residual_add`, normalized, "attention"),
    multiHyperConnectionModule(`${id}.mhc_ffn_pre`, normalized, "fused_post_pre"),
    ffn,
    residualAddSpec(`${id}.ffn_residual_add`, normalized, "feed-forward"),
    ...(isLastLayer ? [
      multiHyperConnectionModule(`${id}.mhc_final_post`, normalized, "post"),
      multiHyperConnectionModule(`${id}.mhc_contract`, normalized, "contract"),
    ] : []),
  ] : layerMix === "hyper_connection" ? [
    ...(hasPle ? [pleModule(`${id}.ple`, normalized, { layerIndex })] : []),
    hyperConnectionModule(`${id}.attn_hyper_connection`, normalized, "attn_mix"),
    attentionModule(`${id}.${attnAttr}`, normalized, attentionKind, layerIndex),
    residualAddSpec(`${id}.attn_residual_add`, normalized, "attention"),
    hyperConnectionModule(`${id}.mlp_hyper_connection`, normalized, "mlp_combine_mix"),
    ffn,
    residualAddSpec(`${id}.ffn_residual_add`, normalized, "feed-forward"),
  ] : [
    layerInSpec(`${id}.layer_in`, normalized),
    rmsNormModule(`${id}.input_layernorm`, "input layernorm", normalized),
    attentionModule(`${id}.${attnAttr}`, normalized, attentionKind, layerIndex),
    residualAddSpec(`${id}.attn_residual_add`, normalized, "attention"),
    rmsNormModule(`${id}.post_attention_layernorm`, "post attention layernorm", normalized),
    ffn,
    residualAddSpec(`${id}.ffn_residual_add`, normalized, "feed-forward"),
    ...(hasPle ? [pleModule(`${id}.ple`, normalized, { layerIndex })] : []),
    ...(hasHyper ? [
      hyperConnectionModule(`${id}.attn_hyper_connection`, normalized),
      hyperConnectionModule(`${id}.mlp_hyper_connection`, normalized),
    ] : []),
    ...(hasAttnRes ? [attentionResidualModule(`${id}.attn_residual`, normalized, { layerIndex })] : []),
  ];
  return withShapeDims(moduleSpec(
    id,
    "DecoderLayer",
    "decoder",
    {
      class: hfNamedClass(normalized, "decoderLayerClass", "DecoderLayer"),
      layer_kind: layerKind,
      dataflow_edges: decoderLayerEdges({ isMhc, layerMix, isLastLayer, hasPle, hasHyper, hasAttnRes, hasEngram, attnAttr, ffnAttr }),
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    children,
  ), dims.hidden, dims.hidden);
}
