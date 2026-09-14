// 对标 vLLM models/qwen4_exp：主干 + nvidia/mtp.py
import { rmsNormModule } from "../layers/norm.js";
import { hyperConnectionModule } from "../layers/hybrid.js";
import { operatorSpec } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { moduleSpec, withShapeDims } from "../layers/base.js";
import { hfNamedClass } from "../archs/index.js";
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import { draftBilling, mtpBlock, mtpModuleCount } from "./deepseek_mtp.js";

function fcEmbeddingFcHidden(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const hidden = normalized.hiddenSize || 0;
  const streams = normalized.hyperConnectionCount || 1;
  const groupedHidden = `[batch, sequence, hc_count x hidden size=${hidden * streams}]`;
  return [
    rmsNormModule(`${id}.pre_fc_norm_embedding`, "pre-fc embedding norm", normalized),
    withShapeDims(moduleSpec(`${id}.pre_fc_norm_hidden`, "pre-fc grouped hidden norm", "normalization", {
      class: hfNamedClass(normalized, "rmsNormClass", "RMSNorm", "RMSNorm"),
      ...shapeFlow(groupedHidden, groupedHidden),
    }, [
      operatorSpec(`${id}.pre_fc_norm_hidden.rmsnorm`, "Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(groupedHidden, groupedHidden), {
        input: [-1, -1, hidden * streams],
        output: [-1, -1, hidden * streams],
      }),
    ]), [-1, -1, hidden * streams], [-1, -1, hidden * streams]),
    operatorSpec(`${id}.fc_embedding`, "embedding projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.hidden),
      implementation: ["vLLM.Qwen4ExpMultiTokenPredictor.fc_embedding"],
    }, { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${id}.fc_hidden`, "hidden projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.hidden),
      implementation: ["vLLM.Qwen4ExpMultiTokenPredictor.fc_hidden"],
    }, { input: dims.hidden, output: dims.hidden }),
  ];
}

function qwen4ExpMultiTokenPredictor(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const count = mtpModuleCount(normalized);
  const layerKind = normalized.experts ? "moe" : "dense";
  return withShapeDims(moduleSpec(
    id,
    "MTP",
    "mtp",
    {
      class: "Qwen4ExpMultiTokenPredictor",
      modules: count,
      ...draftBilling(),
      implementation: ["vLLM.models.qwen4_exp.nvidia.mtp.Qwen4ExpMultiTokenPredictor"],
      dataflow_edges: [
        ["pre_fc_norm_embedding", "fc_embedding"],
        ["pre_fc_norm_hidden", "fc_hidden"],
        ["fc_embedding", "layer"],
        ["fc_hidden", "layer"],
        ["layer", "hyper_connection_mixer"],
      ],
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [
      ...fcEmbeddingFcHidden(id, normalized),
      mtpBlock(`${id}.layer`, normalized, {
        layerKind,
        attentionKind: "qsa",
        layerIndex: normalized.layers || 0,
        disablePle: true,
      }),
      hyperConnectionModule(`${id}.hyper_connection_mixer`, normalized, "final"),
    ],
    0,
  ), dims.hidden, dims.hidden);
}

function qwen4ExpDraft(normalized) {
  if (!mtpModuleCount(normalized)) return null;
  return qwen4ExpMultiTokenPredictor("mtp", normalized);
}

export function assembleQwen4Exp(resolved, normalized) {
  const draft = qwen4ExpDraft(normalized);
  const opts = { defaultLayerKind: normalized.experts ? "moe" : "dense", draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
