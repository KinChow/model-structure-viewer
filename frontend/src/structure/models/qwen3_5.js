// 对标 vLLM model_executor/models/qwen3_5.py + qwen3_5_mtp.py
import { rmsNormModule } from "../layers/norm.js";
import { operatorSpec } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { moduleSpec, withShapeDims } from "../layers/base.js";
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import { draftBilling, mtpBlock, mtpModuleCount } from "./deepseek_mtp.js";

function fc(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const hidden = normalized.hiddenSize || 0;
  const concatShape = `[batch, sequence, 2 x hidden size=${2 * hidden}]`;
  return [
    rmsNormModule(`${id}.pre_fc_norm_embedding`, "pre-fc embedding norm", normalized),
    rmsNormModule(`${id}.pre_fc_norm_hidden`, "pre-fc hidden norm", normalized),
    operatorSpec(`${id}.fc`, "embedding/hidden concat projection", "linear", {
      ...shapeFlow(concatShape, shapes.hidden),
      implementation: ["vLLM.Qwen3_5MultiTokenPredictor.fc"],
    }, { input: [-1, -1, 2 * hidden], output: dims.hidden }),
  ];
}

function qwen3_5MultiTokenPredictor(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const count = mtpModuleCount(normalized);
  const layerKind = normalized.experts ? "moe" : "dense";
  return withShapeDims(moduleSpec(
    id,
    "MTP",
    "mtp",
    {
      class: "Qwen3_5MultiTokenPredictor",
      modules: count,
      ...draftBilling(),
      implementation: ["vLLM.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"],
      dataflow_edges: [
        ["pre_fc_norm_embedding", "fc"],
        ["pre_fc_norm_hidden", "fc"],
        ["fc", "layer"],
        ["layer", "norm"],
      ],
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [
      ...fc(id, normalized),
      mtpBlock(`${id}.layer`, normalized, { layerKind, attentionKind: "qwen35_full", layerIndex: 0 }),
      rmsNormModule(`${id}.norm`, "MTP head norm", normalized),
    ],
    0,
  ), dims.hidden, dims.hidden);
}

function qwen35Draft(normalized) {
  if (!mtpModuleCount(normalized)) return null;
  return qwen3_5MultiTokenPredictor("mtp", normalized);
}

export function assembleQwen3_5(resolved, normalized) {
  const draft = qwen35Draft(normalized);
  const defaultLayerKind = normalized.experts ? "moe" : "dense";
  const opts = { defaultLayerKind, draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
