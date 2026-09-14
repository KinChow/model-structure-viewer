// deepseek_mtp.js —— 对标 vLLM model_executor/models/deepseek_mtp.py
// SharedHead 被 V4 / GLM-5 Next 引用，不升格成跨架构 dispatcher。
import { moduleSpec, withShapeDims } from "../layers/base.js";
import { decoderLayerModule } from "../layers/decoderLayer.js";
import { rmsNormModule } from "../layers/norm.js";
import { operatorSpec, weightMatrixDecl } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { hfNamedClass } from "../archs/index.js";

export function mtpModuleCount(normalized) {
  if ((normalized.dsparkTargetLayerIds || []).length > 0) return 0;
  return normalized.mtpModules || 0;
}

export function dsparkLayerCount(normalized) {
  const ids = normalized.dsparkTargetLayerIds || [];
  return ids.length > 0 ? ids.length : 0;
}

export function draftBilling() {
  return {
    speculative_decoding: "disabled",
    compute_multiplier: 0,
    note: "投机解码未启用：参数计入显存，不计入每次前向的算力与访存",
  };
}

export function mtpBlock(id, normalized, { layerKind, attentionKind, layerIndex = 0, forceLastMhc = false, disablePle = false, disableAttnRes = false }) {
  return decoderLayerModule(id, {
    ...normalized,
    ...(disablePle ? { pleLayerIds: [] } : {}),
    ...(disableAttnRes ? { attnResBlockSize: undefined } : {}),
  }, { layerKind, attentionKind, layerIndex, forceLastMhc });
}

/** vLLM SharedHead：self.norm + 共享 lm_head。checkpoint 路径 shared_head.norm。 */
export function sharedHead(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    `${id}.shared_head`,
    "SharedHead",
    "shared-head",
    {
      class: "SharedHead",
      implementation: ["vLLM.models.deepseek_mtp.SharedHead"],
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [rmsNormModule(`${id}.shared_head.norm`, "shared head norm", normalized)],
  ), dims.hidden, dims.hidden);
}

function ehProj(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const hidden = normalized.hiddenSize || 0;
  const concatShape = `[batch, sequence, 2 x hidden size=${2 * hidden}]`;
  return [
    rmsNormModule(`${id}.enorm`, "embedding norm", normalized),
    rmsNormModule(`${id}.hnorm`, "hidden norm", normalized),
    operatorSpec(`${id}.eh_proj`, "embedding/hidden concat projection", "linear", {
      ...shapeFlow(concatShape, shapes.hidden),
      implementation: ["vLLM.DeepSeekMultiTokenPredictorLayer.eh_proj"],
    }, { input: [-1, -1, 2 * hidden], output: dims.hidden }),
  ];
}

function ehProjKind(normalized) {
  if ((normalized.sparseTopkBlocks || 0) > 0) return { attentionKind: "sparse", layerKind: "moe" };
  if (normalized.dsaIndexKpool > 1 || normalized.kvLoraRank) {
    return { attentionKind: normalized.kvLoraRank ? "qsa" : "mla", layerKind: normalized.experts ? "moe" : "dense" };
  }
  return { attentionKind: normalized.kvLoraRank ? "mla" : "gqa", layerKind: normalized.experts ? "moe" : "dense" };
}

function deepSeekMultiTokenPredictorLayer(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const count = mtpModuleCount(normalized);
  const { attentionKind, layerKind } = ehProjKind(normalized);
  return withShapeDims(moduleSpec(
    id,
    "MTP",
    "mtp",
    {
      class: hfNamedClass(normalized, "mtpClass", "DeepSeekMultiTokenPredictorLayer"),
      modules: count,
      ...draftBilling(),
      implementation: ["vLLM.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer"],
      dataflow_edges: [
        ["enorm", "eh_proj"],
        ["hnorm", "eh_proj"],
        ["eh_proj", "layer"],
        ["layer", "shared_head"],
      ],
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [
      ...ehProj(id, normalized),
      mtpBlock(`${id}.layer`, normalized, { layerKind, attentionKind, layerIndex: normalized.layers || 0, disableAttnRes: true }),
      sharedHead(id, normalized),
    ],
    0,
  ), dims.hidden, dims.hidden);
}

export function deepSeekMtpChild(normalized) {
  if (!mtpModuleCount(normalized)) return null;
  return deepSeekMultiTokenPredictorLayer("mtp", normalized);
}
