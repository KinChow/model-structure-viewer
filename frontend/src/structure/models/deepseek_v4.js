// 对标 vLLM models/deepseek_v4：主干 + nvidia/mtp.py + nvidia/dspark.py。
import { decoderLayerModule } from "../layers/decoderLayer.js";
import { rmsNormModule } from "../layers/norm.js";
import { operatorSpec, weightMatrixDecl } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { moduleSpec, withShapeDims } from "../layers/base.js";
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import {
  draftBilling,
  dsparkLayerCount,
  mtpBlock,
  mtpModuleCount,
  sharedHead,
} from "./deepseek_mtp.js";

function hcHeadGroups(normalized) {
  const hidden = normalized.hiddenSize || 0;
  const streams = normalized.mhcNumResidualStreams || 0;
  return [
    weightMatrixDecl("replicated", { shape: [streams, streams * hidden], param_dtype: "mhc_fn", quantizable: false }),
    weightMatrixDecl("replicated", { shape: [streams], param_dtype: "mhc_base", quantizable: false }),
    weightMatrixDecl("replicated", { shape: [1], param_dtype: "mhc_scale", quantizable: false }),
  ];
}

function hcHead(id, normalized, implementation) {
  const shapes = tensorShapes(normalized);
  const hidden = normalized.hiddenSize || 0;
  return operatorSpec(`${id}.hc_head`, "hc_head", "linear", {
    ...shapeFlow(`[residual streams=${normalized.mhcNumResidualStreams}, ${shapes.hidden}]`, shapes.hidden),
    weightMatrices: hcHeadGroups(normalized),
    implementation,
  }, { input: [-1, normalized.mhcNumResidualStreams, hidden], output: tensorDims(normalized).hidden });
}

function swaMtpNormalized(normalized, layers) {
  return {
    ...normalized,
    compressRatios: Array.from({ length: Math.max(normalized.layers || 0, layers) }, () => 0),
    numHashLayers: 0,
  };
}

function eProjHProj(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return [
    rmsNormModule(`${id}.enorm`, "embedding norm", normalized),
    rmsNormModule(`${id}.hnorm`, "hidden norm", normalized),
    operatorSpec(`${id}.e_proj`, "embedding projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.hidden),
      implementation: ["vLLM.DeepSeekV4MultiTokenPredictorLayer.e_proj", "SGLang.DeepseekV4ModelNextN.e_proj"],
    }, { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${id}.h_proj`, "hidden projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.hidden),
      implementation: ["vLLM.DeepSeekV4MultiTokenPredictorLayer.h_proj", "SGLang.DeepseekV4ModelNextN.h_proj"],
    }, { input: dims.hidden, output: dims.hidden }),
  ];
}

function deepSeekV4MultiTokenPredictorLayer(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const count = mtpModuleCount(normalized);
  const draftNormalized = swaMtpNormalized(normalized, 1);
  return withShapeDims(moduleSpec(
    id,
    "MTP",
    "mtp",
    {
      class: "DeepSeekV4MultiTokenPredictorLayer",
      modules: count,
      ...draftBilling(),
      implementation: [
        "vLLM.models.deepseek_v4.nvidia.mtp.DeepSeekV4MultiTokenPredictorLayer",
        "SGLang.srt.models.deepseek_v4_nextn.DeepseekV4ModelNextN",
      ],
      dataflow_edges: [
        ["enorm", "e_proj"],
        ["hnorm", "h_proj"],
        ["e_proj", "layer"],
        ["h_proj", "layer"],
        ["layer", "hc_head"],
        ["hc_head", "shared_head"],
      ],
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [
      ...eProjHProj(id, normalized),
      mtpBlock(`${id}.layer`, draftNormalized, {
        layerKind: normalized.experts ? "moe" : "dense",
        attentionKind: "dsv4",
        layerIndex: 0,
        forceLastMhc: true,
      }),
      hcHead(id, normalized, ["vLLM.DeepSeekV4MultiTokenPredictorLayer.hc_head_*"]),
      sharedHead(id, normalized),
    ],
    0,
  ), dims.hidden, dims.hidden);
}

function mainProjMainNorm(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const hidden = normalized.hiddenSize || 0;
  const targetIds = normalized.dsparkTargetLayerIds || [];
  return [
    operatorSpec(`${id}.main_proj`, "target-hidden projection", "linear", {
      ...shapeFlow(`[batch, sequence, ${targetIds.length} x hidden size=${hidden}]`, shapes.hidden),
      implementation: ["vLLM.DSparkDeepseekV4Model.main_proj", "SGLang.DSparkV4Stage.main_proj"],
    }, { input: [-1, -1, hidden * targetIds.length], output: dims.hidden }),
    rmsNormModule(`${id}.main_norm`, "main norm", normalized),
  ];
}

function markovHead(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const markovRank = normalized.dsparkMarkovRank || 0;
  const vocab = normalized.vocabSize || 0;
  return withShapeDims(moduleSpec(
    `${id}.markov_head`,
    "Markov head",
    "dspark-markov",
    {
      class: "DSparkMarkovHead",
      markov_rank: markovRank,
      dataflow_edges: [["markov_w1", "markov_w2"]],
      ...shapeFlow(shapes.tokenIds, shapes.logits),
    },
    [
      withShapeDims(moduleSpec(`${id}.markov_head.markov_w1`, "markov w1", "embedding", {
        class: "Embedding",
        hidden_size: markovRank,
        vocab_size: vocab,
        weightMatrices: [weightMatrixDecl("replicated", { shape: [vocab, markovRank], quantizable: false })],
        ...shapeFlow(shapes.tokenIds, `[batch, sequence, markov rank=${markovRank}]`),
      }), dims.tokenIds, [-1, -1, markovRank]),
      operatorSpec(`${id}.markov_head.markov_w2`, "markov w2", "linear", {
        ...shapeFlow(`[batch, sequence, markov rank=${markovRank}]`, shapes.logits),
        implementation: ["vLLM.DSparkMarkovHead.markov_w2"],
      }, { input: [-1, -1, markovRank], output: dims.logits }),
    ],
  ), dims.tokenIds, dims.logits);
}

function confidenceHead(id, normalized) {
  const hidden = normalized.hiddenSize || 0;
  const markovRank = normalized.dsparkMarkovRank || 0;
  return operatorSpec(`${id}.confidence_head`, "confidence head", "linear", {
    ...shapeFlow(`[batch, sequence, hidden+markov=${hidden + markovRank}]`, `[batch, sequence, 1]`),
    weightMatrices: [weightMatrixDecl("replicated", {
      shape: [1, hidden + markovRank],
      quantizable: false,
      param_dtype: "dspark_confidence",
    })],
    implementation: ["vLLM.DSparkConfidenceHead.proj"],
  }, { input: [-1, -1, hidden + markovRank], output: [-1, -1, 1] });
}

function dsparkDeepseekV4Model(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const stages = dsparkLayerCount(normalized);
  const targetIds = normalized.dsparkTargetLayerIds || [];
  const markovRank = normalized.dsparkMarkovRank || 0;
  const draftNormalized = swaMtpNormalized(normalized, stages);
  const stageKind = normalized.experts ? "moe" : "dense";
  const children = [
    ...mainProjMainNorm(id, normalized),
    ...Array.from({ length: stages }, (_, stage) => decoderLayerModule(
      `${id}.${stage}`,
      draftNormalized,
      { layerKind: stageKind, attentionKind: "dsv4", layerIndex: stage, forceLastMhc: stage === stages - 1 },
    )),
    hcHead(id, normalized, ["vLLM.DSparkDeepseekV4Model.hc_head_*"]),
    rmsNormModule(`${id}.norm`, "head norm", normalized),
    markovHead(id, normalized),
    confidenceHead(id, normalized),
  ];
  const dataflowEdges = [
    ["main_proj", "main_norm"],
    ["main_norm", "0"],
    ...Array.from({ length: Math.max(stages - 1, 0) }, (_, stage) => [String(stage), String(stage + 1)]),
    ...(stages > 0 ? [[String(stages - 1), "hc_head"]] : []),
    ["hc_head", "norm"],
    // confidence_head = Linear(H + r)：hc_head 给隐层 H，markov_head 给上一步草稿
    // token 的 markov 嵌入 r（SGLang compute_confidence 里 markov_embed_stack =
    // markov_head.get_prev_embeddings(prev_seq)）。两路 fan-in 拼成 H+r。
    ["hc_head", "confidence_head"],
    ...(markovRank > 0 ? [["markov_head", "confidence_head"]] : []),
  ];
  return withShapeDims(moduleSpec(
    id,
    "DSpark",
    "dspark",
    {
      class: "DSparkDeepseekV4Model",
      modules: 1,
      stages,
      dspark_block_size: normalized.dsparkBlockSize,
      dspark_markov_rank: markovRank,
      dspark_target_layer_ids: targetIds,
      ...draftBilling(),
      implementation: [
        "vLLM.models.deepseek_v4.nvidia.dspark.DSparkDeepseekV4Model",
        "SGLang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark",
      ],
      dataflow_edges: dataflowEdges,
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    children,
    0,
  ), dims.hidden, dims.hidden);
}

function v4Draft(normalized) {
  if (dsparkLayerCount(normalized)) return dsparkDeepseekV4Model("mtp", normalized);
  if (!mtpModuleCount(normalized)) return null;
  return deepSeekV4MultiTokenPredictorLayer("mtp", normalized);
}

export function assembleDeepseekV4(resolved, normalized) {
  const draft = v4Draft(normalized);
  const opts = { attentionKind: "mla", defaultLayerKind: "moe", draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
