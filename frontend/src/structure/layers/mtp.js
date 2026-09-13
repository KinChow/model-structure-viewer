// mtp.js —— 投机头零件 + 按 vLLM 类组网。
//
// 对标：MTP/DSpark 是独立注册项（各模型 `mtp.py` / `dspark.py`），不是 decoder
// 子层。零件名 = 权重/成员名（enorm、eh_proj、e_proj、fc、fc_embedding、
// SharedHead.norm、hc_head、markov_head、confidence_head）。组网函数名 =
// 对标类名。变化轴是这些成员怎么拼，不是家族名。
//
// **计费**：`repeat: 0`。投机默认关，不算每次前向；参数仍占显存
// （`residentRepeat`，原则 §3.8）。一份模板 × N 写 `modules=N`；已展开
// stage（DSpark `mtp.0/1/2`）写 `modules=1`。
import { moduleSpec, withShapeDims } from "./base.js";
import { decoderLayerModule } from "./decoderLayer.js";
import { rmsNormModule } from "./norm.js";
import { hyperConnectionModule } from "./hybrid.js";
import { operatorSpec, weightMatrixDecl } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { hfNamedClass, recipeLinearAttentionMode } from "../archs/index.js";

export function mtpModuleCount(normalized) {
  if ((normalized.dsparkTargetLayerIds || []).length > 0) return 0;
  return normalized.mtpModules || 0;
}

export function dsparkLayerCount(normalized) {
  const ids = normalized.dsparkTargetLayerIds || [];
  return ids.length > 0 ? ids.length : 0;
}

/** 字段分派到对标类。搜 vLLM 仓库能对上这些 class 名。 */
export function draftClassOf(normalized) {
  if ((normalized.dsparkTargetLayerIds || []).length > 0) return "DSparkDeepseekV4Model";
  if (!(normalized.mtpModules > 0)) return null;
  if ((normalized.compressRatios || []).length > 0) return "DeepSeekV4MultiTokenPredictorLayer";
  if (normalized.hyperConnectionCount && recipeLinearAttentionMode(normalized) === "qwen4_exp") {
    return "Qwen4ExpMultiTokenPredictor";
  }
  if (recipeLinearAttentionMode(normalized) === "qwen3_5" || recipeLinearAttentionMode(normalized) === "qwen4_exp") {
    return "Qwen3_5MultiTokenPredictor";
  }
  return "DeepSeekMultiTokenPredictorLayer";
}

function hcHeadGroups(normalized) {
  const hidden = normalized.hiddenSize || 0;
  const streams = normalized.mhcNumResidualStreams || 0;
  return [
    weightMatrixDecl("replicated", { shape: [streams, streams * hidden], param_dtype: "mhc_fn", quantizable: false }),
    weightMatrixDecl("replicated", { shape: [streams], param_dtype: "mhc_base", quantizable: false }),
    weightMatrixDecl("replicated", { shape: [1], param_dtype: "mhc_scale", quantizable: false }),
  ];
}

function draftBilling() {
  return {
    speculative_decoding: "disabled",
    compute_multiplier: 0,
    note: "投机解码未启用：参数计入显存，不计入每次前向的算力与访存",
  };
}

function mtpBlock(id, normalized, { layerKind, attentionKind, layerIndex = 0, forceLastMhc = false, disablePle = false, disableAttnRes = false }) {
  return decoderLayerModule(id, {
    ...normalized,
    ...(disablePle ? { pleLayerIds: [] } : {}),
    ...(disableAttnRes ? { attnResBlockSize: undefined } : {}),
  }, { layerKind, attentionKind, layerIndex, forceLastMhc });
}

function swaMtpNormalized(normalized, layers) {
  return {
    ...normalized,
    compressRatios: Array.from({ length: Math.max(normalized.layers || 0, layers) }, () => 0),
    numHashLayers: 0,
  };
}

// ----- 零件：名字 = vLLM 成员 -----

/** vLLM SharedHead：self.norm + 共享 lm_head。checkpoint 路径 shared_head.norm。 */
function sharedHead(id, normalized) {
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

/** vLLM DeepSeekV4 / DSpark 的 hc_head_fn / hc_head_base / hc_head_scale。 */
function hcHead(id, normalized, implementation) {
  const shapes = tensorShapes(normalized);
  const hidden = normalized.hiddenSize || 0;
  return operatorSpec(`${id}.hc_head`, "hc_head", "linear", {
    ...shapeFlow(`[residual streams=${normalized.mhcNumResidualStreams}, ${shapes.hidden}]`, shapes.hidden),
    weightMatrices: hcHeadGroups(normalized),
    implementation,
  }, { input: [-1, normalized.mhcNumResidualStreams, hidden], output: tensorDims(normalized).hidden });
}

/** vLLM DeepSeekMultiTokenPredictorLayer：enorm + hnorm + eh_proj。 */
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

/** vLLM DeepSeekV4MultiTokenPredictorLayer：enorm + hnorm + e_proj + h_proj。 */
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

/** vLLM Qwen3_5MultiTokenPredictor：pre_fc_norm_* + fc。 */
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

/** vLLM Qwen4ExpMultiTokenPredictor：pre_fc_norm_* + fc_embedding + fc_hidden。 */
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

/** vLLM DSparkDeepseekV4Model.main_proj + main_norm。 */
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

/** vLLM DSparkMarkovHead：markov_w1 Embedding + markov_w2 Linear。 */
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

/** vLLM DSparkConfidenceHead.proj。 */
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

function ehProjKind(normalized) {
  if ((normalized.sparseTopkBlocks || 0) > 0) return { attentionKind: "sparse", layerKind: "moe" };
  if (normalized.dsaIndexKpool > 1 || normalized.kvLoraRank) {
    return { attentionKind: normalized.kvLoraRank ? "qsa" : "mla", layerKind: normalized.experts ? "moe" : "dense" };
  }
  return { attentionKind: normalized.kvLoraRank ? "mla" : "gqa", layerKind: normalized.experts ? "moe" : "dense" };
}

// ----- 组网：名字 = vLLM 类 -----

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

export function dsparkDeepseekV4Model(id, normalized) {
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
    ["hc_head", "confidence_head"],
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
      note: "DSpark draft：参数计入显存，不计入每次前向的算力与访存。embed/lm_head 与主干共享。",
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

export function mtpModule(id, normalized) {
  const cls = draftClassOf(normalized);
  if (cls === "DSparkDeepseekV4Model") return dsparkDeepseekV4Model(id, normalized);
  if (cls === "DeepSeekV4MultiTokenPredictorLayer") return deepSeekV4MultiTokenPredictorLayer(id, normalized);
  if (cls === "Qwen4ExpMultiTokenPredictor") return qwen4ExpMultiTokenPredictor(id, normalized);
  if (cls === "Qwen3_5MultiTokenPredictor") return qwen3_5MultiTokenPredictor(id, normalized);
  return deepSeekMultiTokenPredictorLayer(id, normalized);
}

/** 组网 children 用：有投机头就返回节点，没有返回 null。树 id 仍是 mtp（checkpoint）。 */
export function mtpChild(normalized) {
  if (dsparkLayerCount(normalized)) return dsparkDeepseekV4Model("mtp", normalized);
  if (!mtpModuleCount(normalized)) return null;
  return mtpModule("mtp", normalized);
}
