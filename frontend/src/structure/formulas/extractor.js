// extractor.js —— node → counts ctx 提取器（docs/details/cost_counts.md 提取器规格）。
//
// 原则（principles §3.2 / §4.3）：
// - 查表优先：节点已有 weight_shapes / input_shape / output_shape / attributes；
//   提取器只补 phase 的 T/S、变体参数、expertFraction。
// - 分派只基于 type / attributes.operator_id / 结构化 attributes 与节点路径（结构化 id），
//   禁止显示名（node.name）参与分派。
// - 不 import cost 层；bytesPerElement 由调用方传入（W5 接线点）。
// - 返回单实例 counts；repeat 倍乘由 walker 的 multiplier 处理（与旧链同）。
// - 标注 "旧链镜像" 的分支：为通过差分而逐字复刻旧公式，W5 切换后随旧链一并删除。

import {
  linearCounts,
  attentionCounts,
  softmaxCounts,
  rmsnormCounts,
  gateCounts,
  swigluCounts,
  ropeCounts,
  causalConvCounts,
  linearAttentionStateCounts,
  topkCounts,
  moeDispatchCounts,
  moeCombineCounts,
  addCounts,
  hashRouteCounts,
  rearrangeCounts,
} from "./counts.js";
import { formulaForOperator } from "./index.js";
import { tensorDims } from "../model_executor/dims.js";
import { visionDimensions } from "../model_executor/layers/vision.js";

// 路径正则全仓统一处（旧 compute.js/parallel.js 三种变体收敛于此）
export const LAYER_INDEX_RE = /(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/;
export const ROUTED_EXPERT_RE = /(?:^|\.)(?<!shared_)(?:experts|expert_mlp)(?:\.|$)/;

/** 与旧 tokensFor 逐字等价：decode 1 token；vision 用 visionTokens。 */
export function tokensFor({ batch = 1, sequence = 1, phase = "prefill", vision = false, visionTokens = 1 } = {}) {
  return batch * (vision ? visionTokens : phase === "decode" ? 1 : sequence);
}

export function layerIndexOf(path) {
  const match = String(path || "").match(LAYER_INDEX_RE);
  return match ? Number(match[1]) : null;
}

/** 与旧 compute.js:309-312 逐字等价（routed expert 且该层非 dense 时按 k/E 缩放）。 */
export function expertFractionFor(path, config) {
  const layerIndex = layerIndexOf(path);
  const layerKind = layerIndex != null ? config?.layerSchedule?.[layerIndex] : null;
  const routed = ROUTED_EXPERT_RE.test(String(path || ""));
  return routed && layerKind !== "dense" && config?.experts && config?.expertsPerToken
    ? config.expertsPerToken / config.experts
    : 1;
}

function productOf(values) {
  return values.reduce((total, value) => total * value, 1);
}

/** 与旧 staticWidth 逐字等价：只保留正有限维并求积；无正维返回 null。 */
function staticWidth(shape) {
  if (!Array.isArray(shape) || shape.length < 1) return null;
  const dimensions = shape.filter((value) => Number.isFinite(value) && value > 0);
  if (dimensions.length === 0) return null;
  return productOf(dimensions);
}

/** 线性逻辑形状：logical_weight_shape 属性优先，其次 weight_shapes 中首个 ≥2 维形状。
 *  packed（qweight）且无逻辑形状 → null（未知，沿用旧链诚实语义）。 */
function linearLogicalShape(node) {
  if (node?.weight_shapes?.qweight && !node?.attributes?.logical_weight_shape) return null;
  const logical = node?.attributes?.logical_weight_shape
    || Object.values(node?.weight_shapes || {}).find((shape) => Array.isArray(shape) && shape.length >= 2);
  return logical || null;
}

/** 旧 derivedLinearMacs 等价：从 input/output 正维积推导 [out, in]。 */
function derivedLinearShape(node) {
  const inputWidth = staticWidth(node?.input_shape);
  const outputWidth = staticWidth(node?.output_shape);
  if (inputWidth == null || outputWidth == null) return null;
  return [outputWidth, inputWidth];
}

function shapeMatchesPattern(shape, pattern) {
  if (!Array.isArray(shape) || !Array.isArray(pattern) || shape.length !== pattern.length) return false;
  return shape.every((value, index) => pattern[index] === -1 || pattern[index] === value);
}

/** scores / context 的输出 shape 模式（text + vision 两套，来源 dims.js / vision.js）。 */
function attentionShapePatterns(config) {
  const dims = tensorDims(config);
  const v = visionDimensions(config);
  return {
    scores: [dims.attentionScores, v.scores],
    context: [dims.attentionContext, v.context],
  };
}

// ---------- 旧链镜像（W5 切换后随旧链删除） ----------

// 旧 attentionMacs：batch·heads·lengthTerm·(D+dv)；不含 vision tokens（旧链现状）。
function legacyAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const lengthTerm = phase === "decode" ? sequence : sequence ** 2;
  return batch * heads * lengthTerm * (qk + value);
}

// 旧 qsaAttentionMacs。
function legacyQsaAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const selected = Math.min(sequence, config?.indexerBudget || sequence);
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  return queryTokens * heads * selected * (qk + value);
}

// 旧 minimaxSparseAttentionMacs。
function legacyMinimaxSparseAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  const selectedBlocks = (config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0);
  const selectedTokens = selectedBlocks * (config?.sparseBlockSize || 1);
  return queryTokens * heads * selectedTokens * (headDim + headDim);
}

// 旧 deepseekV4AttentionMacs（layerIndex 来自节点路径）。
function legacyDeepseekV4AttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill", layerIndex = 0 } = {}) {
  const heads = config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const ratio = config?.compressRatios?.[layerIndex] ?? 0;
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  const available = phase === "decode" ? 1 : sequence;
  const visible = ratio === 0
    ? Math.min(available, config?.slidingWindow || available)
    : ratio === 4
      ? Math.min(Math.ceil(available / ratio) + (config?.slidingWindow || 0), config?.indexerBudget || available)
      : Math.ceil(available / Math.max(ratio, 1));
  return queryTokens * heads * visible * (headDim + headDim);
}

// 旧 linearAttentionMacs（含各 mode 变体的逐字镜像）。
function legacyLinearAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  if (config?.linearAttentionMode === "glm5_next") return legacyGlm5Next(config, { batch, sequence, phase });
  if (config?.linearAttentionMode === "kimi_k3") return legacyKimiK3(config, { batch, sequence, phase });
  if (config?.linearAttentionMode === "qwen4_exp") return legacyQwen4Exp(config, { batch, sequence, phase });
  if (config?.linearAttentionMode === "qwen3_5") return legacyQwen35(config, { batch, sequence, phase });
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || 0;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  return tokens * (hidden * (keyHeads * keyDim + valueHeads * valueDim) + keyHeads * valueHeads * keyDim * valueDim);
}
function legacyQwen35(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const keyHeads = config?.linearKeyHeads || 0;
  const valueHeads = config?.linearValueHeads || 0;
  const keyDim = config?.linearKeyDim || 0;
  const valueDim = config?.linearValueDim || 0;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const convDim = 2 * keyProjection + valueProjection;
  const kernel = config?.linearConvKernelSize || 0;
  const qkvzProjection = hidden * (2 * keyProjection + 2 * valueProjection);
  const baProjection = 2 * hidden * valueHeads;
  const shortConvolution = convDim * kernel;
  const recurrentState = 3 * valueHeads * valueDim * keyDim;
  const gatedNorm = 3 * valueProjection;
  const outputProjection = valueProjection * hidden;
  return tokens * (qkvzProjection + baProjection + shortConvolution + recurrentState + gatedNorm + outputProjection);
}
function legacyGlm5Next(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const heads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const headDim = config?.linearKeyDim || config?.headDim || 0;
  const projection = heads * headDim;
  const convKernel = config?.linearConvKernelSize || 0;
  const fusedProjection = hidden * (3 * projection + heads + 2 * headDim);
  const gateProjections = 2 * headDim * projection;
  const shortConvolution = 3 * projection * convKernel;
  const recurrentState = 3 * heads * headDim * headDim;
  const gatedNorm = 3 * projection;
  const outputProjection = projection * hidden;
  return tokens * (fusedProjection + gateProjections + shortConvolution + recurrentState + gatedNorm + outputProjection);
}
function legacyKimiK3(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const heads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const headDim = config?.linearKeyDim || config?.headDim || 0;
  const projection = heads * headDim;
  const convKernel = config?.linearConvKernelSize || 0;
  const fusedQkvg = hidden * 4 * projection;
  const betaProjection = hidden * heads;
  const decayProjection = hidden * headDim + headDim * projection;
  const shortConvolution = 3 * projection * convKernel;
  const recurrentState = 3 * heads * headDim * headDim;
  const gatedNorm = 3 * projection;
  const outputProjection = projection * hidden;
  return tokens * (fusedQkvg + betaProjection + decayProjection + shortConvolution + recurrentState + gatedNorm + outputProjection);
}
function legacyQwen4Exp(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || keyHeads;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const convDim = 2 * keyProjection + valueProjection;
  const kernel = config?.linearConvKernelSize || 0;
  const qkvzProjection = hidden * (2 * keyProjection + 2 * valueProjection);
  const baProjection = 2 * hidden * valueHeads;
  const shortConvolution = convDim * kernel;
  const recurrentState = 3 * valueHeads * valueDim * keyDim;
  const gatedNorm = 3 * valueProjection;
  const outputProjection = valueProjection * hidden;
  return tokens * (qkvzProjection + baProjection + shortConvolution + recurrentState + gatedNorm + outputProjection);
}
// 旧 linearShortConvolutionMacs。
function legacyShortConvolutionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const { keyProjection, valueProjection } = legacyLinearAttentionDimensions(config);
  const kernel = config?.linearConvKernelSize || 0;
  return batch * (phase === "decode" ? 1 : sequence) * (2 * keyProjection + valueProjection) * kernel;
}
// 旧 linearAttentionDimensions。
function legacyLinearAttentionDimensions(config = {}) {
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || keyHeads;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  return { keyHeads, valueHeads, keyDim, valueDim, keyProjection: keyHeads * keyDim, valueProjection: valueHeads * valueDim };
}
// 旧 linearStateUpdateMacs。
function legacyStateUpdateMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const { keyHeads, valueHeads, keyDim, valueDim } = legacyLinearAttentionDimensions(config);
  const stateUpdate = config?.linearAttentionMode === "generic"
    ? keyHeads * valueHeads * keyDim * valueDim
    : 3 * valueHeads * valueDim * keyDim;
  return batch * (phase === "decode" ? 1 : sequence) * stateUpdate;
}
// ---------- 旧链镜像结束 ----------

/**
 * 计算单个算子节点的动作向量。
 * @returns 动作向量；matrix 无法确定时返回 null（调用方计入 unknownComputePaths）。
 */
export function countsForNode(node, env = {}) {
  const { config, options = {}, path = "", bytesPerElement = 2 } = env;
  const operatorId = String(node?.attributes?.operator_id || "").toLowerCase();
  const type = String(node?.type || "").toLowerCase();
  const kind = String(node?.attributes?.attention_kind || "").toLowerCase();
  const vision = node?.attributes?.modality === "vision";
  const phase = options.phase ?? "prefill";
  const tokens = tokensFor({
    batch: options.batch ?? 1,
    sequence: options.sequence ?? 1,
    phase,
    vision,
    visionTokens: config?.visionTokens || 1,
  });

  // 注意力模块节点（type === "attention"）：旧链镜像（模块 own 值不进总量，W5 裁决去留）
  if (type === "attention") {
    const legacyOptions = { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase };
    let matrix = null;
    if (kind === "linear") matrix = legacyLinearAttentionMacs(config, legacyOptions);
    else if (kind === "qsa") matrix = legacyQsaAttentionMacs(config, legacyOptions);
    else if (kind === "sparse" && config?.modelType === "minimax_m3_vl") matrix = legacyMinimaxSparseAttentionMacs(config, legacyOptions);
    else if (kind === "dsv4") matrix = legacyDeepseekV4AttentionMacs(config, { ...legacyOptions, layerIndex: layerIndexOf(node?.id || path) ?? 0 });
    else matrix = legacyAttentionMacs(config, legacyOptions);
    return { matrix, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-module-mirror" };
  }

  switch (operatorId) {
    case "linear": {
      // 与旧 isLinear 的 embed 排除等价：以结构化路径判断（node.name 不参与，§3.2）。
      // TODO(W3): builder 为 embed 投影声明结构化标记后移除路径判断。
      if (/(^|\.)(patch_)?embed/.test(String(node?.id || path))) return { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
      const expertFraction = expertFractionFor(node?.id || path, config);
      const logical = linearLogicalShape(node) || derivedLinearShape(node);
      if (!logical) return null;
      return linearCounts({ logicalShape: logical, tokens, bytesPerElement, expertFraction });
    }
    case "matmul": {
      // scores/context 用输出 shape 模式匹配区分（结构化判据，§3.2）。
      const patterns = attentionShapePatterns(config);
      const output = node?.output_shape;
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const queryTokens = tokens;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      // context 模式更具体（含具体 heads/value 维），必须先判；
      // scores 的全 -1 通配模式会吞掉一切 4D 输出。
      const part = patterns.context.some((pattern) => shapeMatchesPattern(output, pattern)) ? "context"
        : patterns.scores.some((pattern) => shapeMatchesPattern(output, pattern)) ? "scores"
        : null;
      if (part === "scores") {
        return { matrix: queryTokens * heads * keyTokens * headDim, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-matmul-mirror" };
      }
      if (part === "context") {
        return { matrix: queryTokens * heads * keyTokens * valueDim, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-matmul-mirror" };
      }
      return null;
    }
    case "qsa_attention": {
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      const selected = Math.min(keyTokens, config?.indexerBudget || keyTokens);
      return { matrix: tokens * heads * selected * (headDim + valueDim), vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-mirror" };
    }
    case "minimax_sparse_attention": {
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      const selectedTokens = (config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0);
      const size = config?.sparseBlockSize || 1;
      return { matrix: tokens * heads * selectedTokens * size * (headDim + valueDim), vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-mirror" };
    }
    case "dsv4_swa_attention":
    case "dsv4_compressed_attention": {
      const layerIndex = layerIndexOf(node?.id || path) ?? 0;
      const batch = options.batch ?? 1;
      const sequence = options.sequence ?? 1;
      return { matrix: legacyDeepseekV4AttentionMacs(config, { batch, sequence, phase, layerIndex }), vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-mirror" };
    }
    case "softmax": {
      const dims = attentionShapePatterns(config);
      void dims;
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const keyTokens = vision ? config?.visionTokens || 1 : phase === "decode" ? options.sequence ?? 1 : options.sequence ?? 1;
      const queryTokens = tokens;
      return softmaxCounts({ elements: heads * queryTokens * keyTokens, bytesPerElement });
    }
    case "rope": {
      const factor = node?.attributes?.partial_rotary_factor ?? config?.partialRotaryFactor ?? 1;
      return ropeCounts({ tokens, ropeDims: (config?.headDim || 0) * factor, bytesPerElement });
    }
    case "rmsnorm":
    case "gemma_rmsnorm":
      return rmsnormCounts({ tokens, hidden: staticWidth(node?.input_shape) || 0, bytesPerElement, weightOne: operatorId === "gemma_rmsnorm" });
    case "gated_rmsnorm":
      return rmsnormCounts({ tokens, hidden: staticWidth(node?.input_shape) || 0, bytesPerElement, gated: true });
    case "attention_output_gate":
    case "mla_output_gate":
    case "linear_attention_gate":
    case "shared_expert_gate":
      return gateCounts({ tokens, width: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_activation":
      return swigluCounts({ tokens, intermediate: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_position":
      return addCounts({ tokens, hidden: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_merge":
      return rearrangeCounts({ copy: true, inElements: staticWidth(node?.input_shape) || 0, outElements: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "split":
    case "mla_kv_split":
    case "qwen_qkvz_split":
    case "attention_qkv_split":
      return rearrangeCounts();
    case "swiglu": {
      // 旧链：仅 expert 路径的 swiglu 计矩阵（gate/up/down GEMM 语义融合）；
      // 结构化判据用路径（专家目录），W3 换 attributes 标记。
      const routed = ROUTED_EXPERT_RE.test(String(node?.id || path));
      if (!routed) return swigluCounts({ tokens, intermediate: staticWidth(node?.output_shape) || 0, bytesPerElement });
      const expertFraction = expertFractionFor(node?.id || path, config);
      const expertHidden = node?.attributes?.latent_size || config?.routedExpertHiddenSize || config?.hiddenSize || 0;
      const expertIntermediate = config?.moeIntermediateSize || config?.intermediateSize || 0;
      return { matrix: tokens * 3 * expertHidden * expertIntermediate * expertFraction, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-mirror" };
    }
    case "causal_conv1d": {
      const { keyProjection, valueProjection } = legacyLinearAttentionDimensions(config);
      const kernel = config?.linearConvKernelSize || 0;
      return { matrix: tokens * (2 * keyProjection + valueProjection) * kernel, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-mirror" };
    }
    case "linear_attention": {
      // 叶级 state/conv 用路径区分（旧链用显示名；W3 换结构化标记）。
      const idPath = String(node?.id || path);
      if (/short_conv|conv/.test(idPath)) {
        const { keyProjection, valueProjection } = legacyLinearAttentionDimensions(config);
        const kernel = config?.linearConvKernelSize || 0;
        return { matrix: tokens * (2 * keyProjection + valueProjection) * kernel, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-mirror" };
      }
      if (/state|recurrent/.test(idPath)) {
        return { matrix: legacyStateUpdateMacs(config, { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase }), vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-mirror" };
      }
      return { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
    }
    case "gated_delta_attention":
      return { matrix: legacyStateUpdateMacs(config, { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase }), vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 }, source: "legacy-mirror" };
    case "topk":
      return topkCounts({ tokens, experts: config?.experts || 0, topk: config?.expertsPerToken || 0, bytesPerElement, normTopkProb: config?.normTopkProb ?? true });
    case "moe_dispatch":
      return moeDispatchCounts({ tokens, hidden: config?.hiddenSize || 0, topk: config?.expertsPerToken || 0, bytesPerElement });
    case "moe_combine":
      return moeCombineCounts({ tokens, hidden: config?.hiddenSize || 0, topk: config?.expertsPerToken || 0, bytesPerElement });
    case "moe_add":
      return addCounts({ tokens, hidden: staticWidth(node?.output_shape) || config?.hiddenSize || 0, bytesPerElement });
    case "dsv4_hash_route":
      return hashRouteCounts({ tokens, topk: config?.expertsPerToken || 0, tableRows: 0, bytesPerElement });
    default: {
      // 复合节点：调注册表的组合 counts（§3.1 唯一注册点），ctx 按规格构建。
      // 精度为初版（n 流参数用 normalized 近似），恒等式（T4）校准后复核。
      const entry = formulaForOperator(operatorId);
      if (typeof entry?.counts !== "function") return null;
      const H = config?.hiddenSize || 0;
      const ctxBuilders = {
        mla_query_compress: () => ({
          qa: { logicalShape: [config?.qLoraRank || 0, H], tokens, bytesPerElement },
          norm: { tokens, hidden: config?.qLoraRank || 0, bytesPerElement },
          qb: { logicalShape: [(config?.attentionHeads || 0) * (config?.headDim || 0), config?.qLoraRank || 0], tokens, bytesPerElement },
        }),
        mla_kv_compress: () => ({
          proj: { logicalShape: [(config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0), H], tokens, bytesPerElement },
        }),
        qsa_indexer: () => ({
          score: { heads: config?.indexerNHeads || 0, queryTokens: tokens, keyTokens: options.sequence ?? 1, headDim: config?.indexerHeadDim || 0, valueDim: config?.indexerHeadDim || 0, bytesPerElement },
          topk: { tokens, experts: options.sequence ?? 1, topk: config?.indexerBudget || 0, bytesPerElement },
        }),
        minimax_sparse_indexer: () => ({
          score: { heads: config?.sparseIndexHeads || 0, queryTokens: tokens, keyTokens: options.sequence ?? 1, headDim: config?.sparseIndexDim || 0, valueDim: config?.sparseIndexDim || 0, bytesPerElement },
          topk: { tokens, experts: options.sequence ?? 1, topk: config?.sparseTopkBlocks || 0, bytesPerElement },
        }),
        attention_residual: () => ({
          norms: { tokens, hidden: H, bytesPerElement },
          scoreProj: { logicalShape: [1, H], tokens, bytesPerElement },
          aggregate: { elements: H * tokens, bytesPerElement },
          mix: { tokens, hidden: H, bytesPerElement },
        }),
        hyper_connection: () => ({
          grouped: { tokens, hidden: H, bytesPerElement },
          mix: { tokens, intermediate: H, bytesPerElement },
          mixers: { logicalShape: [H, H], tokens, bytesPerElement },
          gate: { tokens, width: H, bytesPerElement },
          combine: { tokens, hidden: H, bytesPerElement },
        }),
        ple: () => ({
          embed: { tokens, topk: 1, tableRows: 0, bytesPerElement },
          kv: { logicalShape: [2 * (config?.pleEmbedDim || 0), H], tokens, bytesPerElement },
          norm: { tokens, hidden: config?.pleEmbedDim || 0, bytesPerElement },
          conv: { tokens: tokens, channels: config?.pleEmbedDim || 0, kernel: config?.pleNgramSize || 1, bytesPerElement },
          add: { tokens, hidden: H, bytesPerElement },
        }),
        mhc_pre: () => ({
          mix: { tokens, width: H, bytesPerElement },
          matrix: { logicalShape: [H, config?.mhcNumResidualStreams || 1], tokens, bytesPerElement },
          merge: { tokens, hidden: H, bytesPerElement },
        }),
        mhc_post: () => ({
          combine: { logicalShape: [H, config?.mhcNumResidualStreams || 1], tokens, bytesPerElement },
          inject: { tokens, hidden: H, bytesPerElement },
        }),
        mhc_fused_post_pre: () => ({
          post: { tokens, width: H, bytesPerElement },
          inject: { tokens, hidden: H, bytesPerElement },
          pre: { tokens, width: H, bytesPerElement },
          matrix: { logicalShape: [H, config?.mhcNumResidualStreams || 1], tokens, bytesPerElement },
        }),
        mhc_contract: () => ({
          contract: { tokens, hidden: H, bytesPerElement },
        }),
      };
      const builder = ctxBuilders[operatorId];
      if (!builder) return null;
      return { ...entry.counts(builder()), source: "composite" };
    }
  }
}
