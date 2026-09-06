// 推理场景的逐模块 MACs 估算；显式暴露假设，不用于预测延迟。

import { nodeWeightBytes, product, tensorElements } from "./memory.js";
import { walkStructure } from "./traverse.js";

function tokensFor({ batch = 1, sequence = 1, phase = "prefill" } = {}) {
  return batch * (phase === "decode" ? 1 : sequence);
}

function isLinear(node) {
  if (`${node?.type || ""} ${node?.name || ""}`.toLowerCase().includes("embed")) return false;
  if (Object.values(node?.weight_shapes || {}).some((shape) => Array.isArray(shape) && shape.length >= 2)) return true;
  return ["linear", "projection"].some((term) =>
    `${node?.type || ""} ${node?.attributes?.operator_id || ""} ${node?.name || ""}`.toLowerCase().includes(term),
  );
}

function staticWidth(shape) {
  if (!Array.isArray(shape) || shape.length < 3) return null;
  const dimensions = shape.slice(2);
  if (dimensions.some((value) => !Number.isFinite(value) || value <= 0)) return null;
  return dimensions.reduce((total, value) => total * value, 1);
}

function hasLogicalLinearShape(node) {
  return Boolean(
    node?.attributes?.logical_weight_shape
    || Object.values(node?.weight_shapes || {}).some((shape) => Array.isArray(shape) && shape.length >= 2),
  );
}

function derivedLinearMacs(node, { batch, sequence, phase, expertFraction = 1 } = {}) {
  const inputWidth = staticWidth(node?.input_shape);
  const outputWidth = staticWidth(node?.output_shape);
  if (inputWidth == null || outputWidth == null) return null;
  return tokensFor({ batch, sequence, phase }) * inputWidth * outputWidth * expertFraction;
}

// 来源：llm-analysis 的 LLMAnalysis.get_num_flops_fwd_per_layer_linear。
export function linearMacs(node, { batch, sequence, phase, expertFraction = 1 } = {}) {
  // GPTQ/AWQ 的 packed shape 是存储形状，不是逻辑矩阵乘形状。
  // 没有明确的逻辑形状时，不输出看似合理但实际错误的 MACs。
  if (node?.weight_shapes?.qweight && !node?.attributes?.logical_weight_shape) return null;
  const shape = Object.values(node?.weight_shapes || {}).find((value) => Array.isArray(value) && value.length >= 2);
  const logicalShape = node?.attributes?.logical_weight_shape || shape;
  return logicalShape
    ? tokensFor({ batch, sequence, phase }) * product(logicalShape) * expertFraction
    : derivedLinearMacs(node, { batch, sequence, phase, expertFraction });
}

// 来源：llm-analysis 的 LLMAnalysis.get_num_flops_fwd_per_layer_attn。
export function attentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const lengthTerm = phase === "decode" ? sequence : sequence ** 2;
  return batch * heads * lengthTerm * (qk + value);
}

// DeepSeek V4 每层由 compress_ratio 决定可见的 attention 序列长度；这是理论 MAC 估计，
// 不把 vLLM/SGLang 的 FlashMLA、FlashInfer 或 Triton kernel 当成新的语义节点。
export function deepseekV4AttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill", layerIndex = 0 } = {}) {
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

export function linearAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  if (config?.linearAttentionMode === "glm5_next") return glm5NextLinearAttentionMacs(config, { batch, sequence, phase });
  if (config?.linearAttentionMode === "kimi_k3") return kimiK3LinearAttentionMacs(config, { batch, sequence, phase });
  if (config?.linearAttentionMode === "qwen4_exp") return qwen4ExpLinearAttentionMacs(config, { batch, sequence, phase });
  if (config?.linearAttentionMode === "qwen3_5") return qwen35LinearAttentionMacs(config, { batch, sequence, phase });
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || 0;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  // Linear attention keeps a recurrent state, so its state update is O(T),
  // unlike full attention's O(T^2) score/context products.
  return tokens * (hidden * (keyHeads * keyDim + valueHeads * valueDim) + keyHeads * valueHeads * keyDim * valueDim);
}

function qwen35LinearAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
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

// GLM-5.3-Flash KDA cost follows the actual vLLM execution chain rather than
// treating every linear-attention family as one generic projection.
function glm5NextLinearAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
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

function kimiK3LinearAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
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

function qwen4ExpLinearAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
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

export function qsaAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const selected = Math.min(sequence, config?.indexerBudget || sequence);
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  return queryTokens * heads * selected * (qk + value);
}

function minimaxSparseAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  const selectedBlocks = (config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0);
  const selectedTokens = selectedBlocks * (config?.sparseBlockSize || 1);
  return queryTokens * heads * selectedTokens * (headDim + headDim);
}

export function nodeMacs(node, config, options = {}) {
  const type = String(node?.type || "").toLowerCase();
  const operatorId = String(node?.attributes?.operator_id || "").toLowerCase();
  if (type === "attention" || operatorId === "attention") {
    const attentionKind = node?.attributes?.attention_kind || "gqa";
    if (attentionKind === "linear") return linearAttentionMacs(config, options);
    if (attentionKind === "qsa") return qsaAttentionMacs(config, options);
    if (attentionKind === "sparse" && config?.modelType === "minimax_m3_vl") return minimaxSparseAttentionMacs(config, options);
    if (attentionKind === "dsv4") {
      const layerMatch = String(node?.id || "").match(/(?:^|\.)(?:layers|decoder)\.(\d+)/);
      return deepseekV4AttentionMacs(config, { ...options, layerIndex: layerMatch ? Number(layerMatch[1]) : 0 });
    }
    return attentionMacs(config, options);
  }
  if (isLinear(node)) return linearMacs(node, options);
  const output = node?.output_shape || node?.attributes?.output_shape;
  return Array.isArray(output) ? tensorElements(output, options) : 0;
}

function computeMacsForNode(node, config, options = {}) {
  const type = String(node?.type || "").toLowerCase();
  const operatorId = String(node?.attributes?.operator_id || "").toLowerCase();
  if (type === "attention" || operatorId === "attention") return nodeMacs(node, config, options);
  if (isLinear(node)) return linearMacs(node, options);
  return 0;
}

function macsSource(node, config, options = {}) {
  const type = String(node?.type || "").toLowerCase();
  const operatorId = String(node?.attributes?.operator_id || "").toLowerCase();
  if (type === "attention" || operatorId === "attention") return "formula";
  if (!isLinear(node)) return "not-compute";
  if (node?.weight_shapes?.qweight && !node?.attributes?.logical_weight_shape) return "unknown";
  if (hasLogicalLinearShape(node)) return "checkpoint-shape";
  return derivedLinearMacs(node, options) == null ? "unknown" : "config-derived-shape";
}

export function computeNodeCosts(root, config, options = {}) {
  const rows = [];
  walkStructure(root, ({ node, path, multiplier }) => {
    const modulePath = node?.id || path;
    const layerMatch = modulePath.match(/(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/);
    const layerIndex = layerMatch ? Number(layerMatch[1]) : null;
    const layerKind = layerIndex != null ? config?.layerSchedule?.[layerIndex] : null;
    const routedExpert = /(?:^|\.)(?:experts|expert_mlp)(?:\.|$)/.test(modulePath);
    const expertFraction = routedExpert && layerKind !== "dense" && config?.experts && config?.expertsPerToken
      ? config.expertsPerToken / config.experts
      : 1;
    const costOptions = { ...options, expertFraction };
    const ownMacs = nodeMacs(node, config, costOptions);
    const computeMacs = computeMacsForNode(node, config, costOptions);
    const own = ownMacs == null ? null : ownMacs * multiplier;
    const compute = computeMacs == null ? null : computeMacs * multiplier;
    rows.push({ path, node, multiplier, macs: own, compute_macs: compute,
      macs_source: macsSource(node, config, costOptions),
      weightBytes: nodeWeightBytes(node) * multiplier,
      estimate_status: compute == null ? "unknown" : "estimated" });
  });
  return rows;
}
