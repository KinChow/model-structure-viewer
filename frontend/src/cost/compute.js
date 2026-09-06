// 推理场景的逐模块 MACs 估算；显式暴露假设，不用于预测延迟。

import { nodeWeightBytes, product, tensorElements } from "./memory.js";

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

// 来源：llm-analysis 的 LLMAnalysis.get_num_flops_fwd_per_layer_linear。
export function linearMacs(node, { batch, sequence, phase, expertFraction = 1 } = {}) {
  // GPTQ/AWQ 的 packed shape 是存储形状，不是逻辑矩阵乘形状。
  // 没有明确的逻辑形状时，不输出看似合理但实际错误的 MACs。
  if (node?.weight_shapes?.qweight && !node?.attributes?.logical_weight_shape) return null;
  const shape = Object.values(node?.weight_shapes || {}).find((value) => Array.isArray(value) && value.length >= 2);
  const logicalShape = node?.attributes?.logical_weight_shape || shape;
  return logicalShape ? tokensFor({ batch, sequence, phase }) * product(logicalShape) * expertFraction : 0;
}

// 来源：llm-analysis 的 LLMAnalysis.get_num_flops_fwd_per_layer_attn。
export function attentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const lengthTerm = phase === "decode" ? sequence : sequence ** 2;
  return batch * heads * lengthTerm * (qk + value);
}

export function linearAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  if (config?.linearAttentionMode === "glm5_next") return glm5NextLinearAttentionMacs(config, { batch, sequence, phase });
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

export function qsaAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const selected = Math.min(sequence, config?.indexerBudget || sequence);
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  return queryTokens * heads * selected * (qk + value);
}

export function nodeMacs(node, config, options = {}) {
  const type = String(node?.type || "").toLowerCase();
  const operatorId = String(node?.attributes?.operator_id || "").toLowerCase();
  if (type === "attention" || operatorId === "attention") {
    const attentionKind = node?.attributes?.attention_kind || "gqa";
    if (attentionKind === "linear") return linearAttentionMacs(config, options);
    if (attentionKind === "qsa") return qsaAttentionMacs(config, options);
    return attentionMacs(config, options);
  }
  if (isLinear(node)) return linearMacs(node, options);
  const output = node?.output_shape || node?.attributes?.output_shape;
  return Array.isArray(output) ? tensorElements(output, options) : 0;
}

export function computeNodeCosts(root, config, options = {}) {
  const rows = [];
  function visit(node, path = "root", multiplier = 1) {
    const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
    const childHasExplicitRepeat = (node?.children || []).some((child) => Number.isFinite(child?.repeat));
    const modulePath = node?.id || path;
    const layerMatch = modulePath.match(/(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/);
    const layerIndex = layerMatch ? Number(layerMatch[1]) : null;
    const layerKind = layerIndex != null ? config?.layerSchedule?.[layerIndex] : null;
    const routedExpert = /(?:^|\.)(?:experts|expert_mlp)(?:\.|$)/.test(modulePath);
    const expertFraction = routedExpert && layerKind !== "dense" && config?.experts && config?.expertsPerToken
      ? config.expertsPerToken / config.experts
      : 1;
    const ownMacs = nodeMacs(node, config, { ...options, expertFraction });
    const own = ownMacs == null ? null : ownMacs * multiplier;
    rows.push({ path, node, multiplier, macs: own, weightBytes: nodeWeightBytes(node) * multiplier,
      estimate_status: own == null ? "unknown" : "estimated" });
    const childMultiplier = multiplier * (childHasExplicitRepeat ? 1 : repeat);
    (node?.children || []).forEach((child, index) => visit(child, `${path}.${index}`, childMultiplier));
  }
  if (root) visit(root);
  return rows;
}
