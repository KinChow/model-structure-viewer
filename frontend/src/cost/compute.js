// 推理场景的逐模块 MACs 估算；显式暴露假设，不用于预测延迟。

import { product } from "./memory.js";

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

export function nodeMacs(node, config, options = {}) {
  if (isLinear(node)) return linearMacs(node, options);
  const label = `${node?.type || ""} ${node?.name || ""} ${node?.attributes?.operator_id || ""}`.toLowerCase();
  if (label.includes("attention") || label.includes("attn")) return attentionMacs(config, options);
  const output = node?.output_shape || node?.attributes?.output_shape;
  return Array.isArray(output) ? product(output.filter((value) => value >= 0)) * tokensFor(options) : 0;
}

export function computeNodeCosts(root, config, options = {}) {
  const rows = [];
  function visit(node, path = "root", multiplier = 1) {
    const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
    const modulePath = node?.id || path;
    const layerMatch = modulePath.match(/(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/);
    const layerIndex = layerMatch ? Number(layerMatch[1]) : null;
    const layerKind = layerIndex != null ? config?.layerSchedule?.[layerIndex] : null;
    const routedExpert = /(?:^|\.)(?:experts|expert_mlp)(?:\.|$)/.test(modulePath);
    const expertFraction = routedExpert && layerKind === "moe" && config?.experts && config?.expertsPerToken
      ? config.expertsPerToken / config.experts
      : 1;
    const ownMacs = nodeMacs(node, config, { ...options, expertFraction });
    const own = ownMacs == null ? null : ownMacs * multiplier;
    rows.push({ path, node, macs: own, estimate_status: own == null ? "unknown" : "estimated" });
    (node?.children || []).forEach((child, index) => visit(child, `${path}.${index}`, multiplier * repeat));
  }
  if (root) visit(root);
  return rows;
}
