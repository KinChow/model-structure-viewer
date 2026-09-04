// Per-module MAC estimates for inference. They intentionally expose assumptions rather than predict latency.

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

// ref: llm-analysis LLMAnalysis.get_num_flops_fwd_per_layer_linear
export function linearMacs(node, { batch, sequence, phase, expertFraction = 1 } = {}) {
  const shape = Object.values(node?.weight_shapes || {}).find((value) => Array.isArray(value) && value.length >= 2);
  return shape ? tokensFor({ batch, sequence, phase }) * product(shape) * expertFraction : 0;
}

// ref: llm-analysis LLMAnalysis.get_num_flops_fwd_per_layer_attn
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
    const own = nodeMacs(node, config, options) * multiplier;
    rows.push({ path, node, macs: own });
    (node?.children || []).forEach((child, index) => visit(child, `${path}.${index}`, multiplier * repeat));
  }
  if (root) visit(root);
  return rows;
}
