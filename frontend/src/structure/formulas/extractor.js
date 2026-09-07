// extractor.js —— node → counts ctx 提取器（docs/details/cost_counts.md 提取器规格）。
//
// 原则（principles §3.2 / §4.3）：
// - 查表优先：节点已有 weight_shapes / input_shape / output_shape / attributes；
//   提取器只补 phase 的 T/S、变体参数、expertFraction。
// - 分派只基于 type / attributes.operator_id / 结构化 attributes 与节点路径（结构化 id），
//   禁止显示名（node.name）参与分派。
// - 不 import cost 层；bytesPerElement 由调用方传入（W5 接线点）。
// - 返回单实例 counts；repeat 倍乘由 walker 的 multiplier 处理（与旧链同）。

import {
  linearCounts,
} from "./counts.js";

// 路径正则全仓统一处（旧 compute.js/parallel.js 三种变体收敛于此）
export const LAYER_INDEX_RE = /(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/;
export const ROUTED_EXPERT_RE = /(?:^|\.)(?:experts|expert_mlp)(?:\.|$)/;

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

/**
 * 计算单个算子节点的动作向量。
 * @returns 动作向量；matrix 无法确定时返回 null（调用方计入 unknownComputePaths）。
 */
export function countsForNode(node, env = {}) {
  const { config, options = {}, path = "", bytesPerElement = 2 } = env;
  const operatorId = String(node?.attributes?.operator_id || "").toLowerCase();
  const type = String(node?.type || "").toLowerCase();
  const vision = node?.attributes?.modality === "vision";
  const tokens = tokensFor({
    batch: options.batch ?? 1,
    sequence: options.sequence ?? 1,
    phase: options.phase ?? "prefill",
    vision,
    visionTokens: config?.visionTokens || 1,
  });

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
    default:
      // T1 仅覆盖线性族；其余算子在 T2/T3 接入前返回 null（计入 unknown，不计 0）。
      return null;
  }
}
