import { bytesPerDtype, nodeWeightBytes, memoryBreakdown } from "./memory.js";
import { computeNodeCosts } from "./compute.js";
import { derivedWeightBytes, derivedWeightParameters, derivedBufferBytes } from "./derivedWeights.js";
import { walkStructure } from "./traverse.js";
import { quantizationConfigOf, isQuantizedPath, quantLinearWeightBytes } from "./quantBytes.js";

// 量化命中且可枚举 [out,in] 的算子（线性族）。embedding/lm_head/hash 路由表等
// 是否量化由 config 的 dynamic / modules_to_not_convert 声明决定（isQuantizedPath）。
const QUANTIZABLE_OPS = new Set(["linear", "mla_query_compress", "mla_kv_compress"]);

function positiveWidth(shape) {
  if (!Array.isArray(shape)) return null;
  let width = 1;
  let any = false;
  for (const value of shape) {
    if (Number.isFinite(value) && value > 0) { width *= value; any = true; }
  }
  return any ? width : null;
}

/**
 * 无 checkpoint 时的**逐矩阵**量化容量：枚举树上全部线性族叶子的 [out, in]
 *（derivedLinearShape 同口径：output/input shape 的正维乘积），命中 quant 方案
 * 的按 quantLinearWeightBytes 精确计（权重 + scale + zeros），返回
 * 「被量化矩阵的元素数」与「它们的精确字节」。其余参数（norm/embed/mtp/vision
 * 与未命中量化的矩阵）仍按派生标量字节宽计。
 */
function quantizedMatrixBytes(root, graph, quant) {
  let elements = 0;
  let bytes = 0;
  walkStructure(root, ({ node, multiplier }) => {
    const op = String(node?.attributes?.operator_id || "").toLowerCase();
    if (!QUANTIZABLE_OPS.has(op)) return;
    const path = String(node?.canonical_id ?? node?.id ?? "");
    if (!isQuantizedPath(path, quant)) return;
    const out = positiveWidth(node?.output_shape);
    const inn = positiveWidth(node?.input_shape);
    if (out == null || inn == null) return;
    const matrixBytes = quantLinearWeightBytes({ out, inn, quant });
    if (matrixBytes == null) return;
    elements += out * inn * multiplier;
    bytes += matrixBytes * multiplier;
  }, graph);
  return { elements, bytes };
}

export function aggregateCost({ root, graph, config, parameterCount, batch = 1, sequence = 1, phase = "prefill", visionTokens,
  kvBytes = 2, activationPeak, runtimeConst, commBuffer, weightBytesPerParameter } = {}) {
  const hasParameterCount = parameterCount && Object.keys(parameterCount).length > 0;
  const nodeWeights = sumNodeWeights(root, graph);
  const quant = quantizationConfigOf(config);
  const naturalWeightBytes = hasParameterCount
    ? Object.entries(parameterCount).reduce((sum, [dtype, count]) => sum + count * bytesPerDtype(dtype), 0)
    : nodeWeights > 0 ? nodeWeights : quantCapacityBytes(root, graph, config, quant);
  const parameterTotal = hasParameterCount
    ? Object.values(parameterCount).reduce((sum, count) => sum + count, 0)
    : derivedWeightParameters(config);
  const hasWeightOverride = typeof weightBytesPerParameter === "number" && weightBytesPerParameter > 0;
  const weightBytes = hasWeightOverride ? parameterTotal * weightBytesPerParameter : naturalWeightBytes;
  const memory = memoryBreakdown({ weightBytes, bufferBytes: derivedBufferBytes(config), config, batch, tokens: sequence, kvBytes,
    activationPeak, runtimeConst, commBuffer });
  const nodes = computeNodeCosts(root, config, { batch, sequence, phase, graph, visionTokens: visionTokens ?? undefined });
  const unknownComputePaths = nodes
    .filter((row) => row.compute_macs == null)
    .map((row) => row.path);
  const knownMacs = nodes.reduce((sum, row) => sum + (row.compute_macs ?? 0), 0);
  const computeComplete = unknownComputePaths.length === 0;
  const totalMacs = computeComplete ? knownMacs : null;
  const forwardTokens = batch * (phase === "decode" ? 1 : sequence);
  const derivedSource = config?.quantizationBytesPerParameter > 0 || quant ? "derived-quantized" : "derived";
  const actions = summarizeActions(nodes, computeComplete);
  return { phase, batch, sequence, memory, weightSource: hasWeightOverride ? "what-if" : hasParameterCount ? "checkpoint" : nodeWeights > 0 ? "node" : derivedSource, nodes, totalMacs, totalFlops: totalMacs == null ? null : totalMacs * 2,
    actions,
    knownMacs, computeComplete, unknownComputePaths,
    macsPerToken: totalMacs != null && forwardTokens > 0 ? totalMacs / forwardTokens : null,
    flopsPerToken: totalMacs != null && forwardTokens > 0 ? (totalMacs * 2) / forwardTokens : null,
    macsSources: summarizeMacsSources(nodes),
    assumptions: { theoretical: true, activationPeak, runtimeConst, commBuffer, kvBytes, quantization: config?.quantizationMethod || null,
      weightBytesPerParameter: config?.quantizationBytesPerParameter || null } };
}

// 模型级动作向量汇总（§3.4：ERT 与 counts 分离）；任一叶子未实现则整体 unknown。
function summarizeActions(nodes, computeComplete) {
  if (!computeComplete) return null;
  return nodes.reduce((acc, row) => {
    if (!row.actions) return acc;
    acc.matrix += row.actions.matrix ?? 0;
    // computeDtype 桶（N2-1）：声明了非默认精度的 matrix 单列——actions.matrix
    // 仍是**总量**（含该桶），roofline 据此把矩阵时间拆成两段费率。
    if (row.actions.computeDtype) {
      acc.computeDtypes = acc.computeDtypes || {};
      acc.computeDtypes[row.actions.computeDtype] = (acc.computeDtypes[row.actions.computeDtype] || 0) + (row.actions.matrix ?? 0);
    }
    acc.vector += row.actions.vector ?? 0;
    acc.sfu += row.actions.sfu ?? 0;
    acc.weights += row.actions.bytes.weights ?? 0;
    acc.actIn += row.actions.bytes.actIn ?? 0;
    acc.actOut += row.actions.bytes.actOut ?? 0;
    return acc;
  }, { matrix: 0, vector: 0, sfu: 0, weights: 0, actIn: 0, actOut: 0 });
}

function summarizeMacsSources(nodes) {
  return nodes.reduce((counts, row) => {
    const source = row.macs_source || "unknown";
    counts[source] = (counts[source] || 0) + 1;
    return counts;
  }, {});
}

/**
 * 量化容量（无 checkpoint、树可枚举时）：
 *   base = 全部参数按 bf16（2B，含 fp32 参数修正）；被量化矩阵换为其精确
 *   字节（fp8 1B + scale / gptq 0.5B + scales+zeros）。
 * **排除矩阵（modules_to_not_convert / dynamic 命中）必须留在 bf16 桶** ——
 * 它们以未量化精度运行，若留在标量量化宽（如 fp8 的 1B）会把排除项算小
 *（M2.7 实测 lm_head/gate 排除后出现 -150,048 的反常下降，2026-09-09 修正）。
 * 树不可枚举（bare config，无 root 结构）时退回标量
 * quantizationBytesPerParameter —— 没有 [out,in] 就没有 scale 形状与排除
 * 归属，这是信息极限而非建模缺口（登记于 operators_reference §7）。
 */
function quantCapacityBytes(root, graph, config, quant) {
  const base = derivedWeightBytes(config, 2);
  if (!quant) {
    const scalarBytes = config?.quantizationBytesPerParameter || 2;
    return scalarBytes !== 2 ? derivedWeightBytes(config, scalarBytes) : base;
  }
  const { elements, bytes } = quantizedMatrixBytes(root, graph, quant);
  if (elements <= 0) return base;
  return base - elements * 2 + bytes;
}

function sumNodeWeights(root, graph) {
  let total = 0;
  walkStructure(root, ({ node, multiplier }) => {
    total += nodeWeightBytes(node) * multiplier;
  }, graph);
  return total;
}
