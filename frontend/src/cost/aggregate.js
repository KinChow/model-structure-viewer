import { bytesPerDtype, graphShapedWeightBytes, graphWeightCapacity, memoryBreakdown } from "./memory.js";
import { computeNodeCosts } from "./compute.js";
import { walkStructure } from "./traverse.js";
import { quantizationConfigOf, isQuantizedPath, quantLinearWeightBytes } from "./quantBytes.js";

/**
 * 无 checkpoint 时的**逐矩阵**量化容量：枚举树上全部线性族叶子的 [out, in]
 *（derivedLinearShape 同口径：output/input shape 的正维乘积），命中 quant 方案
 * 的按 quantLinearWeightBytes 精确计（权重 + scale + zeros），返回
 * 「被量化矩阵的元素数」与「它们的精确字节」。其余参数（norm/embed/mtp/vision
 * 与未命中量化的矩阵）仍按声明字节宽计。
 *
 * N2-4 W-B：带 weightMatrices 声明的叶子优先按声明组枚举（feature flag =
 * 声明存在，无声明叶逐位走原路径）。这补上了此前最大的枚举缺口——MoE 专家
 * 融合叶（fused_moe_mlp）的 gate/up/down 3×E 个矩阵不在 linear 族内，整块
 * 留在 bf16 桶（25 个量化 MoE 模型实测容量 ≈2× 偏高）。
 */
// P7（步骤 7）：root 入参退役——量化枚举直接遍历 Graph IR。
function quantizedMatrixBytes(graph, quant) {
  let elements = 0;
  let bytes = 0;
  walkStructure(graph, ({ node, resident }) => {
    const path = String(node?.canonical_id ?? node?.id ?? "");
    const declaration = node?.attributes?.weightMatrices;
    if (Array.isArray(declaration) && declaration.length > 0) {
      // 排除矩阵（modules_to_not_convert / dynamic 命中）留 bf16 基桶，与
      // linear 族同一判定（isQuantizedPath），分桶与枚举同源。
      if (!isQuantizedPath(path, quant)) return;
      for (const group of declaration) {
        // 量化方案只作用于 **Linear 权重矩阵**（vLLM/SGLang 的 quant config
        // targets: ["Linear"]，量化 norm scale 与 bias 都不在其中）。声明用显式
        // quantizable=false 标记这类参数，不用维度大小猜——K3 的
        // attn_residual res_proj 就是 out=1 的真 GEMM（[1, 7168] 打分投影），
        // 按"维度>1"过滤会误伤它。
        if (group.quantizable === false || group.shared) continue;
        const matrixBytes = quantLinearWeightBytes({ out: group.out, inn: group.in, quant });
        // 无法计算的 quant 方案留在基桶（诚实缺项，不伪造 1B 标量宽）
        if (matrixBytes == null) continue;
        const instances = (group.count ?? 1) * (group.matrices ?? 1) * resident;
        elements += group.out * group.in * instances;
        bytes += matrixBytes * instances;
      }
      return;
    }
    // P5：QUANTIZABLE_OPS 回退已删 —— 量化枚举只消费声明组（weightMatrices
    // 是权重归属唯一入口）。无声明叶不入枚举，留在 bf16 基桶（保守高估）；
    // 覆盖率护栏（P2 棘轮=0）保证内置模型不会走到这里。
  });
  return { elements, bytes };
}

/** 仅供部署默认策略使用的内存入口；与 Cost 复用同一份驻留账本。 */
export function aggregateModelMemory({ graph, config, parameterCount, batch = 1, sequence = 1,
  kvBytes = 2, weightBytesPerParameter, frameworkProfile = "neutral", speculative = {} } = {}) {
  const hasParameterCount = parameterCount && Object.keys(parameterCount).length > 0;
  const shapedWeights = graphShapedWeightBytes(graph);
  const declared = graphWeightCapacity(graph);
  const quant = quantizationConfigOf(config);
  const naturalWeightBytes = hasParameterCount
    ? Object.entries(parameterCount).reduce((sum, [dtype, count]) => sum + count * bytesPerDtype(dtype), 0)
    : shapedWeights > 0 ? shapedWeights : quantCapacityBytes(graph, declared, quant, config);
  const parameterTotal = hasParameterCount
    ? Object.values(parameterCount).reduce((sum, count) => sum + count, 0)
    : declared.elements;
  const hasWeightOverride = typeof weightBytesPerParameter === "number" && weightBytesPerParameter > 0;
  const weightBytes = hasWeightOverride ? parameterTotal * weightBytesPerParameter : naturalWeightBytes;
  const memory = memoryBreakdown({
    weightBytes, graph, batch, tokens: sequence, kvBytes, frameworkProfile, config, speculative,
  });
  const graphSource = quant ? "derived-quantized" : "node";
  return {
    memory,
    weightSource: hasWeightOverride ? "what-if" : hasParameterCount ? "checkpoint" : (shapedWeights > 0 || declared.bytes > 0) ? graphSource : "empty",
  };
}

export function aggregateCost({ graph, config, parameterCount, batch = 1, sequence = 1, phase = "prefill", visionTokens,
  kvBytes = 2, weightBytesPerParameter, frameworkProfile = "neutral", speculative = {} } = {}) {
  const { memory, weightSource } = aggregateModelMemory({
    graph, config, parameterCount, batch, sequence, kvBytes, weightBytesPerParameter, frameworkProfile, speculative,
  });
  const nodes = computeNodeCosts(graph, config, { batch, sequence, phase, visionTokens: visionTokens ?? undefined, frameworkProfile });
  const unknownComputePaths = nodes
    .filter((row) => row.compute_macs == null)
    .map((row) => row.path);
  const knownMacs = nodes.reduce((sum, row) => sum + (row.compute_macs ?? 0), 0);
  const computeComplete = unknownComputePaths.length === 0;
  const totalMacs = computeComplete ? knownMacs : null;
  const forwardTokens = batch * (phase === "decode" ? 1 : sequence);
  const actions = summarizeActions(nodes, computeComplete);
  return { phase, batch, sequence, memory, weightSource, nodes, totalMacs, totalFlops: totalMacs == null ? null : totalMacs * 2,
    actions,
    knownMacs, computeComplete, unknownComputePaths,
    macsPerToken: totalMacs != null && forwardTokens > 0 ? totalMacs / forwardTokens : null,
    flopsPerToken: totalMacs != null && forwardTokens > 0 ? (totalMacs * 2) / forwardTokens : null,
    macsSources: summarizeMacsSources(nodes),
    assumptions: { theoretical: true, kvBytes, frameworkProfile, quantization: config?.quantizationMethod || null,
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
    acc.kvRead += row.actions.bytes.kvRead ?? 0;
    acc.indexRead += row.actions.bytes.indexRead ?? 0;
    return acc;
  }, { matrix: 0, vector: 0, sfu: 0, weights: 0, actIn: 0, actOut: 0, kvRead: 0, indexRead: 0, computeDtypes: {} });
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
 *   base = 图上声明的驻留字节（bf16 + 已标 param_dtype 的 fp32）；被量化矩阵换为其精确
 *   字节（fp8 1B + scale / gptq 0.5B + scales+zeros）。
 * **排除矩阵（modules_to_not_convert / dynamic 命中）必须留在 bf16 桶** ——
 * 它们以未量化精度运行，若留在标量量化宽（如 fp8 的 1B）会把排除项算小
 *（M2.7 实测 lm_head/gate 排除后出现 -150,048 的反常下降，2026-09-09 修正）。
 * 无图 → 0，不编造闭式（原则 §3.8）。
 */
function quantCapacityBytes(graph, declared, quant, config) {
  const scalarBytes = config?.quantizationBytesPerParameter || 2;
  const base = !quant && scalarBytes !== 2
    ? graphWeightCapacity(graph, { fallbackBytes: scalarBytes }).bytes
    : declared.bytes;
  if (!quant) return base;
  const { elements, bytes } = quantizedMatrixBytes(graph, quant);
  if (elements <= 0) return base;
  return base - elements * 2 + bytes;
}
