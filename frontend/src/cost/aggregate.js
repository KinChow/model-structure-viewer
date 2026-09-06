import { bytesPerDtype, nodeWeightBytes, memoryBreakdown } from "./memory.js";
import { computeNodeCosts } from "./compute.js";
import { derivedWeightBytes, derivedWeightParameters } from "./derivedWeights.js";
import { walkStructure } from "./traverse.js";

export function aggregateCost({ root, graph, config, parameterCount, batch = 1, sequence = 1, phase = "prefill",
  kvBytes = 2, activationPeak, runtimeConst, commBuffer, weightBytesPerParameter } = {}) {
  const hasParameterCount = parameterCount && Object.keys(parameterCount).length > 0;
  const nodeWeights = sumNodeWeights(root, graph);
  const naturalWeightBytes = hasParameterCount
    ? Object.entries(parameterCount).reduce((sum, [dtype, count]) => sum + count * bytesPerDtype(dtype), 0)
    : nodeWeights > 0 ? nodeWeights : derivedWeightBytes(config, config?.quantizationBytesPerParameter || 2);
  const parameterTotal = hasParameterCount
    ? Object.values(parameterCount).reduce((sum, count) => sum + count, 0)
    : derivedWeightParameters(config);
  const hasWeightOverride = typeof weightBytesPerParameter === "number" && weightBytesPerParameter > 0;
  const weightBytes = hasWeightOverride ? parameterTotal * weightBytesPerParameter : naturalWeightBytes;
  const memory = memoryBreakdown({ weightBytes, config, batch, tokens: sequence, kvBytes,
    activationPeak, runtimeConst, commBuffer });
  const nodes = computeNodeCosts(root, config, { batch, sequence, phase, graph });
  const unknownComputePaths = nodes
    .filter((row) => row.compute_macs == null)
    .map((row) => row.path);
  const knownMacs = nodes.reduce((sum, row) => sum + (row.compute_macs ?? 0), 0);
  const computeComplete = unknownComputePaths.length === 0;
  const totalMacs = computeComplete ? knownMacs : null;
  const forwardTokens = batch * (phase === "decode" ? 1 : sequence);
  const derivedSource = config?.quantizationBytesPerParameter > 0 ? "derived-quantized" : "derived";
  return { phase, batch, sequence, memory, weightSource: hasWeightOverride ? "what-if" : hasParameterCount ? "checkpoint" : nodeWeights > 0 ? "node" : derivedSource, nodes, totalMacs, totalFlops: totalMacs == null ? null : totalMacs * 2,
    knownMacs, computeComplete, unknownComputePaths,
    macsPerToken: totalMacs != null && forwardTokens > 0 ? totalMacs / forwardTokens : null,
    flopsPerToken: totalMacs != null && forwardTokens > 0 ? (totalMacs * 2) / forwardTokens : null,
    macsSources: summarizeMacsSources(nodes),
    assumptions: { theoretical: true, activationPeak, runtimeConst, commBuffer, kvBytes, quantization: config?.quantizationMethod || null,
      weightBytesPerParameter: config?.quantizationBytesPerParameter || null } };
}

function summarizeMacsSources(nodes) {
  return nodes.reduce((counts, row) => {
    const source = row.macs_source || "unknown";
    counts[source] = (counts[source] || 0) + 1;
    return counts;
  }, {});
}

function sumNodeWeights(root, graph) {
  let total = 0;
  walkStructure(root, ({ node, multiplier }) => {
    total += nodeWeightBytes(node) * multiplier;
  }, graph);
  return total;
}
