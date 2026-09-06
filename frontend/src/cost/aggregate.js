import { bytesPerDtype, nodeWeightBytes, memoryBreakdown } from "./memory.js";
import { computeNodeCosts } from "./compute.js";
import { derivedWeightBytes, derivedWeightParameters } from "./derivedWeights.js";

export function aggregateCost({ root, config, parameterCount, batch = 1, sequence = 1, phase = "prefill",
  kvBytes = 2, activationPeak, runtimeConst, commBuffer, weightBytesPerParameter } = {}) {
  const hasParameterCount = parameterCount && Object.keys(parameterCount).length > 0;
  const nodeWeights = sumNodeWeights(root);
  const naturalWeightBytes = hasParameterCount
    ? Object.entries(parameterCount).reduce((sum, [dtype, count]) => sum + count * bytesPerDtype(dtype), 0)
    : nodeWeights > 0 ? nodeWeights : derivedWeightBytes(config, 2);
  const parameterTotal = hasParameterCount
    ? Object.values(parameterCount).reduce((sum, count) => sum + count, 0)
    : derivedWeightParameters(config);
  const hasWeightOverride = typeof weightBytesPerParameter === "number" && weightBytesPerParameter > 0;
  const weightBytes = hasWeightOverride ? parameterTotal * weightBytesPerParameter : naturalWeightBytes;
  const memory = memoryBreakdown({ weightBytes, config, batch, tokens: sequence, kvBytes,
    activationPeak, runtimeConst, commBuffer });
  const nodes = computeNodeCosts(root, config, { batch, sequence, phase });
  const totalMacs = nodes.reduce((sum, row) => sum + (row.compute_macs ?? 0), 0);
  const forwardTokens = batch * (phase === "decode" ? 1 : sequence);
  return { phase, batch, sequence, memory, weightSource: hasWeightOverride ? "what-if" : hasParameterCount ? "checkpoint" : nodeWeights > 0 ? "node" : "derived", nodes, totalMacs, totalFlops: totalMacs * 2,
    macsPerToken: forwardTokens > 0 ? totalMacs / forwardTokens : null,
    flopsPerToken: forwardTokens > 0 ? (totalMacs * 2) / forwardTokens : null,
    macsSources: summarizeMacsSources(nodes),
    assumptions: { theoretical: true, activationPeak, runtimeConst, commBuffer, kvBytes } };
}

function summarizeMacsSources(nodes) {
  return nodes.reduce((counts, row) => {
    const source = row.macs_source || "unknown";
    counts[source] = (counts[source] || 0) + 1;
    return counts;
  }, {});
}

function sumNodeWeights(root) {
  let total = 0;
  function visit(node, multiplier = 1) {
    total += nodeWeightBytes(node) * multiplier;
    const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
    const childHasExplicitRepeat = (node?.children || []).some((child) => Number.isFinite(child?.repeat));
    (node?.children || []).forEach((child) => visit(child, multiplier * (childHasExplicitRepeat ? 1 : repeat)));
  }
  if (root) visit(root);
  return total;
}
