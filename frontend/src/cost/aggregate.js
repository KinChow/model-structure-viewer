import { bytesPerDtype, nodeWeightBytes, memoryBreakdown } from "./memory.js";
import { computeNodeCosts } from "./compute.js";

export function aggregateCost({ root, config, parameterCount, batch = 1, sequence = 1, phase = "prefill",
  kvBytes = 2, activationPeak, runtimeConst, commBuffer } = {}) {
  const hasParameterCount = parameterCount && Object.keys(parameterCount).length > 0;
  const weightBytes = hasParameterCount
    ? Object.entries(parameterCount).reduce((sum, [dtype, count]) => sum + count * bytesPerDtype(dtype), 0)
    : sumNodeWeights(root);
  const memory = memoryBreakdown({ weightBytes, config, batch, tokens: sequence, kvBytes,
    activationPeak, runtimeConst, commBuffer });
  const nodes = computeNodeCosts(root, config, { batch, sequence, phase });
  return { phase, batch, sequence, memory, nodes, totalMacs: nodes.reduce((sum, row) => sum + (row.macs ?? 0), 0),
    assumptions: { theoretical: true, activationPeak, runtimeConst, commBuffer, kvBytes } };
}

function sumNodeWeights(root) {
  let total = 0;
  function visit(node, multiplier = 1) {
    total += nodeWeightBytes(node) * multiplier;
    const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
    (node?.children || []).forEach((child) => visit(child, multiplier * repeat));
  }
  if (root) visit(root);
  return total;
}
