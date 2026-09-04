import { bytesPerDtype, nodeWeightBytes, memoryBreakdown } from "./memory.js";
import { computeNodeCosts } from "./compute.js";

export function aggregateCost({ root, config, parameterCount, batch = 1, sequence = 1, phase = "prefill",
  kvBytes = 2, activationPeak, runtimeConst, commBuffer } = {}) {
  const weightBytes = parameterCount
    ? Object.entries(parameterCount).reduce((sum, [dtype, count]) => sum + count * bytesPerDtype(dtype), 0)
    : sumNodeWeights(root);
  const memory = memoryBreakdown({ weightBytes, config, batch, tokens: sequence, kvBytes,
    activationPeak, runtimeConst, commBuffer });
  const nodes = computeNodeCosts(root, config, { batch, sequence, phase });
  return { phase, batch, sequence, memory, nodes, totalMacs: nodes.reduce((sum, row) => sum + row.macs, 0),
    assumptions: { theoretical: true, activationPeak, runtimeConst, commBuffer, kvBytes } };
}

function sumNodeWeights(root) {
  let total = 0;
  function visit(node) { total += nodeWeightBytes(node); (node?.children || []).forEach(visit); }
  if (root) visit(root);
  return total;
}
