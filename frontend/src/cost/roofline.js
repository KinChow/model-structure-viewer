import { resolveEfficiency } from "./efficiency.js";

function positive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function peakFlops(chip, dtype) {
  return chip?.peak_flops?.[String(dtype || "bf16").toLowerCase()];
}

/**
 * 计算单个模块的三类理论时间并给出瓶颈分类。
 * 来源：llm-analysis 的效率因子设计；仅用于互相比大小，不展示绝对延迟预测。
 * @param {{macs?: number|null, weightBytes?: number, actInBytes?: number, actOutBytes?: number, commBytes?: number}} cost
 * @param {object} chip 芯片规格
 * @param {{dtype?: string, efficiency?: object}} options
 */
export function classifyRoofline(cost = {}, chip = {}, options = {}) {
  const dtype = options.dtype || "bf16";
  const eta = resolveEfficiency(chip, options.efficiency);
  const missing = [];
  const flops = peakFlops(chip, dtype);
  const memoryBandwidth = chip.memory_bandwidth;
  const intraLink = chip.interconnect?.intra_node?.bandwidth;
  const interLink = chip.interconnect?.inter_node?.bandwidth;

  if (!positive(cost.macs)) missing.push("macs");
  if (!positive(flops)) missing.push(`peak_flops.${String(dtype).toLowerCase()}`);
  const bytesMoved = (cost.weightBytes || 0) + (cost.actInBytes || 0) + (cost.actOutBytes || 0);
  if (!positive(bytesMoved)) missing.push("bytes_moved");
  if (!positive(memoryBandwidth)) missing.push("memory_bandwidth");

  const computeTime = positive(cost.macs) && positive(flops) ? (2 * cost.macs) / (flops * eta.flops) : null;
  const memoryTime = positive(bytesMoved) && positive(memoryBandwidth)
    ? bytesMoved / (memoryBandwidth * eta.hbm)
    : null;
  const commBytes = cost.commBytes || 0;
  const link = options.interNode ? interLink : intraLink;
  const commEta = options.interNode ? eta.comm : eta.intra_node_comm;
  const commTime = positive(commBytes) && positive(link) ? commBytes / (link * commEta) : null;
  const candidates = [
    ["compute", computeTime],
    ["memory", memoryTime],
    ["comm", commTime],
  ].filter(([, value]) => value != null);
  const bound = candidates.length > 0 ? candidates.reduce((best, current) => current[1] > best[1] ? current : best)[0] : "unknown";

  return {
    dtype,
    bound,
    arithmeticIntensity: computeTime != null && positive(bytesMoved) ? (2 * cost.macs) / bytesMoved : null,
    ridgePoint: positive(flops) && positive(memoryBandwidth) ? (flops * eta.flops) / (memoryBandwidth * eta.hbm) : null,
    times: { compute: computeTime, memory: memoryTime, comm: commTime },
    bytesMoved,
    missing,
    efficiency: eta,
  };
}
