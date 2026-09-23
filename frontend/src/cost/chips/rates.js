// rates.js —— ERT 的 R（费率表）：芯片规格 × 效率因子 → 四列单位费率（§3.4）。
// 动作向量与费率的交点即理论时间；本模块不含任何模型知识。
// 列：matrixPerSecond（MACs/s，peak_flops·η/2）、vectorPerSecond（vector_flops·η）、
//     sfuPerSecond（sfu_ops·η）、bytesPerSecond（memory_bandwidth·η_hbm）、
//     通信两列（intra/inter·η）。
// 语义映射说明：昇腾类无独立 sfu_ops 规格时，允许声明 sfu→vector rate
// （sfuRateSource: "vector"），这是语义映射而非估算（§3.7）。
import { resolveEfficiency } from "../efficiency.js";
import { missingFields } from "./coverage.js";

function positive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/**
 * @param {object} chip 芯片规格
 * @param {{dtype?: string, efficiency?: object}} options
 * @returns {{matrixPerSecond: number|null, vectorPerSecond: number|null, sfuPerSecond: number|null,
 *            bytesPerSecond: number|null, intraNodeBytesPerSecond: number|null, interNodeBytesPerSecond: number|null,
 *            missing: string[], efficiency: object}}
 */
export function chipRates(chip = {}, options = {}) {
  const dtype = String(options.dtype || "bf16").toLowerCase();
  const eta = resolveEfficiency(chip, options.efficiency);
  const matrixFlops = chip?.peak_flops?.[dtype];
  const missing = missingFields(chip, dtype);
  const sfuFallback = chip?.sfu_ops == null && chip?.sfu_rate_source === "vector";

  // N4-第4项：跨节点带宽「可调」。芯片规格缺 inter_node.bandwidth 时（三张公开卡皆缺），
  // 允许调用方传入用户可调覆盖值 options.interNodeBandwidth（bytes/s，与芯片同单位）。
  // 覆盖只影响 inter-node 通信费率，不写回芯片数据；仍乘 η.comm 保持"有效带宽"口径。
  const interNodeBandwidth = positive(options.interNodeBandwidth)
    ? options.interNodeBandwidth
    : chip?.interconnect?.inter_node?.bandwidth;

  return {
    matrixPerSecond: positive(matrixFlops) ? (matrixFlops * eta.flops) / 2 : null,
    vectorPerSecond: positive(chip?.vector_flops) ? chip.vector_flops * (eta.vector ?? 1) : null,
    sfuPerSecond: sfuFallback
      ? (positive(chip?.vector_flops) ? chip.vector_flops * (eta.vector ?? 1) : null)
      : (positive(chip?.sfu_ops) ? chip.sfu_ops * (eta.sfu ?? 1) : null),
    bytesPerSecond: positive(chip?.memory_bandwidth) ? chip.memory_bandwidth * eta.hbm : null,
    intraNodeBytesPerSecond: positive(chip?.interconnect?.intra_node?.bandwidth)
      ? chip.interconnect.intra_node.bandwidth * (eta.intra_node_comm ?? 1) : null,
    interNodeBytesPerSecond: positive(interNodeBandwidth)
      ? interNodeBandwidth * (eta.comm ?? 1) : null,
    // 追溯该 inter-node 费率的来源：用户可调覆盖 vs 芯片规格 vs 缺失（诚实标注用）。
    interNodeBandwidthSource: positive(options.interNodeBandwidth)
      ? "user_input"
      : (positive(chip?.interconnect?.inter_node?.bandwidth) ? "spec" : "missing"),
    missing,
    efficiency: eta,
  };
}
