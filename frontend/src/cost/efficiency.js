// 理论分析使用的效率因子默认值与芯片级覆盖。
// 来源：llm-analysis 的 flops_efficiency、hbm_memory_efficiency、interconnect efficiency 设计。

export const DEFAULT_EFFICIENCY = Object.freeze({
  flops: 0.7,
  hbm: 0.9,
  comm: 0.6,
  intra_node_comm: 0.8,
});

function validEfficiency(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value <= 1;
}

/** 按芯片覆盖效率因子；非法覆盖值被忽略，不静默改变默认假设。 */
export function resolveEfficiency(chip = {}, overrides = {}) {
  const chipEfficiency = chip.efficiency || {};
  const merged = {
    ...DEFAULT_EFFICIENCY,
    ...chipEfficiency,
    ...overrides,
  };
  return {
    flops: validEfficiency(merged.flops) ? merged.flops : DEFAULT_EFFICIENCY.flops,
    hbm: validEfficiency(merged.hbm) ? merged.hbm : DEFAULT_EFFICIENCY.hbm,
    comm: validEfficiency(merged.comm) ? merged.comm : DEFAULT_EFFICIENCY.comm,
    intra_node_comm: validEfficiency(merged.intra_node_comm)
      ? merged.intra_node_comm
      : DEFAULT_EFFICIENCY.intra_node_comm,
  };
}
