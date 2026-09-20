// 理论分析使用的效率因子默认值与芯片级覆盖。
// 默认值是 UI 可调假设（原则 §3.6），不是实测。不再参考 llm-analysis。

export const DEFAULT_EFFICIENCY = Object.freeze({
  flops: 0.7,
  // η_hbm 由 NV-3 在机尺寸扫描校准：0.9→0.7（A100 memory-bound 逐元素实测可达带宽
  // 0.44–0.57，0.7 取跨厂商可辩护的可达 HBM 比例、不 overfit；且有效带宽地板 1427GB/s
  // > 实测最好点 1170GB/s 仍保持"下界"性质。证据 details/nv_evidence/nv3/operator_cost/
  // roofline_size_sweep.md。仍是 UI/芯片可覆盖假设，非某次实测利用率。
  hbm: 0.7,
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
