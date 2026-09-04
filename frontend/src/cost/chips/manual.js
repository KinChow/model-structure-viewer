// 把用户手动输入的常用单位转换为内部 SI 单位；不推测任何缺失规格。

export function createManualChip({ id, name, memoryGb, memoryBandwidthTb, bf16Tflops, interconnectGb } = {}) {
  const safeId = String(id || name || "manual-chip").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return {
    id: safeId || "manual-chip",
    vendor: "local",
    name: String(name || "Manual chip").trim(),
    memory_bytes: Number(memoryGb) * 1e9,
    memory_bandwidth: Number(memoryBandwidthTb) * 1e12,
    peak_flops: { bf16: Number(bf16Tflops) * 1e12 },
    interconnect: { intra_node: { kind: "custom", bandwidth: Number(interconnectGb) * 1e9 } },
    source: "manual-session",
    confidence: "local",
  };
}

export function validateManualChipInput(input = {}) {
  const errors = [];
  if (!String(input.name || "").trim()) errors.push("名称不能为空");
  for (const [key, label] of [["memoryGb", "显存"], ["memoryBandwidthTb", "HBM 带宽"], ["bf16Tflops", "BF16 算力"], ["interconnectGb", "互联带宽"]]) {
    if (!(Number(input[key]) > 0)) errors.push(`${label}必须大于 0`);
  }
  return errors;
}
