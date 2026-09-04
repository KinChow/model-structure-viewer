// 把用户手动输入的常用单位转换为内部 SI 单位；不推测任何缺失规格。

export function createManualChip({ id, name, memoryGb, memoryBandwidthTb, fp32Tflops, fp16Tflops, bf16Tflops, fp8Tflops, int8Tops, interconnectGb, intraNodeGb, interNodeGb } = {}) {
  const safeId = String(id || name || "manual-chip").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return {
    id: safeId || "manual-chip",
    vendor: "local",
    name: String(name || "Manual chip").trim(),
    memory_bytes: Number(memoryGb) * 1e9,
    memory_bandwidth: Number(memoryBandwidthTb) * 1e12,
    peak_flops: Object.fromEntries([
      ["fp32", fp32Tflops], ["fp16", fp16Tflops], ["bf16", bf16Tflops], ["fp8", fp8Tflops], ["int8", int8Tops],
    ].filter(([, value]) => Number(value) > 0).map(([dtype, value]) => [dtype, Number(value) * 1e12])),
    interconnect: {
      intra_node: { kind: "custom", bandwidth: Number(intraNodeGb ?? interconnectGb) * 1e9 },
      inter_node: Number(interNodeGb) > 0 ? { kind: "custom", bandwidth: Number(interNodeGb) * 1e9 } : undefined,
    },
    source: "manual-session",
    confidence: "local",
  };
}

export function validateManualChipInput(input = {}) {
  const errors = [];
  if (!String(input.name || "").trim()) errors.push("名称不能为空");
  for (const [key, label] of [["memoryGb", "显存"], ["memoryBandwidthTb", "HBM 带宽"], ["bf16Tflops", "BF16 算力"]]) {
    if (!(Number(input[key]) > 0)) errors.push(`${label}必须大于 0`);
  }
  if (!(Number(input.intraNodeGb ?? input.interconnectGb) > 0)) errors.push("节点内带宽必须大于 0");
  return errors;
}
