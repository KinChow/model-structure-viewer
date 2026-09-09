// 把用户手动输入的常用单位转换为内部 SI 单位；不推测任何缺失规格。
// 手工条目与公开卡同结构：所有实际写入的规格字段必须带 field_sources（§7 逐字段标来源）；
// 表单未提供来源 URL 时统一记 "user-input"。

const USER_INPUT_SOURCE = "user-input";

function normalizeFieldSource(sourceUrl) {
  const trimmed = String(sourceUrl ?? "").trim();
  return trimmed || USER_INPUT_SOURCE;
}

export function createManualChip({ id, name, memoryGb, memoryBandwidthTb, fp32Tflops, tf32Tflops, fp16Tflops, bf16Tflops, fp8Tflops, int8Tops, interconnectGb, intraNodeGb, interNodeGb, sourceUrl } = {}) {
  const safeId = String(id || name || "manual-chip").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const peakFlops = Object.fromEntries([
    ["fp32", fp32Tflops], ["tf32", tf32Tflops], ["fp16", fp16Tflops], ["bf16", bf16Tflops], ["fp8", fp8Tflops], ["int8", int8Tops],
  ].filter(([, value]) => Number(value) > 0).map(([dtype, value]) => [dtype, Number(value) * 1e12]));
  const intraNodeBandwidth = Number(intraNodeGb ?? interconnectGb);
  const interNodeBandwidth = Number(interNodeGb);
  const fieldSource = normalizeFieldSource(sourceUrl);
  // 与 public.js 同结构的逐字段来源表：只登记实际写入（数值 > 0）的字段。
  const fieldSources = {};
  if (Number(memoryGb) > 0) fieldSources.memory_bytes = fieldSource;
  if (Number(memoryBandwidthTb) > 0) fieldSources.memory_bandwidth = fieldSource;
  if (Object.keys(peakFlops).length > 0) fieldSources.peak_flops = fieldSource;
  if (intraNodeBandwidth > 0 || interNodeBandwidth > 0) fieldSources.interconnect = fieldSource;
  return {
    id: safeId || "manual-chip",
    vendor: "local",
    name: String(name || "Manual chip").trim(),
    memory_bytes: Number(memoryGb) * 1e9,
    memory_bandwidth: Number(memoryBandwidthTb) * 1e12,
    peak_flops: peakFlops,
    interconnect: {
      intra_node: { kind: "custom", bandwidth: intraNodeBandwidth * 1e9 },
      inter_node: interNodeBandwidth > 0 ? { kind: "custom", bandwidth: interNodeBandwidth * 1e9 } : undefined,
    },
    source: "manual-session",
    confidence: "local",
    field_sources: fieldSources,
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
