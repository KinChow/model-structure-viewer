// 芯片规格字段覆盖判定。
// 来源：evolution_design.md §5.4(c) 的“按字段降级”规则；不使用估算值补齐缺失字段。

const CONFIDENCE_VALUES = new Set(["official", "vendor-marketing", "community", "local"]);
const REQUIRED_INTERCONNECT = ["intra_node.bandwidth"];

function hasPositiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function getPath(object, path) {
  return path.split(".").reduce((value, key) => value?.[key], object);
}

function missingFields(chip, dtype = "bf16") {
  const missing = [];
  if (!hasPositiveNumber(chip?.memory_bytes)) missing.push("memory_bytes");
  if (!hasPositiveNumber(chip?.memory_bandwidth)) missing.push("memory_bandwidth");
  if (!hasPositiveNumber(chip?.peak_flops?.[dtype])) missing.push(`peak_flops.${dtype}`);
  if (!hasPositiveNumber(chip?.vector_flops)) missing.push("vector_flops");
  if (!hasPositiveNumber(chip?.sfu_ops)) missing.push("sfu_ops");
  for (const path of REQUIRED_INTERCONNECT) {
    if (!hasPositiveNumber(getPath(chip?.interconnect, path))) missing.push(`interconnect.${path}`);
  }
  return missing;
}

/**
 * 返回当前芯片和 dtype 可用的能力，不会把缺失字段静默当成估算值。
 * @param {object} chip 芯片规格条目
 * @param {string} dtype 当前计算精度，例如 bf16/fp16/fp8/int8
 */
export function getChipCoverage(chip, dtype = "bf16") {
  const missing = missingFields(chip, dtype);
  const hasMemory = !missing.includes("memory_bytes");
  const hasBandwidth = !missing.includes("memory_bandwidth");
  const hasFlops = !missing.includes(`peak_flops.${dtype}`);
  const hasIntraLink = !missing.includes("interconnect.intra_node.bandwidth");
  const hasInterLink = hasPositiveNumber(chip?.interconnect?.inter_node?.bandwidth);
  const hasVector = !missing.includes("vector_flops");
  const hasSfu = !missing.includes("sfu_ops");
  const warnings = [];

  if (!hasInterLink && hasIntraLink) {
    warnings.push("缺少 interconnect.inter_node.bandwidth，跨节点按节点内带宽估算，结果偏乐观");
  }
  if (!hasVector) warnings.push("缺少 vector_flops，向量单元瓶颈不可判；请补充带来源的 FP32 吞吐");
  if (!hasSfu) warnings.push("缺少 sfu_ops，SFU 瓶颈不可判；请补充带来源的特殊函数吞吐（NVIDIA 可按 CUDA guide 每 SM 每时钟比值推导）");
  if (chip?.source == null || chip.source === "") warnings.push("缺少规格来源 source");
  if (chip?.confidence != null && !CONFIDENCE_VALUES.has(chip.confidence)) {
    warnings.push(`未知 confidence：${chip.confidence}`);
  }

  return {
    chipId: chip?.id || null,
    dtype,
    missing,
    warnings,
    source: chip?.source || null,
    confidence: chip?.confidence || null,
    capabilities: {
      fit: hasMemory,
      max_context: hasMemory,
      weight_share: true,
      compute_bound: hasBandwidth && hasFlops,
      memory_bound: hasBandwidth && hasFlops,
      comm_bound: hasBandwidth && hasFlops && hasIntraLink,
      parallel_compare: hasBandwidth && hasFlops && hasIntraLink,
      vector_bound: hasVector,
      sfu_bound: hasSfu,
    },
  };
}

/** 校验公开或本地芯片条目的基本形状；缺失规格返回错误，不自动填值。 */
export function validateChipEntry(chip) {
  const errors = [];
  if (!chip || typeof chip !== "object") return ["芯片条目必须是对象"];
  if (!chip.id) errors.push("缺少 id");
  if (!chip.vendor) errors.push("缺少 vendor");
  if (!chip.name) errors.push("缺少 name");
  if (!chip.source) errors.push("缺少 source");
  if (chip.confidence && !CONFIDENCE_VALUES.has(chip.confidence)) errors.push(`未知 confidence：${chip.confidence}`);
  return errors;
}

export { CONFIDENCE_VALUES };
