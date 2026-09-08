// 芯片规格字段覆盖判定。
// 来源：evolution_design.md §5.4(c) 的“按字段降级”规则；不使用估算值补齐缺失字段。
import { PUBLIC_CHIPS } from "./public.js";

// 单位量级健全性对照（旁路 C）：手工条目的带宽/算力若与所有公开卡同字段偏差超过 10 倍
// （如 GB 写成 Gb、TFLOPS 写成 GFLOPS），提示"单位可能错误"——只警告不拒绝（§3.6：倍数级错误会改变结论）。
// PUBLIC_CHIPS 只在函数内运行时读取，模块加载期不解引用，与 public.js 的相互引用两种加载顺序均安全。
const UNIT_SANITY_FACTOR = 10;
const MANUAL_CHIP_SOURCE = "manual-session";

const CONFIDENCE_VALUES = new Set(["official", "vendor-marketing", "community", "local"]);
const REQUIRED_INTERCONNECT = ["intra_node.bandwidth"];

function hasPositiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function getPath(object, path) {
  return path.split(".").reduce((value, key) => value?.[key], object);
}

/** 值与参照集里每一条都相差超过 10 倍才算离群（只要还在任一公开卡的 10 倍内就视为同量级）。 */
function deviatesOverTenfold(value, references) {
  return references.length > 0
    && references.every((reference) => value > reference * UNIT_SANITY_FACTOR || value < reference / UNIT_SANITY_FACTOR);
}

/** 手工条目的带宽/算力度量级对照；非手工条目不做对照。 */
function unitAnomalyWarnings(chip) {
  const warnings = [];
  if (chip?.source !== MANUAL_CHIP_SOURCE) return warnings;
  if (hasPositiveNumber(chip.memory_bandwidth)
    && deviatesOverTenfold(chip.memory_bandwidth, PUBLIC_CHIPS.map((entry) => entry.memory_bandwidth).filter(hasPositiveNumber))) {
    warnings.push("单位可能错误：memory_bandwidth 与所有公开芯片偏差超过 10 倍，请确认单位（如 GB/s 写成 TB/s、Gb 写成 GB） / Unit may be wrong: memory_bandwidth deviates over 10x from every public chip, check the unit (GB/s vs TB/s, Gb vs GB)");
  }
  for (const [dtype, value] of Object.entries(chip.peak_flops || {})) {
    const references = PUBLIC_CHIPS.map((entry) => entry.peak_flops?.[dtype]).filter(hasPositiveNumber);
    if (hasPositiveNumber(value) && deviatesOverTenfold(value, references)) {
      warnings.push(`单位可能错误：peak_flops.${dtype} 与所有公开芯片偏差超过 10 倍，请确认单位（如 TFLOPS 写成 GFLOPS） / Unit may be wrong: peak_flops.${dtype} deviates over 10x from every public chip, check the unit (TFLOPS vs GFLOPS)`);
    }
  }
  return warnings;
}

export function missingFields(chip, dtype = "bf16") {
  const missing = [];
  if (!hasPositiveNumber(chip?.memory_bytes)) missing.push("memory_bytes");
  if (!hasPositiveNumber(chip?.memory_bandwidth)) missing.push("memory_bandwidth");
  if (!hasPositiveNumber(chip?.peak_flops?.[dtype])) missing.push(`peak_flops.${dtype}`);
  if (!hasPositiveNumber(chip?.vector_flops)) missing.push("vector_flops");
  // sfu_ops 缺失但声明了 sfu→vector 语义映射（如昇腾）且向量吞吐在场时，sfu 速率经映射可判，不算缺项。
  const sfuMapped = chip?.sfu_rate_source === "vector" && hasPositiveNumber(chip?.vector_flops);
  if (!hasPositiveNumber(chip?.sfu_ops) && !sfuMapped) missing.push("sfu_ops");
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
  warnings.push(...unitAnomalyWarnings(chip));

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
