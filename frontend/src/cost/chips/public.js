import { validateChipEntry } from "./coverage.js";

const NVIDIA_A100_SOURCE = "https://www.nvidia.com/en-us/data-center/a100/";
const NVIDIA_H100_SOURCE = "https://www.nvidia.com/en-us/data-center/h100/";
const NVIDIA_L40S_SOURCE = "https://www.nvidia.com/en-us/data-center/l40s/";

// 公开芯片规格表。所有容量和带宽均使用十进制 SI 单位，与厂商规格页保持一致。
// 来源：NVIDIA 官方产品规格页；不包含未经公开资料核实的字段。
export const PUBLIC_CHIPS = [
  {
    id: "nvidia-a100-80gb-sxm",
    vendor: "NVIDIA",
    name: "A100 80GB SXM",
    memory_bytes: 80e9,
    memory_bandwidth: 2039e9,
    peak_flops: {
      fp32: 19.5e12,
      bf16: 312e12,
      fp16: 312e12,
      int8: 624e12,
    },
    interconnect: {
      intra_node: { kind: "NVLink", bandwidth: 600e9 },
    },
    source: NVIDIA_A100_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_A100_SOURCE,
      memory_bandwidth: NVIDIA_A100_SOURCE,
      peak_flops: NVIDIA_A100_SOURCE,
      interconnect: NVIDIA_A100_SOURCE,
    },
    confidence: "official",
    notes: ["峰值算力采用官方表中未启用稀疏性的数值。"],
  },
  {
    id: "nvidia-h100-80gb-sxm",
    vendor: "NVIDIA",
    name: "H100 80GB SXM",
    memory_bytes: 80e9,
    memory_bandwidth: 3.35e12,
    peak_flops: {
      fp32: 67e12,
      bf16: 989.5e12,
      fp16: 989.5e12,
      fp8: 1979e12,
      int8: 1979e12,
    },
    interconnect: {
      intra_node: { kind: "NVLink", bandwidth: 900e9 },
    },
    source: NVIDIA_H100_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_H100_SOURCE,
      memory_bandwidth: NVIDIA_H100_SOURCE,
      peak_flops: NVIDIA_H100_SOURCE,
      interconnect: NVIDIA_H100_SOURCE,
    },
    confidence: "official",
    notes: ["官方峰值算力表标注为启用稀疏性；此处按官方稀疏峰值除以 2，记录稠密峰值。"],
  },
  {
    id: "nvidia-l40s-48gb",
    vendor: "NVIDIA",
    name: "L40S 48GB",
    memory_bytes: 48e9,
    memory_bandwidth: 864e9,
    peak_flops: {
      fp32: 91.6e12,
      bf16: 362.05e12,
      fp16: 362.05e12,
      fp8: 733e12,
      int8: 733e12,
    },
    interconnect: {
      intra_node: { kind: "PCIe Gen4 x16", bandwidth: 64e9 },
    },
    source: NVIDIA_L40S_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_L40S_SOURCE,
      memory_bandwidth: NVIDIA_L40S_SOURCE,
      peak_flops: NVIDIA_L40S_SOURCE,
      interconnect: NVIDIA_L40S_SOURCE,
    },
    confidence: "official",
    notes: ["BF16/FP16/FP8/INT8 采用官方未启用稀疏性的数值；PCIe 为官方双向带宽。"],
  },
];

export function validatePublicChipCatalog(chips = PUBLIC_CHIPS) {
  const errors = [];
  const ids = new Set();
  for (const chip of chips) {
    for (const error of validateChipEntry(chip)) errors.push(`${chip?.id || "<unknown>"}: ${error}`);
    if (ids.has(chip.id)) errors.push(`${chip.id}: id 重复`);
    ids.add(chip.id);
  }
  return errors;
}
