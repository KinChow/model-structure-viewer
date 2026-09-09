// 校验器走 chipValidation.js（零依赖）：public.js（数据）不 import coverage.js（判定），
// 否则与 coverage → public 的取数边构成加载环（M11.5 子项 2）。
import { validateChipEntry } from "./chipValidation.js";

const NVIDIA_A100_SOURCE = "https://www.nvidia.com/en-us/data-center/a100/";
const NVIDIA_H100_SOURCE = "https://www.nvidia.com/en-us/data-center/h100/";
const NVIDIA_L40S_SOURCE = "https://www.nvidia.com/en-us/data-center/l40s/";
// 昇腾 910B4 来源：单元数/显存带宽出自实测论文，HCCS 互联出自华为官方文档，算力/容量取第三方一致口径。
const ASCEND_910B4_SOURCE = "https://arxiv.org/abs/2505.15112";
const ASCEND_910B4_SPECS_SOURCE = "https://blog.ailemon.net/2025/05/24/huawei-ascend-npu-params-for-ai";
const ASCEND_910B4_MEMORY_SOURCE = "https://cset.georgetown.edu/publication/pushing-the-limits-huaweis-ai-chip-tests-u-s-export-controls/";
const ASCEND_HCCS_SOURCE = "https://support.huawei.com/enterprise/zh/doc/EDOC1100317202/f3dba488";
const ASCEND_VECTOR_SOURCE = "https://arxiv.org/abs/2607.20120";

// 公开芯片规格表。所有容量和带宽均使用十进制 SI 单位，与厂商规格页保持一致。
// 来源：厂商官方产品规格页或公开实测文献；不包含未经公开资料核实的字段。
export const PUBLIC_CHIPS = [
  {
    id: "nvidia-a100-80gb-sxm",
    vendor: "NVIDIA",
    name: "A100 80GB SXM",
    memory_bytes: 80e9,
    memory_bandwidth: 2039e9,
    peak_flops: {
      fp32: 19.5e12,
      // TF32 dense（官方页 156/312 两列取未启用稀疏性列，与 bf16 的取列口径一致）
      tf32: 156e12,
      bf16: 312e12,
      fp16: 312e12,
      int8: 624e12,
    },
    vector_flops: 19500000000000,
    // SFU 吞吐 = vector_flops / 4（sm_80：FP32 64/SM/clk，SFU 16/SM/clk，CUDA C++ Programming Guide）
    sfu_ops: 4875000000000,

    interconnect: {
      intra_node: { kind: "NVLink", bandwidth: 600e9 },
    },
    source: NVIDIA_A100_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_A100_SOURCE,
      memory_bandwidth: NVIDIA_A100_SOURCE,
      peak_flops: NVIDIA_A100_SOURCE,
      interconnect: NVIDIA_A100_SOURCE,
          vector_flops: NVIDIA_A100_SOURCE,
      sfu_ops: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions",
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
      // TF32 dense（H100 SXM 官方页 494.5/989 两列取 dense 列）
      tf32: 494.5e12,
      bf16: 989.5e12,
      fp16: 989.5e12,
      fp8: 1979e12,
      int8: 1979e12,
    },
    vector_flops: 67000000000000,
    // SFU 吞吐 = vector_flops / 8（sm_90：FP32 128/SM/clk，SFU 16/SM/clk，CUDA C++ Programming Guide）
    sfu_ops: 8375000000000,

    interconnect: {
      intra_node: { kind: "NVLink", bandwidth: 900e9 },
    },
    source: NVIDIA_H100_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_H100_SOURCE,
      memory_bandwidth: NVIDIA_H100_SOURCE,
      peak_flops: NVIDIA_H100_SOURCE,
      interconnect: NVIDIA_H100_SOURCE,
          vector_flops: NVIDIA_H100_SOURCE,
      sfu_ops: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions",
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
      tf32: 183e12,
      bf16: 362.05e12,
      fp16: 362.05e12,
      fp8: 733e12,
      int8: 733e12,
    },
    vector_flops: 91600000000000,
    // SFU 吞吐 = vector_flops / 8（sm_89：FP32 128/SM/clk，SFU 16/SM/clk，CUDA C++ Programming Guide）
    sfu_ops: 11450000000000,

    interconnect: {
      intra_node: { kind: "PCIe Gen4 x16", bandwidth: 64e9 },
    },
    source: NVIDIA_L40S_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_L40S_SOURCE,
      memory_bandwidth: NVIDIA_L40S_SOURCE,
      peak_flops: NVIDIA_L40S_SOURCE,
      interconnect: NVIDIA_L40S_SOURCE,
          vector_flops: NVIDIA_L40S_SOURCE,
      sfu_ops: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions",
    },
    confidence: "official",
    notes: ["BF16/FP16/FP8/INT8 采用官方未启用稀疏性的数值；PCIe 为官方双向带宽。"],
  },
  {
    // 官方口径有调整史，取保守值：华为未公开 910B4 完整数据表，算力/容量取第三方一致口径
    // （FP16 280 TFLOPS、INT8 560 TOPS、32GB HBM2e），confidence 记 community 而非 official。
    // 达芬奇架构 Cube+Vector 双单元、无独立 SFU——超越函数在向量单元执行，故不写 sfu_ops，
    // 改用 sfu_rate_source: "vector" 语义映射（rates.js 已支持；语义映射非估算，§3.7）。
    id: "huawei-ascend-910b4",
    vendor: "Huawei",
    name: "Ascend 910B4",
    memory_bytes: 32e9,
    memory_bandwidth: 800e9,
    peak_flops: {
      fp32: 9.2e12,
      bf16: 280e12,
      fp16: 280e12,
      int8: 560e12,
    },
    // 向量单元 FP32 吞吐：arXiv 2607.20120 对 910B 系实测值；该文未区分 B 子型号，按保守原则未上调。
    vector_flops: 9.2e12,
    sfu_rate_source: "vector",

    interconnect: {
      intra_node: { kind: "HCCS", bandwidth: 392e9 },
    },
    source: ASCEND_910B4_SOURCE,
    field_sources: {
      memory_bytes: ASCEND_910B4_MEMORY_SOURCE,
      memory_bandwidth: ASCEND_910B4_SOURCE,
      peak_flops: ASCEND_910B4_SPECS_SOURCE,
      interconnect: ASCEND_HCCS_SOURCE,
      vector_flops: ASCEND_VECTOR_SOURCE,
    },
    confidence: "community",
    notes: [
      "官方口径有调整史，取保守值：20 Cube + 40 Vector（向量与立方单元 2:1）与 800GB/s 显存带宽出自 arXiv 2505.15112 实测平台描述；FP16/INT8/容量取第三方一致口径且未取上限值。",
      "BF16 与 FP16 共用 Cube、吞吐相同（达芬奇 Cube 对两种 16-bit 浮点同速率）；FP32 在向量单元执行，取 arXiv 2607.20120 对 910B 系实测 9.2 TFLOPS。",
      "无独立 SFU：超越函数在向量单元执行，sfu 速率按 sfu_rate_source:\"vector\" 语义映射到向量单元费率（语义映射非估算）。HCCS 392GB/s 为每处理器 7 条链路聚合理论带宽（华为官方文档）。",
    ],
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
