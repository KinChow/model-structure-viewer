// 校验器走 chipValidation.js（零依赖）：public.js（数据）不 import coverage.js（判定），
// 否则与 coverage → public 的取数边构成加载环（M11.5 子项 2）。
import { validateChipEntry } from "./chipValidation.js";

const NVIDIA_A100_SOURCE = "https://www.nvidia.com/en-us/data-center/a100/";
const NVIDIA_H100_SOURCE = "https://www.nvidia.com/en-us/data-center/h100/";
const NVIDIA_H200_SOURCE = "https://www.nvidia.com/en-us/data-center/h200/";
const NVIDIA_L40S_SOURCE = "https://www.nvidia.com/en-us/data-center/l40s/";
const NVIDIA_L40_SOURCE = "https://www.nvidia.com/en-us/data-center/l40/";
const NVIDIA_L4_SOURCE = "https://www.nvidia.com/en-us/data-center/l4/";
const NVIDIA_L40_DATASHEET_SOURCE = "https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf";
// TechPowerUp 用作第二来源交叉核对显存类型、总线、核心数和产品变体；
// 计算峰值、带宽和互联仍以厂商规格页为准，避免把第三方推导值当作官方口径。
const TECHPOWERUP_GPU_DATABASE = "https://www.techpowerup.com/gpu-specs/";
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
    hardware: {
      architecture: "Ampere",
      memory_type: "HBM2e",
      memory_bus_bits: 5120,
      sm_count: 108,
      cuda_cores: 6912,
      tensor_cores: 432,
      tdp_watts: 400,
      form_factor: "SXM",
    },
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
      hardware: NVIDIA_A100_SOURCE,
    },
    cross_check_sources: [TECHPOWERUP_GPU_DATABASE],
    confidence: "official",
    notes: ["峰值算力采用官方表中未启用稀疏性的数值。"],
    notes_en: ["Peak compute uses the non-sparsity figures from the official spec table."],
  },
  {
    id: "nvidia-h100-80gb-sxm",
    vendor: "NVIDIA",
    name: "H100 80GB SXM",
    memory_bytes: 80e9,
    memory_bandwidth: 3.35e12,
    hardware: {
      architecture: "Hopper",
      memory_type: "HBM3",
      memory_bus_bits: 5120,
      sm_count: 132,
      cuda_cores: 16896,
      tensor_cores: 528,
      tdp_watts: 700,
      form_factor: "SXM",
    },
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
      hardware: NVIDIA_H100_SOURCE,
    },
    cross_check_sources: [TECHPOWERUP_GPU_DATABASE],
    confidence: "official",
    notes: ["官方峰值算力表标注为启用稀疏性；此处按官方稀疏峰值除以 2，记录稠密峰值。"],
    notes_en: ["The official peak-compute table is labeled with sparsity enabled; the dense peak here is the official sparse peak divided by 2."],
  },
  {
    id: "nvidia-l40s-48gb",
    vendor: "NVIDIA",
    name: "L40S 48GB",
    memory_bytes: 48e9,
    memory_bandwidth: 864e9,
    hardware: {
      architecture: "Ada Lovelace",
      memory_type: "GDDR6 ECC",
      memory_bus_bits: 384,
      sm_count: 142,
      cuda_cores: 18176,
      tensor_cores: 568,
      tdp_watts: 350,
      form_factor: "PCIe",
    },
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
      hardware: NVIDIA_L40S_SOURCE,
    },
    cross_check_sources: [TECHPOWERUP_GPU_DATABASE],
    confidence: "official",
    notes: ["BF16/FP16/FP8/INT8 采用官方未启用稀疏性的数值；PCIe 为官方双向带宽。"],
    notes_en: ["BF16/FP16/FP8/INT8 use the official non-sparsity figures; PCIe is the official bidirectional bandwidth."],
  },
  {
    id: "nvidia-a100-80gb-pcie",
    vendor: "NVIDIA",
    name: "A100 80GB PCIe",
    memory_bytes: 80e9,
    memory_bandwidth: 1935e9,
    hardware: {
      architecture: "Ampere",
      memory_type: "HBM2e",
      memory_bus_bits: 5120,
      sm_count: 108,
      cuda_cores: 6912,
      tensor_cores: 432,
      tdp_watts: 300,
      form_factor: "PCIe",
    },
    peak_flops: {
      fp32: 19.5e12,
      tf32: 156e12,
      bf16: 312e12,
      fp16: 312e12,
      int8: 624e12,
    },
    vector_flops: 19.5e12,
    sfu_ops: 4.875e12,
    interconnect: {
      intra_node: { kind: "PCIe Gen4 x16", bandwidth: 64e9 },
    },
    source: NVIDIA_A100_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_A100_SOURCE,
      memory_bandwidth: NVIDIA_A100_SOURCE,
      peak_flops: NVIDIA_A100_SOURCE,
      interconnect: NVIDIA_A100_SOURCE,
      vector_flops: NVIDIA_A100_SOURCE,
      sfu_ops: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions",
      hardware: NVIDIA_A100_SOURCE,
    },
    cross_check_sources: [TECHPOWERUP_GPU_DATABASE],
    confidence: "official",
    notes: ["PCIe 80GB 的显存带宽为 1,935GB/s；计算峰值采用官方未启用稀疏性的数值。"],
    notes_en: ["The PCIe 80GB memory bandwidth is 1,935 GB/s; compute peaks use the official non-sparsity figures."],
  },
  {
    id: "nvidia-h200-141gb-sxm",
    vendor: "NVIDIA",
    name: "H200 141GB SXM",
    memory_bytes: 141e9,
    memory_bandwidth: 4.8e12,
    hardware: {
      architecture: "Hopper",
      memory_type: "HBM3e",
      memory_bus_bits: 6144,
      sm_count: 132,
      cuda_cores: 16896,
      tensor_cores: 528,
      tdp_watts: 700,
      form_factor: "SXM",
    },
    peak_flops: {
      fp32: 67e12,
      // H200 官方表带 * 的 Tensor Core 数值启用稀疏性；目录统一记录稠密峰值。
      tf32: 494.5e12,
      bf16: 989.5e12,
      fp16: 989.5e12,
      fp8: 1979e12,
      int8: 1979e12,
    },
    vector_flops: 67e12,
    sfu_ops: 8.375e12,
    interconnect: {
      intra_node: { kind: "NVLink", bandwidth: 900e9 },
    },
    source: NVIDIA_H200_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_H200_SOURCE,
      memory_bandwidth: NVIDIA_H200_SOURCE,
      peak_flops: NVIDIA_H200_SOURCE,
      interconnect: NVIDIA_H200_SOURCE,
      vector_flops: NVIDIA_H200_SOURCE,
      sfu_ops: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions",
      hardware: NVIDIA_H200_SOURCE,
    },
    cross_check_sources: [TECHPOWERUP_GPU_DATABASE],
    confidence: "official",
    notes: ["H200 官方 Tensor Core 表为稀疏峰值；此处按官方脚注除以 2 记录稠密峰值。显存为 141GB HBM3e，带宽为 4.8TB/s。"],
    notes_en: ["The H200 official Tensor Core table shows sparse peaks; dense peaks here are divided by 2 per the official footnote. Memory is 141GB HBM3e at 4.8TB/s."],
  },
  {
    id: "nvidia-l40-48gb",
    vendor: "NVIDIA",
    name: "L40 48GB",
    memory_bytes: 48e9,
    memory_bandwidth: 864e9,
    hardware: {
      architecture: "Ada Lovelace",
      memory_type: "GDDR6 ECC",
      memory_bus_bits: 384,
      sm_count: 142,
      cuda_cores: 18176,
      tensor_cores: 568,
      tdp_watts: 300,
      form_factor: "PCIe",
    },
    peak_flops: {
      fp32: 90.5e12,
      tf32: 90.5e12,
      bf16: 181.05e12,
      fp16: 181.05e12,
      fp8: 362e12,
      int8: 362e12,
    },
    vector_flops: 90.5e12,
    sfu_ops: 11.3125e12,
    interconnect: {
      intra_node: { kind: "PCIe Gen4 x16", bandwidth: 64e9 },
    },
    source: NVIDIA_L40_DATASHEET_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_L40_DATASHEET_SOURCE,
      memory_bandwidth: NVIDIA_L40_DATASHEET_SOURCE,
      peak_flops: NVIDIA_L40_DATASHEET_SOURCE,
      interconnect: NVIDIA_L40_DATASHEET_SOURCE,
      vector_flops: NVIDIA_L40_DATASHEET_SOURCE,
      sfu_ops: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions",
      hardware: NVIDIA_L40_DATASHEET_SOURCE,
    },
    cross_check_sources: [TECHPOWERUP_GPU_DATABASE],
    confidence: "official",
    notes: ["L40 的 FP32 为 90.5 TFLOPS；Tensor Core 采用官方未启用稀疏性的数值，PCIe 为双向 64GB/s。"],
    notes_en: ["L40 FP32 is 90.5 TFLOPS; Tensor Core values use the official non-sparsity figures, and PCIe is 64 GB/s bidirectional."],
  },
  {
    id: "nvidia-l4-24gb",
    vendor: "NVIDIA",
    name: "L4 24GB",
    memory_bytes: 24e9,
    memory_bandwidth: 300e9,
    hardware: {
      architecture: "Ada Lovelace",
      memory_type: "GDDR6",
      memory_bus_bits: 192,
      sm_count: 58,
      cuda_cores: 7424,
      tensor_cores: 58,
      tdp_watts: 72,
      form_factor: "PCIe",
    },
    peak_flops: {
      fp32: 30.3e12,
      tf32: 60e12,
      bf16: 121e12,
      fp16: 121e12,
      fp8: 242.5e12,
      int8: 242.5e12,
    },
    vector_flops: 30.3e12,
    sfu_ops: 3.7875e12,
    interconnect: {
      intra_node: { kind: "PCIe Gen4 x16", bandwidth: 64e9 },
    },
    source: NVIDIA_L4_SOURCE,
    field_sources: {
      memory_bytes: NVIDIA_L4_SOURCE,
      memory_bandwidth: NVIDIA_L4_SOURCE,
      peak_flops: NVIDIA_L4_SOURCE,
      interconnect: NVIDIA_L4_SOURCE,
      vector_flops: NVIDIA_L4_SOURCE,
      sfu_ops: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions",
      hardware: TECHPOWERUP_GPU_DATABASE,
    },
    cross_check_sources: [TECHPOWERUP_GPU_DATABASE],
    confidence: "official",
    notes: ["L4 官方 Tensor Core 数值带 * 表示稀疏峰值；此处按官方脚注除以 2 记录稠密峰值。"],
    notes_en: ["L4 official Tensor Core values marked with * are sparse peaks; dense peaks here are divided by 2 per the official footnote."],
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
    notes_en: [
      "Official figures have a revision history, so conservative values are used: 20 Cube + 40 Vector (Vector:Cube = 2:1) and 800 GB/s memory bandwidth come from the measured-platform description in arXiv 2505.15112; FP16/INT8/capacity follow third-party consistent figures without taking upper bounds.",
      "BF16 and FP16 share the Cube unit at the same throughput (the DaVinci Cube runs both 16-bit floats at the same rate); FP32 runs on the vector unit, taking the 9.2 TFLOPS measured for the 910B series in arXiv 2607.20120.",
      "No dedicated SFU: transcendental functions run on the vector unit, and the sfu rate is semantically mapped to the vector-unit rate via sfu_rate_source:\"vector\" (a semantic mapping, not an estimate). HCCS 392 GB/s is the aggregated theoretical bandwidth of 7 links per processor (Huawei official documentation).",
    ],
  },
];

export function validatePublicChipCatalog(chips = PUBLIC_CHIPS) {
  const errors = [];
  const ids = new Set();
  for (const chip of chips) {
    for (const error of validateChipEntry(chip)) errors.push(error);
    if (ids.has(chip.id)) errors.push({ code: "chip.duplicateId", params: { id: chip.id } });
    ids.add(chip.id);
  }
  return errors;
}
