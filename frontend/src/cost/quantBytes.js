// 量化容量的 per-matrix 精确计算。
//
// 为什么存在：无 checkpoint 证据时，量化容量此前是标量
// `quantizationBytesPerParameter`（所有参数一个字节宽），scale 张量整片缺失。
// 量化容量其实是**逐矩阵**可精确计算的 —— 每个线性层的 [out, in] 在结构树上
// 就有（derivedLinearShape），块的 scale 形状 = ceil(out/b0)·ceil(in/b1)。
//
// 覆盖四种量化方案（2026-09-09 全量落地）：
// - fp8（weight_block_size [b0,b1]）：权重 1B/元素 + scale（scale_fmt "ue8m0"
//   为 1B/块，否则 fp32 4B/块）
// - mxfp8（weight_block_size [1,32]）：权重 1B/元素 + e8m0 scale 1B/块
//   （MXFP8 的 scale 是 32 元素一组的 e8m0）
// - gptq（bits 4, group_size gs）：int4 打包 bits/8 B/元素 + scales fp16
//   2B/(out·in/gs) + qzeros int4 (bits/8)B/(out·in/gs)
// - compressed-tensors（Kimi K2 系/K3）：w4a16 int4（0.5B/元素 + fp16 scale
//   每组，symmetric）与 mxfp4（0.5B/元素 + e8m0 1B/32 组）；"ignore" 数组 =
//   modules_to_not_convert 同义，条目支持 "re:" 前缀正则
//
// 哪些矩阵被量化由 config 的 dynamic / modules_to_not_convert（或
// compressed-tensors 的 ignore）声明（"-:" / "re:" 前缀 = 排除路径正则，
// 如 GPTQ 的 "-:.*attn.*"），isQuantizedPath 判定。

import { canonicalModulePath } from "../structure/truth/graphTruth.js";

const I4 = 0.5;
const FP8 = 1;
const FP16 = 2;
const FP32 = 4;

function ceilDiv(a, b) {
  return Math.ceil(a / Math.max(b, 1));
}

/** 解析 config 上的 quantization_config（顶层或 text_config 嵌套）。
 *  第四个查找位（raw.text_config）是 W-C 核实补的：VLM 家族（Kimi K2.5 系）
 *  把 quantization_config 嵌在 text_config 里，而消费方传的是 normalized——
 *  text_config 挂在其 raw 下，此前第三位只查顶层 text_config 会整族漏检。 */
export function quantizationConfigOf(config) {
  return config?.quantization_config
    ?? config?.raw?.quantization_config
    ?? config?.text_config?.quantization_config
    ?? config?.raw?.text_config?.quantization_config
    ?? null;
}

/** 一个 [out, in] 权重矩阵在给定量化方案下的精确字节（权重 + scale + zeros）。 */
export function quantLinearWeightBytes({ out, inn, quant }) {
  if (!quant || !Number.isFinite(out) || !Number.isFinite(inn) || out <= 0 || inn <= 0) return null;
  const method = quant.quant_method;
  if (method === "fp8" || method === "mxfp8") {
    const [b0, b1] = quant.weight_block_size || [];
    if (!b0 || !b1) return null;
    const scaleBytes = method === "mxfp8" || quant.scale_fmt === "ue8m0" ? 1 : FP32;
    return out * inn * FP8 + ceilDiv(out, b0) * ceilDiv(inn, b1) * scaleBytes;
  }
  if (method === "gptq") {
    const bits = quant.bits || 4;
    const groupSize = quant.group_size || 128;
    const groups = ceilDiv(inn, groupSize);
    return out * inn * (bits / 8) + out * groups * (FP16 + (bits / 8));
  }
  if (method === "compressed-tensors") {
    // vLLM compressed-tensors（Kimi K2-Thinking/K2.5/K2.6/K2.7-Code/K3 实证）：
    // 单 config_groups 组、targets ["Linear"]。按 weights.type 分两案：
    // - int + group 策略（w4a16 pack-quantized，symmetric）：0.5B/元素 + fp16
    //   scale 每组（对称无零点；非对称零点未取证 → null 诚实缺项）；
    // - float 4-bit（mxfp4，K3）：0.5B/元素 + e8m0 scale 1B/32 组
    //   （与 mxfp8 的 scale 机制同构，块形状 [1, group_size]）。
    const weights = Object.values(quant.config_groups || {})[0]?.weights;
    if (!weights) return null;
    const groupSize = weights.group_size || 128;
    if (weights.type === "int" && weights.strategy === "group") {
      if (!weights.symmetric) return null;
      const bits = weights.num_bits || 4;
      return out * inn * (bits / 8) + out * ceilDiv(inn, groupSize) * FP16;
    }
    if (weights.type === "float" && weights.num_bits === 4) {
      return out * inn * 0.5 + out * ceilDiv(inn, groupSize) * 1;
    }
    return null;
  }
  return null;
}

/**
 * 路径是否被量化。dynamic / modules_to_not_convert 的语义照抄 vLLM：
 * dynamic 的 key 不带 "-:" 前缀 = 显式量化的模块，"-:" 前缀 = 排除的正则，
 * 后出现的声明覆盖先出现的。
 *
 * 字面路径 pattern 剥 HF 根包装 model. / language_model. 后再匹配树 id。
 * 图 id 已是 HF `_modules` 名（visual / vision_tower），不再做中段别名桥接。
 */
const LITERAL_PATH = /^[A-Za-z0-9_.\-]+$/;

function pathCandidates(path) {
  return [path];
}

/**
 * 把 safetensors header 的 packing numel 解包成逻辑元素（out×in 口径）。
 *
 * header.parameterTotal / parameterCount 是存储单元计数（GPTQ qweight 一个
 * I32 = 8 个 int4；NVFP4 一个 I8 = 2 个 fp4；scale / qzeros 也占 numel）。
 * 图声明 `weightMatrices` 是逻辑 out×in，两边直接相除会差 0.7–1.8 倍。
 *
 * 有逐张量 header 时优先按 `logicalParameterSummaryFromTensors` 解码；
 * 只有轻量 sidecar 没有张量表时，才回退到 dtype 桶 + quant_method 的
 * bits/group_size 近似。fallback 中 scale 桶（F8_E8M0 / U8）不计逻辑元素；
 * GPTQ 的 F16 scale 不计，并从 I32×8 里扣掉约等于 qzeros 的那份；
 * compressed-tensors 的 scale 进了 BF16，按 packed/group_size 扣。
 *
 * 未知方案或缺少 parameterCount → null（调用方不要拿 packing 打恒等）。
 */
export function logicalElementsFromHeader(header, quant) {
  const fromTensors = logicalParameterSummaryFromTensors(header?.tensors, quant);
  if (fromTensors) return fromTensors.total;
  const counts = header?.parameterCount;
  if (!counts || typeof counts !== "object") return null;
  const method = quant?.quant_method;
  const bits = packedWeightBits(quant);
  const i32Factor = 32 / bits;
  let packedLogical = 0;
  let passthrough = 0;
  for (const [dtype, n] of Object.entries(counts)) {
    if (!Number.isFinite(n) || n <= 0) continue;
    const key = String(dtype).toUpperCase();
    if (key === "F8_E8M0" || key === "U8") continue;
    if (key === "F16" && method === "gptq") continue;
    if (key === "I32") {
      packedLogical += n * i32Factor;
      continue;
    }
    if (key === "I8") {
      packedLogical += n * 2;
      continue;
    }
    passthrough += n;
  }
  let logical = packedLogical + passthrough;
  if (method === "gptq") {
    // I32×(32/bits) 把 qzeros 也解成了逻辑元素；qzeros 与 scales 同形，
    // F16 scales 的 numel 就是多出来的那份。
    logical -= counts.F16 || 0;
  } else if (method === "compressed-tensors") {
    const groupSize = compressedGroupSize(quant);
    if (groupSize > 0 && packedLogical > 0) logical -= packedLogical / groupSize;
  }
  return logical > 0 ? logical : null;
}

/**
 * Decode logical parameters from tensor roles when the safetensors header is
 * available. Dtype buckets alone cannot distinguish GPTQ `qweight` from
 * auxiliary `qzeros`/`g_idx`, compressed-tensors `weight_packed` from
 * `weight_shape`, or FP8-family weights from their scale tensors. Those
 * metadata tensors must not inflate model parameters.
 */
export function logicalParameterSummaryFromTensors(tensors, quant) {
  if (!Array.isArray(tensors) || tensors.length === 0 || !quant) return null;
  const method = quant.quant_method;
  if (!["gptq", "compressed-tensors", "fp8", "mxfp8"].includes(method)) return null;
  const bits = packedWeightBits(quant);
  const factor = 32 / bits;
  let total = 0;
  const logicalParameterCount = {};
  const add = (dtype, value) => {
    if (!(value > 0)) return;
    total += value;
    const key = String(dtype || "UNKNOWN").toUpperCase();
    logicalParameterCount[key] = (logicalParameterCount[key] || 0) + value;
  };
  for (const tensor of tensors) {
    const name = String(tensor?.name || "");
    const suffix = name.split(".").at(-1);
    const shape = Array.isArray(tensor?.shape) ? tensor.shape : null;
    if (!shape || shape.some((dim) => !Number.isFinite(dim) || dim < 0)) continue;
    const elements = shape.reduce((product, dim) => product * dim, 1);
    if (!(elements > 0)) continue;
    if (method === "gptq") {
      if (suffix === "qweight") add("GPTQ", elements * factor);
      else if (suffix === "qzeros" || suffix === "g_idx" || suffix === "scales") continue;
      else add(tensor.dtype, elements);
    } else if (method === "compressed-tensors") {
      if (suffix === "weight_packed") {
        // compressed-tensors int4 uses I32 containers (8 values/unit), while
        // native MXFP4 uses U8 containers (2 values/unit). The config's
        // num_bits alone cannot distinguish these storage units.
        const dtype = String(tensor.dtype || "").toUpperCase();
        const packedFactor = dtype === "U8" ? 2 : factor;
        const logicalDtype = dtype === "U8" ? "MXFP4" : "PACKED";
        add(logicalDtype, elements * packedFactor);
      }
      else if (suffix === "weight_scale" || suffix === "weight_shape") continue;
      else add(tensor.dtype, elements);
    } else {
      const dtype = String(tensor.dtype || "").toUpperCase();
      // FP8-family checkpoints store block scales either as E8M0/U8 tensors
      // or as `weight_scale_inv` tensors in BF16/F32. DeepSeek V4 FP8-family
      // checkpoints additionally use I8 `weight` tensors for packed FP4 experts.
      if (dtype === "F8_E8M0" || (method === "mxfp8" && dtype === "U8")) continue;
      if (/(?:^|\.)weight_scale_inv$/.test(name)) continue;
      if (dtype === "I8" && suffix === "weight") add("PACKED_FP4", elements * 2);
      else add(tensor.dtype, elements);
    }
  }
  return total > 0 ? { total, logicalParameterCount } : null;
}

/**
 * 返回 UI/what-if 使用的逻辑参数量。
 *
 * safetensors 的 parameterTotal 是 storage numel；GPTQ I32、NVFP4 I8、
 * MXFP4 U8 以及对应 scale 桶不能直接作为模型参数量展示。显式 sidecar
 * logicalParameterTotal 优先；普通量化 header 按 quant config 解包，失败时
 * 保守回退 storage total，并由诊断层继续展示 packed storage。
 */
export function logicalParameterTotalFromHeader(header, quant) {
  if (Number.isFinite(header?.logicalParameterTotal) && header.logicalParameterTotal > 0) {
    return header.logicalParameterTotal;
  }
  const fromTensors = logicalParameterSummaryFromTensors(header?.tensors, quant);
  if (fromTensors) return fromTensors.total;
  if (!Number.isFinite(header?.parameterTotal) || header.parameterTotal <= 0) return null;
  if (!quant) return header.parameterTotal;
  return logicalElementsFromHeader(header, quant) ?? header.parameterTotal;
}

function packedWeightBits(quant) {
  if (!quant) return 4;
  if (quant.quant_method === "gptq") return quant.bits || 4;
  if (quant.quant_method === "compressed-tensors") {
    const weights = Object.values(quant.config_groups || {})[0]?.weights;
    return weights?.num_bits || 4;
  }
  return 4;
}

function compressedGroupSize(quant) {
  const weights = Object.values(quant?.config_groups || {})[0]?.weights;
  return weights?.group_size || 0;
}

export function isQuantizedPath(path, quant) {
  if (!quant) return false;
  // modules_to_not_convert（HF/vLLM 数组约定，GPTQ/FP8 常用）；compressed-tensors
  // 用同义的 "ignore" 数组（vLLM 把 ignore 映射到同一机制），条目支持 "re:" 前缀
  // 正则（compressed-tensors 约定）。命中即**不量化**，优先于 dynamic 表。
  const excluded = quant.modules_to_not_convert ?? quant.ignore;
  if (Array.isArray(excluded) && excluded.length > 0) {
    for (const entry of excluded) {
      const source = entry.startsWith("re:") ? entry.slice(3) : entry;
      let re;
      try {
        re = new RegExp(LITERAL_PATH.test(source) ? canonicalModulePath(source) : source);
      } catch {
        continue;
      }
      if (pathCandidates(path).some((candidate) => re.test(candidate))) return false;
    }
  }
  const dynamic = quant.dynamic;
  if (!dynamic || typeof dynamic !== "object") return true;
  const candidates = pathCandidates(path);
  let matched = null;
  for (const [pattern, value] of Object.entries(dynamic)) {
    const isExclude = pattern.startsWith("-:");
    const source = isExclude ? pattern.slice(2) : pattern;
    let re;
    try {
      // 字面路径走真值绑定的规范化；正则片段原样通过（见上注）
      re = new RegExp(LITERAL_PATH.test(source) ? canonicalModulePath(source) : source);
    } catch {
      continue;
    }
    if (candidates.some((candidate) => re.test(candidate))) matched = isExclude ? false : value;
  }
  return matched !== false;
}
