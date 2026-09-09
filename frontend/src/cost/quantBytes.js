// 量化容量的 per-matrix 精确计算。
//
// 为什么存在：无 checkpoint 证据时，量化容量此前是标量
// `quantizationBytesPerParameter`（所有参数一个字节宽），scale 张量整片缺失。
// 量化容量其实是**逐矩阵**可精确计算的 —— 每个线性层的 [out, in] 在结构树上
// 就有（derivedLinearShape），块的 scale 形状 = ceil(out/b0)·ceil(in/b1)。
//
// 覆盖三种量化方案（2026-09-09 全量落地）：
// - fp8（weight_block_size [b0,b1]）：权重 1B/元素 + scale（scale_fmt "ue8m0"
//   为 1B/块，否则 fp32 4B/块）
// - mxfp8（weight_block_size [1,32]）：权重 1B/元素 + e8m0 scale 1B/块
//   （MXFP8 的 scale 是 32 元素一组的 e8m0）
// - gptq（bits 4, group_size gs）：int4 打包 bits/8 B/元素 + scales fp16
//   2B/(out·in/gs) + qzeros int4 (bits/8)B/(out·in/gs)
//
// 哪些矩阵被量化由 config 的 dynamic/modules_to_not_convert 声明（"-:" 前缀 =
// 排除路径正则，如 GPTQ 的 "-:.*attn.*"），isQuantizedPath 判定。

import { canonicalModulePath } from "../structure/truth/graphTruth.js";

const I4 = 0.5;
const FP8 = 1;
const FP16 = 2;
const FP32 = 4;

function ceilDiv(a, b) {
  return Math.ceil(a / Math.max(b, 1));
}

/** 解析 config 上的 quantization_config（顶层或 text_config 嵌套）。 */
export function quantizationConfigOf(config) {
  return config?.quantization_config
    ?? config?.raw?.quantization_config
    ?? config?.text_config?.quantization_config
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
  return null;
}

/**
 * 路径是否被量化。dynamic / modules_to_not_convert 的语义照抄 vLLM：
 * dynamic 的 key 不带 "-:" 前缀 = 显式量化的模块，"-:" 前缀 = 排除的正则，
 * 后出现的声明覆盖先出现的。
 *
 * 模式是对 **checkpoint 权重前缀**写的，与 msv 的树 id 命名存在已知差异
 * （checkpoint 叫 model.language_model.embed_tokens / model.visual，树 id 是
 * embed_tokens / vision_tower）。桥接分两类，规则各只有一份：
 * - **字面路径** pattern（不含正则元字符）复用 checkpoint 真值绑定既有用的
 *   `canonicalModulePath`（剥 model/language_model 包装、layers→decoder、
 *   visual/vision→vision_tower）。注意不能对正则片段套它 —— 它的
 *   filter(Boolean) 会把 ".*attn.*" 的前导空段吃掉、产出非法正则 "*attn.*"。
 * - **正则片段**（"-:.*attn.*" 这类）原样通过；其中 visual/mtp 等中段名与
 *   树 id 的差异由 path 侧的 checkpoint 命名候选桥接（vision_tower→visual）。
 */
const LITERAL_PATH = /^[A-Za-z0-9_.\-]+$/;

/**
 * 树 id → checkpoint 命名的反向候选（canonicalModulePath 逆映射的最小子集）：
 * checkpoint 把视觉塔叫 model.visual，树 id 是 vision_tower。
 */
function pathCandidates(path) {
  const candidates = [path];
  if (path.includes("vision_tower")) {
    // 两种 checkpoint 命名都要接得住：Qwen3.5 的 model.visual.*（字面路径
    // pattern 带 model. 前缀）与中段正则 ..*visual.*（不带前缀）
    candidates.push(path.replaceAll("vision_tower", "visual"));
    candidates.push(path.replaceAll("vision_tower", "model.visual"));
  }
  return candidates;
}

export function isQuantizedPath(path, quant) {
  if (!quant) return false;
  // modules_to_not_convert（HF/vLLM 数组约定，GPTQ/FP8 常用）：命中即**不量化**，
  // 优先于 dynamic 表（顺序语义：显式排除压过一切包含）。
  const excluded = quant.modules_to_not_convert;
  if (Array.isArray(excluded) && excluded.length > 0) {
    for (const entry of excluded) {
      let re;
      try {
        re = new RegExp(LITERAL_PATH.test(entry) ? canonicalModulePath(entry) : entry);
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
