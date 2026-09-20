// 推理场景的一阶显存核算；结果是理论估算，不是运行时实测。
// 容量只 walk 图上的声明（原则 §3.8）：KV/KDA 来自叶 attributes，不是 config 闭式。

import { paramBytes } from "../structure/operators/formulas/paramDtypes.js";
import { walkStructure } from "./traverse.js";

const BYTES_PER_DTYPE = {
  BF16: 2, F16: 2, FP16: 2, F32: 4, FP32: 4, F8_E4M3: 1, F8_E5M2: 1, F8_E8M0: 1, I8: 1,
  U8: 1, I16: 2, I32: 4, I64: 8,
  // fp4 (float4_e2m1fn_x2)：2 值/字节 = 0.5 B/elem。含 scale 摊销的有效字节：
  //   F4_E4M3S16 = 压缩 KV（E4M3 scale/16）= 0.5+1/16 = 0.5625；F4_E8M0S32 = index（E8M0 scale/32）= 0.53125。
  F4: 0.5, FP4: 0.5, F4_E2M1: 0.5, F4_E4M3S16: 0.5625, F4_E8M0S32: 0.53125,
};

export function bytesPerDtype(dtype, fallback = 2) {
  return BYTES_PER_DTYPE[String(dtype || "").toUpperCase()] ?? fallback;
}

function product(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return 0;
  return shape.reduce((total, value) => total * (Number.isFinite(value) && value >= 0 ? value : 0), 1);
}

/** 把 IR 数值 shape 中的动态维解析为当前推理负载的元素数。 */
export function tensorElements(shape, { batch = 1, sequence = 1, phase = "prefill", attentionHeads = 1, vision = false, visionTokens = 1 } = {}) {
  if (!Array.isArray(shape) || shape.length === 0 || shape.some((value) => value == null)) return 0;
  // Image/video dimensions need an explicit workload shape; never reinterpret
  // unknown spatial dimensions as text sequence length.
  if (shape.length === 5 && shape.every((value) => value === -1)) return 0;
  if (shape.length === 4 && shape[0] === -1 && shape[2] === -1 && shape[3] === -1) {
    return batch * (shape[1] > 0 ? shape[1] : attentionHeads) * (phase === "decode" ? 1 : sequence) * sequence;
  }
  let dynamicIndex = 0;
  return shape.reduce((total, value) => {
    if (value !== -1) return total * value;
    const replacement = dynamicIndex++ === 0
      ? batch
      : vision
        ? visionTokens
        : phase === "decode" ? 1 : sequence;
    return total * replacement;
  }, 1);
}

export function activationTensorBytes(shape, options = {}, bytesPerElement = 2) {
  return tensorElements(shape, options) * bytesPerElement;
}

export function nodeWeightBytes(node) {
  if (!node?.weight_shapes) return 0;
  const dtypes = node.attributes?.weight_dtypes || {};
  const fallback = bytesPerDtype(node.dtype);
  return Object.entries(node.weight_shapes).reduce(
    (total, [name, shape]) => total + product(shape) * bytesPerDtype(dtypes[name] || node.dtype, fallback),
    0,
  );
}

function groupElements(group) {
  return (group.count ?? 1) * (group.matrices ?? 1) * (group.out || 0) * (group.in || 0);
}

function groupBytes(group, fallbackBytes = 2) {
  return groupElements(group) * (group.param_dtype ? paramBytes(group.param_dtype) : fallbackBytes);
}

/** 单节点驻留权重：checkpoint shape 优先，否则 weightMatrices（shared 跳过）。 */
export function nodeWeightCapacityBytes(node, { fallbackBytes = 2 } = {}) {
  const shaped = nodeWeightBytes(node);
  if (shaped > 0) return shaped;
  const declaration = node?.attributes?.weightMatrices;
  if (!Array.isArray(declaration) || declaration.length === 0) return 0;
  return declaration.reduce((total, group) => (
    total + (group.shared ? 0 : groupBytes(group, fallbackBytes))
  ), 0);
}

function isMtpPath(node) {
  const id = String(node?.id || "");
  return node?.type === "mtp" || node?.type === "dspark" || /(^|\.)mtp(\.|$)/.test(id);
}

/**
 * 权重驻留容量：Σ weightMatrices × residentRepeat（MTP repeat=0 仍计入）。
 * 有 weight_shapes（checkpoint 绑定）的叶优先用形状字节，避免与声明双计。
 * includeMtp=false 只用于身份对账：config 声明的投机头不是 checkpoint 实际。
 * ref: vLLM named_parameters；原则 §3.8。
 */
export function graphWeightCapacity(graph, { fallbackBytes = 2, includeMtp = true } = {}) {
  let elements = 0;
  let bytes = 0;
  if (!graph?.nodes?.length) return { elements: 0, bytes: 0 };
  walkStructure(graph, ({ node, resident }) => {
    if (!includeMtp && isMtpPath(node)) return;
    const shaped = nodeWeightBytes(node);
    if (shaped > 0) {
      bytes += shaped * resident;
      return;
    }
    const declaration = node?.attributes?.weightMatrices;
    if (!Array.isArray(declaration) || declaration.length === 0) return;
    for (const group of declaration) {
      if (group.shared) continue;
      const n = groupElements(group);
      elements += n * resident;
      bytes += groupBytes(group, fallbackBytes) * resident;
    }
  });
  return { elements, bytes };
}

/**
 * 身份测试：checkpoint header 的张量名是实际（vLLM load_weights）。
 * mtp_tensor_count>0 → 计入投机头；=0 → 主干。
 * 字段缺席（sidecar 尚未刷新）回退近邻，避免把「还没扫到」当成空声明。
 */
export function declaredElementsForHeader(graph, header) {
  const withMtp = graphWeightCapacity(graph).elements;
  const withoutMtp = graphWeightCapacity(graph, { includeMtp: false }).elements;
  if (Number.isFinite(header?.mtp_tensor_count)) {
    const includeMtp = header.mtp_tensor_count > 0;
    return {
      declared: includeMtp ? withMtp : withoutMtp,
      withMtp,
      withoutMtp,
      includeMtp,
    };
  }
  const headerElements = Number(header?.parameterTotal);
  if (!Number.isFinite(headerElements) || headerElements <= 0) {
    return { declared: withMtp, withMtp, withoutMtp, includeMtp: true };
  }
  const closerWithout = Math.abs(withoutMtp - headerElements) < Math.abs(withMtp - headerElements);
  return {
    declared: closerWithout ? withoutMtp : withMtp,
    withMtp,
    withoutMtp,
    includeMtp: !closerWithout,
  };
}

export function graphShapedWeightBytes(graph) {
  let total = 0;
  if (!graph?.nodes?.length) return 0;
  walkStructure(graph, ({ node, resident }) => {
    total += nodeWeightBytes(node) * resident;
  });
  return total;
}

/** tid2eid 等 buffer：叶 `buffer_elements` × 4B int32。ref: Megatron-Bridge。 */
export function bufferBytesFromGraph(graph, bytesPerElement = 4) {
  let elements = 0;
  if (graph?.nodes?.length) {
    walkStructure(graph, ({ node, multiplier }) => {
      elements += (node?.attributes?.buffer_elements || 0) * multiplier;
    });
  }
  return elements * bytesPerElement;
}

/**
 * 从图上叶声明汇总 KV / KDA 容量。
 * cache_kv_elements / cache_index_elements = 每 token 驻留元素（vLLM AttentionSpec）。
 * state_elements = 每 sequence 的 request state（vLLM kda_state_shape）。
 * 容量 ≠ counts.bytes.kvRead。
 */
export function residentMemoryFromGraph(graph, { kvBytes = 2, batch = 1, tokens = 1 } = {}) {
  let kvBytesPerTokenValue = 0;
  let stateElements = 0;              // 旧口径：无 split 的合成叶按统一 kvBytes（memory.test 不变）
  let stateBytesPerSequenceSplit = 0; // 新口径：conv bf16(2B) + recurrent 逐 dtype（默认 fp32=4B）
  if (graph?.nodes?.length) {
    walkStructure(graph, ({ node, multiplier }) => {
      const attrs = node?.attributes || {};
      if (attrs.modality === "vision") return;
      // dsv4 边际 + 逐 dtype：带 cache_kv_dtype 的叶用 growth（排除有界滑窗）× dtype 字节；
      // 其余叶（非 dsv4 / 合成图）回退全驻留 cache_kv_elements + cache_index_elements × 统一 kvBytes（不变）。
      if (attrs.cache_kv_dtype != null) {
        const kvB = bytesPerDtype(attrs.cache_kv_dtype, kvBytes);
        const idxB = bytesPerDtype(attrs.cache_index_dtype || attrs.cache_kv_dtype, kvBytes);
        kvBytesPerTokenValue += ((attrs.cache_kv_growth_elements || 0) * kvB
          + (attrs.cache_index_growth_elements || 0) * idxB) * multiplier;
      } else {
        kvBytesPerTokenValue += ((attrs.cache_kv_elements || 0) + (attrs.cache_index_elements || 0)) * kvBytes * multiplier;
      }
      // 线性 recurrent state（Bug2）：架构强制 conv=bf16、temporal/ssm 默认 fp32，非用户统一 kvBytes。
      // 带 split 的叶逐 dtype 计；无 split 的旧叶回退统一 kvBytes（合成图/兼容不变）。
      if (attrs.state_recurrent_elements != null || attrs.state_conv_elements != null) {
        const recB = bytesPerDtype(attrs.state_recurrent_dtype, 4);
        stateBytesPerSequenceSplit += ((attrs.state_conv_elements || 0) * 2
          + (attrs.state_recurrent_elements || 0) * recB) * multiplier;
      } else {
        stateElements += (attrs.state_elements || 0) * multiplier;
      }
    });
  }
  const stateBytesPerSequenceValue = stateBytesPerSequenceSplit + stateElements * kvBytes;
  return {
    kvBytesPerToken: kvBytesPerTokenValue,
    kvBytes: kvBytesPerTokenValue * batch * tokens,
    stateBytesPerSequence: stateBytesPerSequenceValue,
    stateBytes: stateBytesPerSequenceValue * batch,
  };
}

export function linearStateBytesPerSequence(graph, bytesPerElement = 2) {
  return residentMemoryFromGraph(graph, { kvBytes: bytesPerElement }).stateBytesPerSequence;
}

export function kvBytesPerToken(graph, kvBytes = 2) {
  return residentMemoryFromGraph(graph, { kvBytes }).kvBytesPerToken;
}

/** 图能证明的驻留合计。activation workspace / CUDA runtime / comm scratch
 *  无法从 config 得到，不计（runtime-unknown，原则 §3.8 / protocol §四）。 */
export function memoryBreakdown({ weightBytes = 0, bufferBytes, graph, batch = 1, tokens = 1, kvBytes = 2 } = {}) {
  const resident = residentMemoryFromGraph(graph, { kvBytes, batch, tokens });
  const buffers = bufferBytes ?? bufferBytesFromGraph(graph);
  const total = weightBytes + buffers + resident.kvBytes + resident.stateBytes;
  return {
    weightBytes,
    bufferBytes: buffers,
    kvBytes: resident.kvBytes,
    kvBytesPerToken: resident.kvBytesPerToken,
    stateBytes: resident.stateBytes,
    stateBytesPerSequence: resident.stateBytesPerSequence,
    totalBytes: total,
  };
}
