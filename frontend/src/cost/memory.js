// 推理场景的一阶显存核算；结果是理论估算，不是运行时实测。

const BYTES_PER_DTYPE = {
  BF16: 2, F16: 2, FP16: 2, F32: 4, FP32: 4, F8_E4M3: 1, F8_E5M2: 1, I8: 1,
  U8: 1, I16: 2, I32: 4, I64: 8,
};

export function bytesPerDtype(dtype, fallback = 2) {
  return BYTES_PER_DTYPE[String(dtype || "").toUpperCase()] ?? fallback;
}

export function product(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return 0;
  return shape.reduce((total, value) => total * (Number.isFinite(value) && value >= 0 ? value : 0), 1);
}

/** 把 IR 数值 shape 中的动态维解析为当前推理负载的元素数。 */
export function tensorElements(shape, { batch = 1, sequence = 1, phase = "prefill", attentionHeads = 1 } = {}) {
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
    const replacement = dynamicIndex++ === 0 ? batch : phase === "decode" ? 1 : sequence;
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

/**
 * KDA runtime state elements per linear-attention layer and sequence.
 * The recurrent matrix and causal-conv history are request state, not token KV.
 * Shape source: vLLM MambaStateShapeCalculator.kda_state_shape.
 */
export function linearStateElementsPerLayer(config = {}, layerIndex = 0) {
  if (config?.attentionSchedule?.[layerIndex] !== "linear") return 0;
  const keyHeads = config.linearKeyHeads || config.attentionHeads || 0;
  const valueHeads = config.linearValueHeads || config.attentionHeads || 0;
  const keyDim = config.linearKeyDim || config.headDim || 0;
  const valueDim = config.linearValueDim || config.valueHeadDim || keyDim;
  const kernel = Math.max(0, (config.linearConvKernelSize || 1) - 1);
  const convElements = keyHeads * keyDim * 2 + valueHeads * valueDim;
  const recurrentElements = valueHeads * valueDim * keyDim;
  return convElements * kernel + recurrentElements;
}

export function linearStateElementsPerSequence(config = {}) {
  const layers = config?.layers || config?.attentionSchedule?.length || 0;
  let total = 0;
  for (let index = 0; index < layers; index += 1) total += linearStateElementsPerLayer(config, index);
  return total;
}

export function linearStateBytesPerSequence(config = {}, bytesPerElement = 2) {
  return linearStateElementsPerSequence(config) * bytesPerElement;
}

// 来源：llm-analysis 的 LLMAnalysis.get_memory_kv_cache_per_layer。
export function kvBytesPerToken(config, kvBytes = 2) {
  const layers = config?.layers || 0;
  const heads = config?.kvHeads || config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const mlaRank = config?.kvLoraRank;
  const ropeDim = config?.qkRopeHeadDim;
  if (Array.isArray(config?.attentionSchedule) && config.attentionSchedule.length && layers) {
    let perLayer = 0;
    for (let index = 0; index < layers; index += 1) {
      const kind = config.attentionSchedule[index] || "gqa";
      if (kind === "linear") {
        // KDA state is request-scoped and is returned separately below.
        perLayer += 0;
      } else if (kind === "dsv4") {
        // 来源：vLLM DeepseekV4Attention / CompressorStateCache。
        // 每层始终有一份 sliding-window MQA KV；压缩层另有 state_dim / compress_ratio。
        const ratio = config.compressRatios?.[index] ?? 0;
        perLayer += headDim;
        if (ratio > 1) perLayer += (2 * (ratio === 4 ? 2 : 1) * headDim) / ratio;
      } else if (kind === "sparse" && config?.modelType === "minimax_m3_vl") {
        perLayer += 2 * heads * headDim;
        if (config.sparseIndexHeads != null && config.sparseIndexDim != null) {
          perLayer += config.sparseIndexHeads * config.sparseIndexDim;
        }
      } else if ((kind === "mla" || kind === "qsa") && mlaRank != null && ropeDim != null) {
        perLayer += mlaRank + ropeDim;
        if (kind === "qsa" && config?.indexerHeadDim != null) perLayer += config.indexerHeadDim;
      } else {
        perLayer += 2 * heads * headDim;
      }
    }
    return perLayer * kvBytes;
  }
  if (mlaRank != null && ropeDim != null) {
    // 来源：vLLM MLAAttentionSpec.head_size_v = 0；MLA 每 token 只存一个 latent，不分离 K/V。
    return layers * (mlaRank + ropeDim) * kvBytes;
  }
  return 2 * layers * heads * headDim * kvBytes;
}

export function activationPeakBytes({ activationPeak = 1.5 * 1024 ** 3 } = {}) {
  return activationPeak;
}

export function memoryBreakdown({ weightBytes = 0, config, batch = 1, tokens = 1, kvBytes = 2,
  activationPeak, runtimeConst = 1.5 * 1024 ** 3, commBuffer = 0 } = {}) {
  const kv = kvBytesPerToken(config, kvBytes) * batch * tokens;
  const state = linearStateBytesPerSequence(config, kvBytes) * batch;
  const activation = activationPeakBytes({ activationPeak });
  const total = weightBytes + kv + state + activation + runtimeConst + commBuffer;
  return { weightBytes, kvBytes: kv, kvBytesPerToken: kvBytesPerToken(config, kvBytes), stateBytes: state,
    stateBytesPerSequence: linearStateBytesPerSequence(config, kvBytes), activationBytes: activation, runtimeBytes: runtimeConst,
    commBufferBytes: commBuffer, totalBytes: total };
}

export { BYTES_PER_DTYPE };
