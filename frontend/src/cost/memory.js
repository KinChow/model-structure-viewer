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

export function nodeWeightBytes(node) {
  if (!node?.weight_shapes) return 0;
  const dtypes = node.attributes?.weight_dtypes || {};
  const fallback = bytesPerDtype(node.dtype);
  return Object.entries(node.weight_shapes).reduce(
    (total, [name, shape]) => total + product(shape) * bytesPerDtype(dtypes[name] || node.dtype, fallback),
    0,
  );
}

// 来源：llm-analysis 的 LLMAnalysis.get_memory_kv_cache_per_layer。
export function kvBytesPerToken(config, kvBytes = 2) {
  const layers = config?.layers || 0;
  const heads = config?.kvHeads || config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const mlaRank = config?.kvLoraRank;
  const ropeDim = config?.qkRopeHeadDim;
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
  const activation = activationPeakBytes({ activationPeak });
  const total = weightBytes + kv + activation + runtimeConst + commBuffer;
  return { weightBytes, kvBytes: kv, kvBytesPerToken: kvBytesPerToken(config, kvBytes), activationBytes: activation, runtimeBytes: runtimeConst,
    commBufferBytes: commBuffer, totalBytes: total };
}

export { BYTES_PER_DTYPE };
