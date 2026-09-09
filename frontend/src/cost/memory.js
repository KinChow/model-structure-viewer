// 推理场景的一阶显存核算；结果是理论估算，不是运行时实测。

import { deriveBuildPlan } from "../structure/model_executor/plan.js";
const planOf = (config) => deriveBuildPlan(config?.raw ?? config);
const BYTES_PER_DTYPE = {
  BF16: 2, F16: 2, FP16: 2, F32: 4, FP32: 4, F8_E4M3: 1, F8_E5M2: 1, I8: 1,
  U8: 1, I16: 2, I32: 4, I64: 8,
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

/**
 * KDA runtime state elements per linear-attention layer and sequence.
 * The recurrent matrix and causal-conv history are request state, not token KV.
 * Shape source: vLLM MambaStateShapeCalculator.kda_state_shape.
 */
export function linearStateElementsPerLayer(config = {}, layerIndex = 0) {
  if (planOf(config).attentionSchedule?.[layerIndex] !== "linear") return 0;
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
  const layers = config?.layers || planOf(config).attentionSchedule?.length || 0;
  let total = 0;
  for (let index = 0; index < layers; index += 1) total += linearStateElementsPerLayer(config, index);
  return total;
}

export function linearStateBytesPerSequence(config = {}, bytesPerElement = 2) {
  return linearStateElementsPerSequence(config) * bytesPerElement;
}

/**
 * 逐层 KV cache 字节明细（每 token）。拆成两支是因为**读法不同**：
 * - `main`：主注意力读的那份（GQA 的 K/V、MLA/QSA 的 latent、DSV4 的滑窗 + 压缩态）。
 *   稀疏模型只读 indexer 选中的位置，所以乘子是 min(S, 稀疏预算)。
 * - `index`：indexer 自己那份 index-k cache（单头、宽 index_head_dim）。
 *   indexer 每次都要扫**全长** S 才能选出 top-k，乘子是 S。
 * 混成一个标量后这两个乘子无法同时对上，KV 读恒等式就只能留松量 —— 这是
 * 把它收到容差 0 的前置条件（plan §四）。
 *
 * 来源：llm-analysis 的 LLMAnalysis.get_memory_kv_cache_per_layer + 各家 vLLM
 * cache spec（DeepseekV4Attention / CompressorStateCache / MLAAttentionSpec 等）。
 */
export function kvBytesPerTokenBreakdown(config, kvBytes = 2) {
  const layers = config?.layers || 0;
  const heads = config?.kvHeads || config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const mlaRank = config?.kvLoraRank;
  const ropeDim = config?.qkRopeHeadDim;
  const schedule = planOf(config).attentionSchedule;
  const main = [];
  const index = [];
  if (Array.isArray(schedule) && schedule.length && layers) {
    for (let i = 0; i < layers; i += 1) {
      const kind = schedule[i] || "gqa";
      let m = 0;
      let x = 0;
      // index-k cache 的宽度（单头）。**判据是字段存在性，不再挂在 MLA 条件下面** ——
      // 原来只有 `(mla|qsa) && kv_lora_rank && rope_dim` 那一支才加 index 项，于是
      // 无 kv_lora 的 QSA（qwen4_exp）与 DSV4 的 indexer cache 整片没算进容量，
      // KV 读恒等式里表现为「indexer 读量 > 容量 0」（2026-09-09 抓出）。
      const indexDim = config?.dsaIndexHeadDim ?? config?.qsaIndexerHeadDim ?? config?.indexerHeadDim ?? null;
      if (kind === "linear") {
        // KDA state 是 request 级的，单独由 linearStateBytesPerSequence 返回。
        m = 0;
      } else if (kind === "dsv4") {
        const ratio = config.compressRatios?.[i] ?? 0;
        m = headDim;
        if (ratio > 1) m += (2 * (ratio === 4 ? 2 : 1) * headDim) / ratio;
        // DeepSeek V4 的 indexer 只挂在 compress_ratio=4 的层上
        //（与 derivedWeights.deepseekV4AttentionParameters 的 indexer 项同判据）。
        if (ratio === 4 && indexDim != null) x = indexDim;
      } else if (kind === "sparse" && config?.modelType === "minimax_m3_vl") {
        m = 2 * heads * headDim;
        if (config.sparseIndexHeads != null && config.sparseIndexDim != null) {
          x = config.sparseIndexHeads * config.sparseIndexDim;
        }
      } else if ((kind === "mla" || kind === "qsa") && mlaRank != null && ropeDim != null) {
        m = mlaRank + ropeDim;
        if (kind === "qsa" && indexDim != null) x = indexDim;
      } else {
        m = 2 * heads * headDim;
        // 逐头 QSA（qwen4_exp：无 kv_lora，K/V 照 GQA 存）同样有一份 index-k cache。
        if (kind === "qsa" && indexDim != null) x = indexDim;
      }
      main.push(m * kvBytes);
      index.push(x * kvBytes);
    }
    return { main, index };
  }
  const perLayer = mlaRank != null && ropeDim != null
    // 来源：vLLM MLAAttentionSpec.head_size_v = 0；MLA 每 token 只存一个 latent。
    ? (mlaRank + ropeDim) * kvBytes
    : 2 * heads * headDim * kvBytes;
  for (let i = 0; i < layers; i += 1) { main.push(perLayer); index.push(0); }
  return { main, index };
}

// 来源：llm-analysis 的 LLMAnalysis.get_memory_kv_cache_per_layer。
export function kvBytesPerToken(config, kvBytes = 2) {
  const { main, index } = kvBytesPerTokenBreakdown(config, kvBytes);
  return main.reduce((s, x) => s + x, 0) + index.reduce((s, x) => s + x, 0);
}


function activationPeakBytes({ activationPeak = 1.5 * 1024 ** 3 } = {}) {
  return activationPeak;
}

export function memoryBreakdown({ weightBytes = 0, bufferBytes = 0, config, batch = 1, tokens = 1, kvBytes = 2,
  activationPeak, runtimeConst = 1.5 * 1024 ** 3, commBuffer = 0 } = {}) {
  const kv = kvBytesPerToken(config, kvBytes) * batch * tokens;
  const state = linearStateBytesPerSequence(config, kvBytes) * batch;
  const activation = activationPeakBytes({ activationPeak });
  // bufferBytes：常驻 buffer（tid2eid 查表等）—— 不是参数、不进 weightBytes，
  // 但加载后常驻显存（2026-09-09 分类裁决，出处见 derivedBufferBytes）。
  const total = weightBytes + bufferBytes + kv + state + activation + runtimeConst + commBuffer;
  return { weightBytes, bufferBytes, kvBytes: kv, kvBytesPerToken: kvBytesPerToken(config, kvBytes), stateBytes: state,
    stateBytesPerSequence: linearStateBytesPerSequence(config, kvBytes), activationBytes: activation, runtimeBytes: runtimeConst,
    commBufferBytes: commBuffer, totalBytes: total };
}

