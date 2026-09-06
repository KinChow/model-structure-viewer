// 离线或无 checkpoint 时的模型级权重参数量 fallback；结果必须标记为 derived。
// 来源：llm-analysis 的 get_num_params_* 公式形态；不包含架构特有 bias/额外 head。

export function derivedWeightParameters(config = {}) {
  const layers = config.layers || 0;
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const kvHeads = config.kvHeads || heads;
  const qDim = config.headDim || 0;
  const vDim = config.valueHeadDim || qDim;
  const denseIntermediate = config.intermediateSize || 0;
  const moeIntermediate = config.moeIntermediateSize || denseIntermediate;
  const experts = config.experts || 0;
  const routedExpertHidden = config.routedExpertHiddenSize || hidden;
  const sharedExperts = config.sharedExperts || 0;
  const sharedIntermediate = config.sharedExpertIntermediateSize || denseIntermediate;
  const schedule = config.layerSchedule || Array.from({ length: layers }, () => experts ? "moe" : "dense");
  const attention = hidden * (heads * qDim + kvHeads * qDim + kvHeads * vDim + heads * vDim);
  const norms = config.hyperConnectionCount ? 0 : 2 * hidden;
  let decoder = 0;
  for (let i = 0; i < layers; i++) {
    const attentionKind = config.attentionSchedule?.[i] || "gqa";
    let attentionParameters = attention;
    if (attentionKind === "linear") {
      attentionParameters = config.linearAttentionMode === "glm5_next"
        ? glm5NextLinearAttentionParameters(config)
        : config.linearAttentionMode === "kimi_k3"
          ? kimiK3LinearAttentionParameters(config)
      : config.linearAttentionMode === "qwen4_exp"
          ? qwen4ExpLinearAttentionParameters(config)
            : config.linearAttentionMode === "qwen3_5"
              ? qwen35LinearAttentionParameters(config)
            : genericLinearAttentionParameters(config, { hidden, heads, qDim, vDim });
    } else if (attentionKind === "qwen35_full") {
      attentionParameters = qwen35FullAttentionParameters(config);
    } else if (attentionKind === "qsa" && ["deepseek_v32", "glm_moe_dsa"].includes(config.modelType)) {
      attentionParameters = dsaAttentionParameters(config);
    } else if (attentionKind === "sparse" && config.modelType === "minimax_m3_vl") {
      attentionParameters = minimaxSparseAttentionParameters(config);
    } else if (attentionKind === "dsv4" && config.qLoraRank && config.oLoraRank) {
      attentionParameters = deepseekV4AttentionParameters(config, i);
    } else if (attentionKind === "mla" && config.qLoraRank && config.kvLoraRank) {
      const ropeDim = config.qkRopeHeadDim || 0;
      const nopeDim = Math.max(0, qDim - ropeDim);
      attentionParameters = hidden * config.qLoraRank
        + config.qLoraRank * heads * qDim
        + hidden * (config.kvLoraRank + ropeDim)
        + config.kvLoraRank * (heads * nopeDim + vDim * (config.kvHeads || heads));
    }
    const mhcParameters = config.multiHyperConnection ? mhcLayerParameters(config) : 0;
    const hcParameters = config.hyperConnectionCount ? hyperConnectionLayerParameters(config) : 0;
    if (schedule[i] === "moe" && experts > 0) {
      const routedExperts = experts * 3 * routedExpertHidden * moeIntermediate;
      const latentProjection = routedExpertHidden !== hidden
        ? hidden * routedExpertHidden + routedExpertHidden * hidden
        : 0;
      decoder += attentionParameters + norms + mhcParameters + hcParameters + hidden * experts + routedExperts + latentProjection;
      decoder += (config.sharedExpertsAreFused ? 1 : sharedExperts) * 3 * hidden * sharedIntermediate;
    } else {
      decoder += attentionParameters + norms + mhcParameters + hcParameters + 3 * hidden * denseIntermediate;
    }
    if (config.attnResBlockSize) decoder += 4 * hidden;
  }
  const embedding = (config.vocabSize || 0) * hidden;
  const lmHead = config.tieWordEmbeddings ? 0 : embedding;
  const outputResidual = config.attnResBlockSize ? 2 * hidden : 0;
  const finalHyperConnection = config.hyperConnectionCount ? hyperConnectionFinalParameters(config) : 0;
  return embedding + decoder + hidden + lmHead + outputResidual + finalHyperConnection + derivedVisionParameters(config);
}

function derivedVisionParameters(config) {
  const layers = config.visionLayers || 0;
  const hidden = config.visionHiddenSize || 0;
  const heads = config.visionAttentionHeads || 0;
  const headDim = config.visionHeadDim || (heads ? hidden / heads : 0);
  const intermediate = config.visionIntermediateSize || 0;
  const patch = config.visionPatchSize || 0;
  const temporalPatch = config.visionTemporalPatchSize || 1;
  const channels = config.visionChannels || 0;
  if (!layers || !hidden || !heads || !headDim || !intermediate || !patch || !channels) return 0;
  const patchEmbedding = channels * temporalPatch * patch * patch * hidden;
  const attention = hidden * (3 * heads * headDim) + (heads * headDim) * hidden + 2 * hidden;
  const mlp = config.visionMlpGated ? 3 * hidden * intermediate : 2 * hidden * intermediate;
  const projector = (config.visionOutputSize || hidden) * (config.hiddenSize || hidden);
  return patchEmbedding + layers * (attention + mlp) + projector;
}

function qwen35LinearAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const keyHeads = config.linearKeyHeads || 0;
  const valueHeads = config.linearValueHeads || 0;
  const keyDim = config.linearKeyDim || 0;
  const valueDim = config.linearValueDim || 0;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const convDim = 2 * keyProjection + valueProjection;
  const kernel = config.linearConvKernelSize || 0;
  return hidden * (2 * keyProjection + 2 * valueProjection)
    + 2 * hidden * valueHeads
    + convDim * kernel
    + 2 * valueHeads
    + valueDim
    + valueProjection * hidden;
}

function qwen35FullAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const kvHeads = config.kvHeads || heads;
  const headDim = config.headDim || 0;
  return hidden * (2 * heads * headDim + 2 * kvHeads * headDim)
    + hidden * heads * headDim;
}

function dsaAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const qRank = config.qLoraRank || 0;
  const kvRank = config.kvLoraRank || 0;
  const qkNope = config.qkNopeHeadDim || 0;
  const rope = config.qkRopeHeadDim || 0;
  const qkDim = qkNope + rope;
  const valueDim = config.valueHeadDim || config.headDim || 0;
  const indexHeads = config.indexerNHeads || 0;
  const indexDim = config.indexerHeadDim || 0;
  return hidden * qRank
    + qRank * heads * qkDim
    + hidden * (kvRank + rope)
    + kvRank * heads * (qkNope + valueDim)
    + qRank * indexHeads * indexDim
    + hidden * (indexDim + indexHeads)
    + hidden * heads * valueDim;
}

function minimaxSparseAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const kvHeads = config.kvHeads || heads;
  const headDim = config.headDim || 0;
  const indexHeads = config.sparseIndexHeads || kvHeads;
  const indexDim = config.sparseIndexDim || headDim;
  return hidden * (heads * headDim + 2 * kvHeads * headDim + 2 * indexHeads * indexDim)
    + hidden * heads * headDim;
}

function deepseekV4AttentionParameters(config, layerIndex) {
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const headDim = config.headDim || 0;
  const qRank = config.qLoraRank || 0;
  const outputRank = config.oLoraRank || 0;
  const groups = config.oGroups || 1;
  const ratio = config.compressRatios?.[layerIndex] ?? 0;
  const qkv = hidden * (qRank + headDim);
  const query = qRank * heads * headDim;
  const output = (heads * headDim) * (groups * outputRank) + (groups * outputRank) * hidden;
  if (ratio <= 1) return qkv + query + output;
  const compressor = hidden * 2 * (ratio === 4 ? 2 : 1) * headDim;
  const indexer = ratio === 4
    ? hidden * (config.indexerNHeads || 0) + qRank * (config.indexerNHeads || 0) * (config.indexerHeadDim || 0)
    : 0;
  return qkv + query + output + compressor + indexer;
}

function genericLinearAttentionParameters(config, { hidden, heads, qDim, vDim }) {
  const keyHeads = config.linearKeyHeads || heads;
  const valueHeads = config.linearValueHeads || heads;
  const keyDim = config.linearKeyDim || qDim;
  const valueDim = config.linearValueDim || vDim;
  return hidden * (keyHeads * keyDim + valueHeads * valueDim + hidden);
}

// GLM-5.3-Flash uses six-way fused qkvbfg_a plus separate f_b/g_b projections,
// three depthwise causal convolutions, A_log/dt_bias, gated RMSNorm and o_proj.
function glm5NextLinearAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const heads = config.linearKeyHeads || config.attentionHeads || 0;
  const headDim = config.linearKeyDim || config.headDim || 0;
  const projection = heads * headDim;
  const convKernel = config.linearConvKernelSize || 0;
  return hidden * (3 * projection + heads + 2 * headDim)
    + 2 * headDim * projection
    + 3 * projection * convKernel
    + projection
    + heads
    + headDim
    + projection * hidden;
}

function kimiK3LinearAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const heads = config.linearKeyHeads || config.attentionHeads || 0;
  const headDim = config.linearKeyDim || config.headDim || 0;
  const projection = heads * headDim;
  const convKernel = config.linearConvKernelSize || 0;
  return hidden * 4 * projection
    + hidden * heads
    + hidden * headDim
    + headDim * projection
    + 3 * projection * convKernel
    + projection
    + heads
    + headDim
    + projection * hidden;
}

function qwen4ExpLinearAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const keyHeads = config.linearKeyHeads || config.attentionHeads || 0;
  const valueHeads = config.linearValueHeads || config.attentionHeads || keyHeads;
  const keyDim = config.linearKeyDim || config.headDim || 0;
  const valueDim = config.linearValueDim || config.valueHeadDim || keyDim;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const convDim = 2 * keyProjection + valueProjection;
  const kernel = config.linearConvKernelSize || 0;
  return hidden * (2 * keyProjection + 2 * valueProjection)
    + 2 * hidden * valueHeads
    + convDim * kernel
    + 2 * valueHeads
    + valueDim
    + valueProjection * hidden;
}

function hyperConnectionLayerParameters(config) {
  const streams = config.hyperConnectionCount || 0;
  const hidden = config.hiddenSize || 0;
  const lowrank = config.hyperConnectionLowrank || 0;
  if (!streams || !hidden || !lowrank) return 0;
  const hyperHidden = streams * hidden;
  const oneBranch = hyperHidden + hyperHidden * (lowrank + streams) + lowrank * hyperHidden;
  return 2 * oneBranch;
}

function hyperConnectionFinalParameters(config) {
  const streams = config.hyperConnectionCount || 0;
  const hidden = config.hiddenSize || 0;
  const lowrank = config.hyperConnectionLowrank || 0;
  if (!streams || !hidden || !lowrank) return 0;
  const hyperHidden = streams * hidden;
  return hyperHidden + hyperHidden * lowrank + lowrank * hyperHidden;
}

function mhcLayerParameters(config) {
  const streams = config.mhcNumResidualStreams || 0;
  const hidden = config.hiddenSize || 0;
  if (!streams || !hidden) return 0;
  const mixRows = (2 + streams) * streams;
  const oneProjection = mixRows * streams * hidden + mixRows + 3;
  return 2 * oneProjection;
}

export function derivedWeightBytes(config = {}, bytesPerElement = 2) {
  return derivedWeightParameters(config) * bytesPerElement;
}
