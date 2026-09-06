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
  const norms = 2 * hidden;
  let decoder = 0;
  for (let i = 0; i < layers; i++) {
    const attentionKind = config.attentionSchedule?.[i] || "gqa";
    let attentionParameters = attention;
    if (attentionKind === "linear") {
      attentionParameters = config.linearAttentionMode === "glm5_next"
        ? glm5NextLinearAttentionParameters(config)
        : genericLinearAttentionParameters(config, { hidden, heads, qDim, vDim });
    } else if (attentionKind === "mla" && config.qLoraRank && config.kvLoraRank) {
      const ropeDim = config.qkRopeHeadDim || 0;
      const nopeDim = Math.max(0, qDim - ropeDim);
      attentionParameters = hidden * config.qLoraRank
        + config.qLoraRank * heads * qDim
        + hidden * (config.kvLoraRank + ropeDim)
        + config.kvLoraRank * (heads * nopeDim + vDim * (config.kvHeads || heads));
    }
    const mhcParameters = config.multiHyperConnection ? mhcLayerParameters(config) : 0;
    if (schedule[i] === "moe" && experts > 0) {
      const routedExperts = experts * 3 * routedExpertHidden * moeIntermediate;
      const latentProjection = routedExpertHidden !== hidden
        ? hidden * routedExpertHidden + routedExpertHidden * hidden
        : 0;
      decoder += attentionParameters + norms + mhcParameters + hidden * experts + routedExperts + latentProjection;
      decoder += sharedExperts * 3 * hidden * sharedIntermediate;
    } else {
      decoder += attentionParameters + norms + mhcParameters + 3 * hidden * denseIntermediate;
    }
    if (config.attnResBlockSize) decoder += 4 * hidden;
  }
  const embedding = (config.vocabSize || 0) * hidden;
  const lmHead = config.tieWordEmbeddings ? 0 : embedding;
  const outputResidual = config.attnResBlockSize ? 2 * hidden : 0;
  return embedding + decoder + hidden + lmHead + outputResidual;
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
