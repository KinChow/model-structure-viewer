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
  const schedule = config.layerSchedule || Array.from({ length: layers }, () => experts ? "moe" : "dense");
  const attention = hidden * (heads * qDim + kvHeads * qDim + kvHeads * vDim + heads * vDim);
  const norms = 2 * hidden;
  let decoder = 0;
  for (let i = 0; i < layers; i++) {
    if (schedule[i] === "moe" && experts > 0) {
      decoder += attention + norms + hidden * experts + experts * 3 * hidden * moeIntermediate;
    } else {
      decoder += attention + norms + 3 * hidden * denseIntermediate;
    }
  }
  const embedding = (config.vocabSize || 0) * hidden;
  const lmHead = config.tieWordEmbeddings ? 0 : embedding;
  return embedding + decoder + hidden + lmHead;
}

export function derivedWeightBytes(config = {}, bytesPerElement = 2) {
  return derivedWeightParameters(config) * bytesPerElement;
}
