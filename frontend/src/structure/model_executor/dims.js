// dims.js —— 数值形状层（G2）。
// 依据 evolution_design.md §4.3：从 config 推出的数值张量形状，batch/sequence 等自由维用 -1 占位，
// 供 KV cache 与激活张量大小的解析式计算（P1）使用；展示串（shapes.js）读本层渲染。
// 约定：-1 = 自由/动态维；null = 该维度数值未知（config 缺失，P1 无法计算时显式处理）。

export function tensorDims(normalized) {
  const kvHeads = normalized.kvHeads ?? normalized.attentionHeads;
  const valueHeadDim = normalized.valueHeadDim ?? normalized.headDim;
  const moeIntermediate = normalized.moeIntermediateSize ?? normalized.intermediateSize;
  return {
    tokenIds: [-1, -1],
    hidden: [-1, -1, normalized.hiddenSize],
    attentionQuery: [-1, -1, normalized.attentionHeads, normalized.headDim],
    attentionKey: [-1, -1, kvHeads, normalized.headDim],
    attentionValue: [-1, -1, kvHeads, valueHeadDim],
    attentionScores: [-1, -1, -1, -1],
    attentionProbabilities: [-1, -1, -1, -1],
    attentionContext: [-1, -1, normalized.attentionHeads, valueHeadDim],
    intermediate: [-1, -1, normalized.intermediateSize],
    moeIntermediate: [-1, moeIntermediate],
    routerLogits: [-1, -1, normalized.experts],
    topExperts: [-1, -1, normalized.expertsPerToken],
    expertInput: [-1, normalized.hiddenSize],
    logits: [-1, -1, normalized.vocabSize],
    visionInput: [-1, -1, -1, -1, -1],
    visionOutput: [-1, -1, normalized.visionOutputSize ?? normalized.visionHiddenSize],
  };
}
