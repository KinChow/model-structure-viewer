// visionDims.js —— vision 塔的数值维度推导（M11.5 子项 2：自 layers/vision.js 平移）。
// 属 config 层：纯 normalized-config → dims 推导，无任何 import；operators/formulas
// 与 layers/vision.js 双向消费（formulas 不得下探 layers）。

export function visionDimensions(normalized) {
  const hidden = normalized.visionHiddenSize || 0;
  const heads = normalized.visionAttentionHeads || 0;
  // M8-V2：Kimi 系 qkv 宽独立于 hidden（qkv_hidden_size=1536，wqkv 输出 3×1536=4608）
  const qkvHiddenSize = normalized.visionQkvHiddenSize || hidden;
  const headDim = normalized.visionHeadDim || (heads ? qkvHiddenSize / heads : 0);
  const intermediate = normalized.visionIntermediateSize || 0;
  const channels = normalized.visionChannels || 3;
  const patch = normalized.visionPatchSize || 0;
  const temporalPatch = normalized.visionTemporalPatchSize || 1;
  // The visual-token counts are workload assumptions, not matrix widths.
  // Keep both dimensions dynamic so linear MACs do not multiply them twice.
  const tokens = -1;
  const mergedTokens = -1;
  const mergeSize = normalized.visionMergeSize || 1;
  return {
    hidden, heads, headDim, intermediate, channels, patch, temporalPatch, tokens, qkvHiddenSize,
    visual: [-1, tokens, hidden],
    // patch embedding 摊平成一次 GEMM：vLLM qwen2_5_vl.py:557-561 先 view 再 Conv3d，
    // stride == kernel，等价 [L, C·T_p·P²] × [C·T_p·P², hidden]。输入维必须给**摊平后的
    // 单一宽度**，否则 derivedLinearShape 只取最后一维（少了 channels），权重与 MAC 都算错。
    patchInput: [-1, tokens, channels * temporalPatch * patch * patch],
    qkv: [-1, tokens, 3 * qkvHiddenSize],
    q: [-1, tokens, heads, headDim],
    scores: [-1, heads, tokens, tokens],
    context: [-1, tokens, heads, headDim],
    intermediateShape: [-1, tokens, intermediate],
    gatedMlp: Boolean(normalized.visionMlpGated),
    mergedVisual: [-1, mergedTokens, normalized.visionOutputSize || hidden],
    mergedWidth: mergeSize * mergeSize * hidden,
    mergeSize,
  };
}
