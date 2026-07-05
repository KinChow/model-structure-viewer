function dimension(name, value) {
  return value === undefined || value === null ? name : `${name}=${value}`;
}

export function shapeText(parts) {
  return `[${parts.filter(Boolean).join(", ")}]`;
}

export function tensorShapes(normalized) {
  const hidden = dimension("hidden size", normalized.hiddenSize);
  const attentionHeads = dimension("attention heads", normalized.attentionHeads);
  const keyValueHeads = dimension("key value heads", normalized.kvHeads ?? normalized.attentionHeads);
  const headDim = dimension("head dimension", normalized.headDim);
  const valueHeadDim = dimension("value head dimension", normalized.valueHeadDim ?? normalized.headDim);
  const intermediate = dimension("intermediate size", normalized.intermediateSize);
  const moeIntermediate = dimension(
    "expert intermediate size",
    normalized.moeIntermediateSize ?? normalized.intermediateSize,
  );
  const experts = dimension("experts", normalized.experts);
  const expertsPerToken = dimension("experts per token", normalized.expertsPerToken);
  const vocab = dimension("vocab size", normalized.vocabSize);
  const visionHidden = dimension("vision hidden size", normalized.visionHiddenSize);

  return {
    tokenIds: shapeText(["batch", "sequence"]),
    hidden: shapeText(["batch", "sequence", hidden]),
    attentionQuery: shapeText(["batch", "sequence", attentionHeads, headDim]),
    attentionKey: shapeText(["batch", "sequence", keyValueHeads, headDim]),
    attentionValue: shapeText(["batch", "sequence", keyValueHeads, valueHeadDim]),
    attentionScores: shapeText(["batch", "attention heads", "query sequence", "key sequence"]),
    attentionProbabilities: shapeText(["batch", "attention heads", "query sequence", "key sequence"]),
    attentionContext: shapeText(["batch", "sequence", attentionHeads, valueHeadDim]),
    intermediate: shapeText(["batch", "sequence", intermediate]),
    moeIntermediate: shapeText(["tokens_per_expert", moeIntermediate]),
    routerLogits: shapeText(["batch", "sequence", experts]),
    topExperts: shapeText(["batch", "sequence", expertsPerToken]),
    expertInput: shapeText(["tokens_per_expert", hidden]),
    logits: shapeText(["batch", "sequence", vocab]),
    visionInput: shapeText(["batch", "image_or_video", "channels", "height", "width"]),
    visionOutput: shapeText(["batch", "visual_tokens", visionHidden]),
  };
}

export function shapeFlow(inputShape, outputShape, extra = {}) {
  return {
    input_shape: inputShape,
    output_shape: outputShape,
    ...extra,
  };
}
