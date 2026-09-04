// shapes.js —— 展示形状层（G2）。
// tensorShapes 不再自行拼装数值，改为读 dims.js 的数值形状 + 标签表渲染展示串。
// 输出与旧版逐字一致（modelArchitecture.test.js 断言依赖）。

import { tensorDims } from "./dims.js";

// 每个形状 key 的维度标签，顺序与 tensorDims 数值位置一一对应。
const SHAPE_LABELS = {
  tokenIds: ["batch", "sequence"],
  hidden: ["batch", "sequence", "hidden size"],
  attentionQuery: ["batch", "sequence", "attention heads", "head dimension"],
  attentionKey: ["batch", "sequence", "key value heads", "head dimension"],
  attentionValue: ["batch", "sequence", "key value heads", "value head dimension"],
  attentionScores: ["batch", "attention heads", "query sequence", "key sequence"],
  attentionProbabilities: ["batch", "attention heads", "query sequence", "key sequence"],
  attentionContext: ["batch", "sequence", "attention heads", "value head dimension"],
  intermediate: ["batch", "sequence", "intermediate size"],
  moeIntermediate: ["tokens_per_expert", "expert intermediate size"],
  routerLogits: ["batch", "sequence", "experts"],
  topExperts: ["batch", "sequence", "experts per token"],
  expertInput: ["tokens_per_expert", "hidden size"],
  logits: ["batch", "sequence", "vocab size"],
  visionInput: ["batch", "image_or_video", "channels", "height", "width"],
  visionOutput: ["batch", "visual_tokens", "vision hidden size"],
};

export function shapeText(parts) {
  return `[${parts.filter(Boolean).join(", ")}]`;
}

/** 由数值 dims + 标签数组渲染展示串：-1/未知位只显示标签名，已知位显示 label=value。 */
export function shapeTextFromDims(labels, dims) {
  return shapeText(
    labels.map((label, i) => (dims[i] == null || dims[i] === -1 ? label : `${label}=${dims[i]}`)),
  );
}

export function tensorShapes(normalized) {
  const dims = tensorDims(normalized);
  const out = {};
  for (const key of Object.keys(dims)) {
    out[key] = shapeTextFromDims(SHAPE_LABELS[key] || [], dims[key]);
  }
  return out;
}

export function shapeFlow(inputShape, outputShape, extra = {}) {
  return {
    input_shape: inputShape,
    output_shape: outputShape,
    ...extra,
  };
}
