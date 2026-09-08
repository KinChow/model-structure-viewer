import { textDecoderNetwork } from "./common.js";

export function buildGenericDecoderNetwork(resolved, normalized) {
  // generic 兜底也必须是完整 text decoder（embed + decoder + final norm + lm_head）：
  // 缺失会导致恒等式/成本侧少计 lm_head 与 final norm（合成 llama 变体暴露）。
  return textDecoderNetwork(resolved, normalized, { attentionKind: "gqa", defaultLayerKind: "dense" });
}

