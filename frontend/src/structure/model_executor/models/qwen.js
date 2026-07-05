import { textDecoderNetwork } from "./common.js";

export function buildGqaDecoderNetwork(resolved, normalized) {
  return textDecoderNetwork(resolved, normalized, {
    attentionKind: "gqa",
    defaultLayerKind: "dense",
  });
}

export function buildGqaMoeDecoderNetwork(resolved, normalized) {
  return textDecoderNetwork(resolved, normalized, {
    attentionKind: "gqa",
    defaultLayerKind: "moe",
  });
}
