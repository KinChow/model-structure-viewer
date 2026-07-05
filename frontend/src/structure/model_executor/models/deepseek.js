import { textDecoderNetwork } from "./common.js";

export function buildMlaMoeDecoderNetwork(resolved, normalized) {
  return textDecoderNetwork(resolved, normalized, {
    attentionKind: "mla",
    defaultLayerKind: "moe",
  });
}
