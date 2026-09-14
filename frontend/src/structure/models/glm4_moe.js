// 对标 vLLM model_executor/models/glm4_moe.py
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import { deepSeekMtpChild } from "./deepseek_mtp.js";

export function assembleGlm4Moe(resolved, normalized) {
  const draft = deepSeekMtpChild(normalized);
  const opts = { defaultLayerKind: "moe", draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
