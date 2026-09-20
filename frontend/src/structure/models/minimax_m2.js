// 对标 vLLM MiniMax-M2
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import { deepSeekMtpChild } from "./deepseek_mtp.js";

export function assembleMiniMaxM2(resolved, normalized) {
  const draft = deepSeekMtpChild(normalized);
  const opts = { draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
