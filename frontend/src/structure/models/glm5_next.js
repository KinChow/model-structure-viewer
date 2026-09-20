// 对标 vLLM models/glm5next：主干 + nvidia/mtp.py（SharedHead 从 deepseek_mtp 引用）
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import { deepSeekMtpChild } from "./deepseek_mtp.js";

export function assembleGlm5Next(resolved, normalized) {
  const draft = deepSeekMtpChild(normalized);
  const opts = { draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
