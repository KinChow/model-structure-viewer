// 对标 vLLM models/deepseek_v32（DeepseekV32ForCausalLM / GlmMoeDsaForCausalLM）
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import { deepSeekMtpChild } from "./deepseek_mtp.js";

export function assembleDeepseekV32(resolved, normalized) {
  const draft = deepSeekMtpChild(normalized);
  const opts = { draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
