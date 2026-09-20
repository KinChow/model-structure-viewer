// 对标 vLLM model_executor/models/deepseek_v2.py（DeepseekV3ForCausalLM）
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import { deepSeekMtpChild } from "./deepseek_mtp.js";

export function assembleDeepseekV3(resolved, normalized) {
  const draft = deepSeekMtpChild(normalized);
  const opts = { draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
