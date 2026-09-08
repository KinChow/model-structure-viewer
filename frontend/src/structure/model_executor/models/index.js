import { buildMlaMoeDecoderNetwork } from "./deepseek.js";
import { buildGenericDecoderNetwork } from "./generic.js";
import { buildMiniMaxM3Network } from "./minimax.js";
import { buildGqaDecoderNetwork, buildGqaMoeDecoderNetwork, buildHybridMultimodalNetwork, buildMlaMultimodalNetwork, buildQwenMultimodalNetwork } from "./qwen.js";
import { networkSpec } from "./common.js";

const MODEL_BUILDERS = {
  "gqa-decoder": buildGqaDecoderNetwork,
  "gqa-moe-decoder": buildGqaMoeDecoderNetwork,
  "mla-moe-decoder": buildMlaMoeDecoderNetwork,
  "multimodal-gqa-decoder": buildQwenMultimodalNetwork,
  "multimodal-sparse-moe-decoder": buildMiniMaxM3Network,
  "multimodal-gqa-moe-decoder": buildQwenMultimodalNetwork,
  "multimodal-mla-moe-decoder": buildMlaMultimodalNetwork,
  "hybrid-multimodal-moe-decoder": buildHybridMultimodalNetwork,
  "generic-decoder": buildGenericDecoderNetwork,
};

/** 支持的 canonical architecture 清单；不支持诊断用它枚举（vLLM _raise_for_unsupported 模式）。 */
export const SUPPORTED_MODEL_ARCHITECTURES = Object.keys(MODEL_BUILDERS);

export function buildNetwork(resolved, normalized) {
  const builder = MODEL_BUILDERS[resolved.canonicalArchitecture];
  if (builder) return builder(resolved, normalized);
  // generic-config：config 字段不足，无模板可组网。不伪造结构（成熟做法是
  // 显式不支持 + 枚举支持项，collectDiagnostics 会产出 unsupported 诊断，
  // 前端 banner 告警），只保留空网络让管线走完、诊断可达。
  return networkSpec(
    "model",
    resolved.architecture || normalized.modelType || "Configuration",
    resolved.canonicalArchitecture,
    [],
  );
}
