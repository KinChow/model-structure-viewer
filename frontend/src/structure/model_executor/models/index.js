import { buildMlaMoeDecoderNetwork } from "./deepseek.js";
import { buildGenericConfigNetwork, buildGenericDecoderNetwork } from "./generic.js";
import { buildMiniMaxM3Network } from "./minimax.js";
import { buildGqaDecoderNetwork, buildGqaMoeDecoderNetwork, buildHybridMultimodalNetwork, buildMlaMultimodalNetwork, buildQwenMultimodalNetwork } from "./qwen.js";

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

export function buildNetwork(resolved, normalized) {
  const builder = MODEL_BUILDERS[resolved.canonicalArchitecture] || buildGenericConfigNetwork;
  return builder(resolved, normalized);
}
