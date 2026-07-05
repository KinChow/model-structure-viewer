import { buildMlaMoeDecoderNetwork } from "./deepseek.js";
import { buildGenericConfigNetwork, buildGenericDecoderNetwork } from "./generic.js";
import { buildMiniMaxM3Network } from "./minimax.js";
import { buildGqaDecoderNetwork, buildGqaMoeDecoderNetwork } from "./qwen.js";

const MODEL_BUILDERS = {
  "gqa-decoder": buildGqaDecoderNetwork,
  "gqa-moe-decoder": buildGqaMoeDecoderNetwork,
  "mla-moe-decoder": buildMlaMoeDecoderNetwork,
  "multimodal-sparse-moe-decoder": buildMiniMaxM3Network,
  "generic-decoder": buildGenericDecoderNetwork,
};

export function buildNetwork(resolved, normalized) {
  const builder = MODEL_BUILDERS[resolved.canonicalArchitecture] || buildGenericConfigNetwork;
  return builder(resolved, normalized);
}
