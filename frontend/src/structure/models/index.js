import { buildMlaMoeDecoderNetwork } from "./deepseek.js";
import { buildMiniMaxM3Network } from "./minimax.js";
import { buildGqaDecoderNetwork, buildGqaMoeDecoderNetwork, buildMlaMultimodalNetwork, buildQwenMultimodalNetwork } from "./qwen.js";
import { networkSpec } from "./common.js";

function assembleMlaText(resolved, normalized) {
  return normalized.hasVision
    ? buildMlaMultimodalNetwork(resolved, normalized)
    : buildMlaMoeDecoderNetwork(resolved, normalized);
}

function assembleGqaText(resolved, normalized) {
  return normalized.hasVision
    ? buildQwenMultimodalNetwork(resolved, normalized)
    : buildGqaDecoderNetwork(resolved, normalized);
}

function assembleGqaMoe(resolved, normalized) {
  return normalized.hasVision
    ? buildQwenMultimodalNetwork(resolved, normalized)
    : buildGqaMoeDecoderNetwork(resolved, normalized);
}

// 对标 vLLM vllm/model_executor/models/registry.py `_TEXT_GENERATION_MODELS` /
// `_MULTIMODAL_MODELS`：键是 config.architectures[0]，值是该 HF 类的组装。
// 视觉塔是该函数内部的可选子模块，不升格第二种架构名（vLLM Qwen2VL 是另一个
// architectures[0]，不是 Llama 加 hasVision 旗标）。
export const MODELS = {
  DeepseekV3ForCausalLM: assembleMlaText,
  DeepseekV32ForCausalLM: assembleMlaText,
  DeepseekV4ForCausalLM: assembleMlaText,
  Glm4MoeForCausalLM: assembleGqaMoe,
  GlmMoeDsaForCausalLM: assembleMlaText,
  Qwen3ForCausalLM: assembleGqaText,
  Qwen3_5ForConditionalGeneration: assembleGqaText,
  Qwen3_5MoeForConditionalGeneration: assembleGqaMoe,
  Qwen3_5MoeForCausalLM: assembleGqaMoe,
  Qwen3MoeForCausalLM: assembleGqaMoe,
  Qwen4ExpForConditionalGeneration: assembleGqaMoe,
  KimiK25ForConditionalGeneration: assembleMlaText,
  KimiK3ForConditionalGeneration: assembleGqaText,
  Glm5NextForConditionalGeneration: assembleGqaText,
  MiniMaxM2ForCausalLM: assembleGqaMoe,
  MiniMaxM3SparseForConditionalGeneration: buildMiniMaxM3Network,
};

/** 支持的 architectures[0]；不支持诊断用它枚举（vLLM ModelRegistry._raise_for_unsupported）。 */
export const SUPPORTED_MODEL_ARCHITECTURES = Object.keys(MODELS);

export function buildNetwork(resolved, normalized) {
  const architecture = resolved?.architecture;
  const assemble = architecture && MODELS[architecture];
  if (assemble) return assemble(resolved, normalized);
  // vLLM ModelRegistry._raise_for_unsupported：显式不支持 + 枚举支持项。
  return networkSpec(
    "model",
    architecture || normalized.modelType || "Configuration",
    architecture || "unsupported",
    [],
  );
}
