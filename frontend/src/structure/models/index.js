import { networkSpec } from "./common.js";
import { assembleDeepseekV3 } from "./deepseek_v3.js";
import { assembleDeepseekV32 } from "./deepseek_v32.js";
import { assembleDeepseekV4 } from "./deepseek_v4.js";
import { assembleDeepseekV41 } from "./deepseek_v41.js";
import { assembleGlm4Moe } from "./glm4_moe.js";
import { assembleGlm5Next } from "./glm5_next.js";
import { assembleQwen3 } from "./qwen3.js";
import { assembleQwen3Moe } from "./qwen3_moe.js";
import { assembleQwen3_5 } from "./qwen3_5.js";
import { assembleQwen4Exp } from "./qwen4_exp.js";
import { assembleKimiK3 } from "./kimi_k3.js";
import { assembleMiniMaxM2 } from "./minimax_m2.js";
import { assembleMiniMaxM3 } from "./minimax_m3.js";

// 对标 vLLM vllm/model_executor/models/registry.py `_TEXT_GENERATION_MODELS` /
// `_MULTIMODAL_MODELS`：键是 config.architectures[0]，值是该 HF 类所在模块的组装。
// 视觉塔是该函数内部的可选子模块，不升格第二种架构名。
export const MODELS = {
  DeepseekV3ForCausalLM: assembleDeepseekV3,
  DeepseekV32ForCausalLM: assembleDeepseekV32,
  DeepseekV4ForCausalLM: assembleDeepseekV4,
  DeepseekV41ForCausalLM: assembleDeepseekV41,
  Glm4MoeForCausalLM: assembleGlm4Moe,
  GlmMoeDsaForCausalLM: assembleDeepseekV32,
  Qwen3ForCausalLM: assembleQwen3,
  Qwen3_5ForConditionalGeneration: assembleQwen3_5,
  Qwen3_5MoeForConditionalGeneration: assembleQwen3_5,
  Qwen3_5MoeForCausalLM: assembleQwen3_5,
  Qwen3MoeForCausalLM: assembleQwen3Moe,
  Qwen4ExpForConditionalGeneration: assembleQwen4Exp,
  KimiK25ForConditionalGeneration: assembleDeepseekV3,
  KimiK3ForConditionalGeneration: assembleKimiK3,
  Glm5NextForConditionalGeneration: assembleGlm5Next,
  MiniMaxM2ForCausalLM: assembleMiniMaxM2,
  MiniMaxM3SparseForConditionalGeneration: assembleMiniMaxM3,
};

/** 支持的 architectures[0]；不支持诊断用它枚举（vLLM ModelRegistry._raise_for_unsupported）。 */
export const SUPPORTED_MODEL_ARCHITECTURES = Object.keys(MODELS);

export function buildNetwork(resolved, normalized) {
  const architecture = resolved?.architecture;
  const assemble = architecture && MODELS[architecture];
  if (assemble) return assemble(resolved, normalized);
  return networkSpec(
    "model",
    architecture || normalized.modelType || "Configuration",
    architecture || "unsupported",
    [],
  );
}
