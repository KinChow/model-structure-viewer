import { buildMlaMoeDecoderNetwork } from "./deepseek.js";
import { buildMiniMaxM3Network } from "./minimax.js";
import { buildGqaDecoderNetwork, buildGqaMoeDecoderNetwork, buildMlaMultimodalNetwork, buildQwenMultimodalNetwork } from "./qwen.js";
import { networkSpec } from "./common.js";
import { mtpModule, mtpModuleCount } from "../layers/mtp.js";
import { attentionScheduleOf, layerScheduleOf } from "../config/plan.js";

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

/**
 * W4：把 MTP 挂成与 decoder 平级的模块。对标 vLLM —— MTP 在 registry 里是独立
 * 注册项（DeepseekV32MTPModel / Qwen3_5MTP / MiniMaxM3MTP / Glm5NextMTPModel /
 * KimiK3MTPModel），不是 decoder 的子层，所以这里统一在 buildNetwork 出口追加，
 * 不去改每个 builder。repeat=0 的计费口径见 layers/mtp.js 文件头。
 */
function withMtp(network, normalized) {
  if (!mtpModuleCount(normalized) || !network?.children?.length) return network;
  const schedule = attentionScheduleOf(normalized) || [];
  const layerSchedule = layerScheduleOf(normalized) || [];
  const last = Math.max((normalized.layers || 1) - 1, 0);
  const insertAt = Math.max(network.children.findIndex((c) => c?.type === "decoder" || c?.id === "layers" || c?.id === "language_model"), 0) + 1;
  const mtp = mtpModule("mtp", normalized, {
    attentionKind: schedule[last] || (normalized.kvLoraRank ? "mla" : "gqa"),
    layerKind: layerSchedule[last] || (normalized.experts ? "moe" : "dense"),
  });
  return {
    ...network,
    children: [...network.children.slice(0, insertAt), mtp, ...network.children.slice(insertAt)],
  };
}

export function buildNetwork(resolved, normalized) {
  const architecture = resolved?.architecture;
  const assemble = architecture && MODELS[architecture];
  if (assemble) return withMtp(assemble(resolved, normalized), normalized);
  // vLLM ModelRegistry._raise_for_unsupported：显式不支持 + 枚举支持项。
  return networkSpec(
    "model",
    architecture || normalized.modelType || "Configuration",
    architecture || "unsupported",
    [],
  );
}
