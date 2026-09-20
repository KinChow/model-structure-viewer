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
// `_MULTIMODAL_MODELS`：键是 config.architectures[0]，值是装配该 HF 类结构的函数。
// 多个 architectures[0] 指向同一装配器有两种情形（均登记于 SGLANG_REUSED_ARCHITECTURES）：
//   (1) 跨模型复用：上游把另一模型整体当骨干（KimiK25 的 language_model=DeepseekV3、
//       GlmMoeDsa(DeepseekV2) 子类）——MSV 直接指向被复用模型的装配器。
//   (2) 同一 modeling 的 config 变体：上游把 dense/MoE、text/VL 拆成兄弟/子类（如
//       qwen3_5.py 里 Qwen3_5ForConditionalGeneration 与 Qwen3_5MoeForConditionalGeneration
//       同为 Qwen3VLForConditionalGeneration 的**兄弟类**，仅 FFN dense/MoE 不同）——MSV
//       用一个 config 驱动的装配器覆盖：dense/MoE 由 config.experts+逐层 layerKinds 分支，
//       text/VL 由 hasVision 分支（见 assembleQwen3_5）。视觉塔是装配内部的可选子模块，不升格第二种架构名。
export const MODELS = {
  DeepseekV3ForCausalLM: assembleDeepseekV3,
  DeepseekV32ForCausalLM: assembleDeepseekV32,
  DeepseekV4ForCausalLM: assembleDeepseekV4,
  DeepseekV41ForCausalLM: assembleDeepseekV41,
  Glm4MoeForCausalLM: assembleGlm4Moe,
  GlmMoeDsaForCausalLM: assembleDeepseekV32,           // (1) SGLang GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)
  Qwen3ForCausalLM: assembleQwen3,
  Qwen3_5ForConditionalGeneration: assembleQwen3_5,    // (2) dense-VL —— assembleQwen3_5 的属主键
  Qwen3_5MoeForConditionalGeneration: assembleQwen3_5, // (2) MoE-VL：与上者同 Qwen3VL 兄弟类，仅 FFN 变 MoE
  Qwen3_5MoeForCausalLM: assembleQwen3_5,              // (2) MoE-text：hasVision=false 文本分支 + MoE
  Qwen3MoeForCausalLM: assembleQwen3Moe,
  Qwen4ExpForConditionalGeneration: assembleQwen4Exp,
  KimiK25ForConditionalGeneration: assembleDeepseekV3,  // (1) SGLang kimi_k25 language_model=DeepseekV3ForCausalLM
  KimiK3ForConditionalGeneration: assembleKimiK3,
  Glm5NextForConditionalGeneration: assembleGlm5Next,
  MiniMaxM2ForCausalLM: assembleMiniMaxM2,
  MiniMaxM3SparseForConditionalGeneration: assembleMiniMaxM3,
};

// 复用/共享登记：凡「与他键共享装配器」的注册项都在此附**已核对 SGLang 源码**的依据。
// 守卫（sglangArchAlignment.test）据此确保每个共享装配器恰有 1 个属主键，其余登记在案——
// 防止把独立架构错当同构（deepseek_v41≠v4 教训：V4.1 曾裸委托 V4，实为独立架构）。
export const SGLANG_REUSED_ARCHITECTURES = {
  KimiK25ForConditionalGeneration: "跨模型复用：SGLang kimi_k25.py:680 self.language_model = DeepseekV3ForCausalLM（文本骨干）→ MSV assembleDeepseekV3。",
  GlmMoeDsaForCausalLM: "跨模型复用：SGLang glm4_moe.py:1483 class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)；DSA(MLA+稀疏 indexer)+MoE 与 MSV deepseek_v32 同构。",
  Qwen3_5MoeForConditionalGeneration: "config 变体：SGLang qwen3_5.py 中 Qwen3_5MoeForConditionalGeneration 与 Qwen3_5ForConditionalGeneration 同为 Qwen3VLForConditionalGeneration 兄弟类（非父子），仅 FFN dense→MoE；MSV 由 config.experts 分支同一 assembleQwen3_5。",
  Qwen3_5MoeForCausalLM: "config 变体：SGLang qwen3_5_text.py:223 class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM) 子类（text-only）；MSV assembleQwen3_5 走 hasVision=false 文本分支 + config.experts MoE。",
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
