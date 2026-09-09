import { buildMlaMoeDecoderNetwork } from "./deepseek.js";
import { buildGenericDecoderNetwork } from "./generic.js";
import { buildMiniMaxM3Network } from "./minimax.js";
import { buildGqaDecoderNetwork, buildGqaMoeDecoderNetwork, buildHybridMultimodalNetwork, buildMlaMultimodalNetwork, buildQwenMultimodalNetwork } from "./qwen.js";
import { networkSpec } from "./common.js";
import { mtpModule, mtpModuleCount } from "../layers/mtp.js";
import { deriveBuildPlan } from "../../config/plan.js";

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

/**
 * W4：把 MTP 挂成与 decoder 平级的模块。对标 vLLM —— MTP 在 registry 里是独立
 * 注册项（DeepseekV32MTPModel / Qwen3_5MTP / MiniMaxM3MTP / Glm5NextMTPModel /
 * KimiK3MTPModel），不是 decoder 的子层，所以这里统一在 buildNetwork 出口追加，
 * 不去改每个 builder。repeat=0 的计费口径见 layers/mtp.js 文件头。
 * 实测：51/59 内置模型带 MTP 字段（49 个 1 模块、2 个 3 模块）。
 */
function withMtp(network, normalized) {
  if (!mtpModuleCount(normalized) || !network?.children?.length) return network;
  const plan = deriveBuildPlan(normalized.raw ?? normalized);
  const schedule = plan.attentionSchedule || [];
  const layerSchedule = plan.layerSchedule || [];
  const last = Math.max((normalized.layers || 1) - 1, 0);
  // 插在 decoder **之后、final norm 之前** —— MTP 消费的是主干最后一层的
  // hidden state（vLLM 的 previous_hidden_states），不是 lm_head 的 logits。
  // W5 形状连续性检查抓出：追加到末尾会生成 lm_head → mtp 的边，末维是 vocab，
  // 语义错。
  const insertAt = Math.max(network.children.findIndex((c) => c?.type === "decoder" || c?.id === "decoder"), 0) + 1;
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
  const builder = MODEL_BUILDERS[resolved.canonicalArchitecture];
  if (builder) return withMtp(builder(resolved, normalized), normalized);
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
