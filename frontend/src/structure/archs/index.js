// archs/ —— 每架构 checkpoint 映射（W3-B2，§4.3 层 2，§8.1 的"一处"）。
//
// 形态对标 llama.cpp tensor_mapping（候选模式 + {bid} 占位）+ transformers
// WeightConverter，但按自家 §8.1 收敛为"每家族一份覆盖文件"：
// - index.js：canonical 解析（HF 标准命名约定），任何家族开箱即用；
// - <family>.js：仅当某家族的 checkpoint 命名偏离 canonical 时才创建，
//   以覆盖规则数组登记（首个匹配生效）。可逆校验测试（§4.6）发现新偏离
//   → 在对应家族文件加规则。
//
// 解析结果 = { role, bid, domain }，与模板节点（role + 模板 id 内的层号）做
// 连接键匹配；role 见 model_executor/roles.js。

import { SUFFIX_ROLES } from "../model_executor/roles.js";

// ===========================================================================
// ARCH_RECIPES —— 「模型 → 配方」声明表（W5，§8.1 认可的"一处数据文件"）。
//
// 这里放的是**写不出 config 字段判据**的配方位。判据能用字段表达的一律不进来
// （逐层调度走 layer_types / moe_layer_freq / first_k_dense_replace /
// compress_ratios / sparse_attention_freq；kernel 变体走 index_topk /
// index_kpool / indexer_budget / sparse_attention_config 的存在性；输出门走
// attn_output_gate）。剩下这四位在 config 里没有对应字段，属于必须人工登记的
// 家族知识 —— 显式声明比藏在 `model_type.includes(...)` 里诚实（用户原则：
// 不把人工适配当自动推断）。
//
// key = `architectures[0]` 原字符串（vLLM `_MODELS` 的键形态），不做子串匹配。
// 反例证据：`use_gemma_norm` 全库仅 2/59 命中，却有 35 个模型实际走 gemma
// norm —— 这就是「没有字段判据」的实证，只能登记。
export const ARCH_RECIPES = {
  DeepseekV3ForCausalLM: { moeStem: "MoE" },
  DeepseekV32ForCausalLM: { moeStem: "MoE" },
  DeepseekV4ForCausalLM: { moeStem: "MoE" },
  Glm4MoeForCausalLM: { moeStem: "MoE" },
  GlmMoeDsaForCausalLM: { moeStem: "MoE" },
  Glm5NextForConditionalGeneration: { linearAttentionMode: "glm5_next", visionInternalMerger: true, moeStem: "TextMoE", decoderLayerStem: "TextDecoderLayer", mlpStem: "TextMLP", rmsNormStem: "TextRMSNorm", visionBlockStem: "VisionBlock", visionModelStem: "VisionModel", patchMergerStem: "VisionPatchMerger", attentionStemByKind: { linear: "TextLinearAttention", gqa: "TextAttention", qwen35_full: "TextAttention" } },
  KimiK25ForConditionalGeneration: { linearAttentionMode: "kimi", moeStem: "MoE", modulePrefix: "DeepseekV3" },
  KimiK3ForConditionalGeneration: { linearAttentionMode: "kimi_k3", sharedExpertsAreFused: true, moeStem: "SparseMoeBlock", modulePrefix: "Kimi", attentionStemByKind: { mla: "MLAAttention", linear: "DeltaAttention" } },
  MiniMaxM2ForCausalLM: { moeStem: "SparseMoeBlock" },
  MiniMaxM3SparseForConditionalGeneration: { normMode: "gemma_rmsnorm", moeStem: "SparseMoeBlock", modulePrefix: "MiniMaxM3VL", visionBlockStem: "VisionEncoderLayer", visionModelStem: "VisionModel" },
  Qwen3_5ForConditionalGeneration: { normMode: "gemma_rmsnorm", linearAttentionMode: "qwen3_5", visionInternalMerger: true, patchMergerStem: "VisionPatchMerger" },
  Qwen3_5MoeForCausalLM: { normMode: "gemma_rmsnorm", linearAttentionMode: "qwen3_5", moeStem: "SparseMoeBlock" },
  Qwen3_5MoeForConditionalGeneration: { normMode: "gemma_rmsnorm", linearAttentionMode: "qwen3_5", visionInternalMerger: true, moeStem: "SparseMoeBlock", patchMergerStem: "VisionPatchMerger" },
  Qwen4ExpForConditionalGeneration: { linearAttentionMode: "qwen4_exp", visionInternalMerger: true, moeStem: "TextSparseMoeBlock", attentionStem: "TextAttention", decoderLayerStem: "TextDecoderLayer", mlpStem: "TextMLP", rmsNormStem: "TextRMSNorm", gatedResidualStem: "TextGatedResidual", pleStem: "TextPLELayer", visionBlockStem: "VisionBlock", visionModelStem: "VisionModel", patchMergerStem: "VisionPatchMerger" },
};

/** 取某架构的配方；未登记的架构返回空配方（走各位的默认值）。 */
export function archRecipe(architecture) {
  return ARCH_RECIPES[String(architecture || "")] || {};
}

// transformers 把入口类写成 `{Prefix}ForCausalLM` / `{Prefix}ForConditionalGeneration`，
// 模块类写成 `{Prefix}Attention` / `{Prefix}DecoderLayer` / `{Prefix}MLP`。
// 出处：modeling_llama.py `LlamaForCausalLM` → `LlamaAttention`；
// modeling_minimax_m2.py `MiniMaxM2ForCausalLM` → `MiniMaxM2Attention`。
// GQA/MLA 是 config 字段，不进类名——没有 GQAAttention 这种算法类。
const HF_TASK_SUFFIXES = [
  "ForConditionalGeneration",
  "ForCausalLM",
  "ForSequenceClassification",
  "ForTokenClassification",
  "ForQuestionAnswering",
];

export function hfModulePrefix(architecture) {
  const recipe = archRecipe(architecture);
  if (recipe.modulePrefix) return recipe.modulePrefix;
  let name = String(architecture || "").trim();
  if (!name) return "";
  for (const suffix of HF_TASK_SUFFIXES) {
    if (name.endsWith(suffix)) {
      name = name.slice(0, -suffix.length);
      break;
    }
  }
  return name;
}

export function hfModuleClass(architecture, stem, { fallback } = {}) {
  const prefix = hfModulePrefix(architecture);
  if (prefix && stem) return `${prefix}${stem}`;
  return fallback || stem || null;
}

export function hfNamedClass(normalized, recipeKey, defaultStem, fallback, { kind } = {}) {
  const architecture = normalized?.architecture;
  const recipe = archRecipe(architecture);
  const byKind = kind != null ? recipe.attentionStemByKind?.[kind] : undefined;
  const stem = byKind || recipe[recipeKey] || defaultStem;
  return hfModuleClass(architecture, stem, { fallback: fallback || defaultStem });
}

const WRAPPERS = new Set(["model", "language_model", "model_tower"]);
const BID_MARKERS = new Set(["layers", "blocks", "h", "layer"]);
const EXPERT_MARKERS = new Set(["experts", "expert"]);
const VISION_MARKERS = new Set(["visual", "vision", "vision_tower", "mm"]);

// 家族覆盖注册表：modelType → 规则数组（首个匹配生效）。
// 规则形态：{ when: (segments) => boolean, scope?: string, role?: string, bid?: number|null }
// 当前所有内置家族均符合 canonical 命名；首个偏离出现时新建 <family>.js 并在此登记。
export const FAMILY_OVERRIDES = {};

function applyOverrides(segments, modelType) {
  const rules = FAMILY_OVERRIDES[modelType];
  if (!rules) return {};
  for (const rule of rules) {
    if (rule.when(segments)) return rule;
  }
  return {};
}

/**
 * checkpoint 模块路径 → { role, bid, domain }（无法定位 role 时 role=undefined，
 * 绑定退化为路径匹配兜底）。
 * @param {string} moduleId checkpoint 模块路径（如 "model.layers.0.self_attn.q_proj"）
 * @param {string} modelType 用于家族覆盖分派
 */
export function resolveCheckpointModule(moduleId, modelType) {
  const segments = String(moduleId || "").split(".").filter(Boolean);
  while (segments.length > 0 && WRAPPERS.has(segments[0])) segments.shift();

  const override = applyOverrides(segments, modelType);
  const domain = segments.some((s) => VISION_MARKERS.has(s)) ? "vision" : "text";

  let bid = null;
  for (let i = 1; i < segments.length; i++) {
    if (BID_MARKERS.has(segments[i - 1]) && /^\d+$/.test(segments[i])) {
      bid = Number(segments[i]);
      break;
    }
  }
  const hasExpertIndex = segments.some((s, i) => EXPERT_MARKERS.has(s) && /^\d+$/.test(segments[i + 1] || ""));

  const suffix = segments.at(-1);
  let role = override.role ?? SUFFIX_ROLES[suffix];
  // checkpoint 顶层 final norm（model.norm）：通用后缀 "norm" 不入表（vision
  // 的 merger.norm 会误标），仅在剥完 wrapper 后深度为 1 时特判。
  if (!override.role && segments.length === 1 && suffix === "norm") {
    role = "output_norm";
  }
  if (role && !override.role) {
    const scoped = override.scope
      ?? (segments.includes("shared_experts") ? "shexp" : undefined);
    // shexp 作用域改写由 roles.js 的 ROLE_SCOPES 承担
    if (scoped === "shexp") {
      role = { ffn_gate: "ffn_gate_shexp", ffn_up: "ffn_up_shexp", ffn_down: "ffn_down_shexp" }[role] ?? role;
    }
  }
  return { role, bid, domain, hasExpertIndex };
}

/** §4.6 可逆校验用：role → 合法 checkpoint 后缀集合（SUFFIX_ROLES 的逆像）。 */
export function suffixesForRole(role) {
  return Object.entries(SUFFIX_ROLES)
    .filter(([, r]) => r === role)
    .map(([suffix]) => suffix);
}

/** 绑定连接键：domain|bid|role。role 未知的两侧都走路径兜底，不经此键。 */
export function bindingKey({ role, bid, domain }) {
  return `${domain ?? "text"}|${bid ?? "-"}|${role ?? "-"}`;
}

/** 模板节点 → 绑定键（role 已在 B1 落树；bid/域从模板 id 提取）。 */
export function templateBindingKey(node) {
  const id = String(node?.canonical_id || node?.module_id || node?.id || "");
  const segments = id.split(".").filter(Boolean);
  const domain = segments.some((s) => VISION_MARKERS.has(s)) ? "vision" : "text";
  let bid = null;
  for (const segment of segments) {
    if (/^\d+$/.test(segment)) {
      bid = Number(segment);
      break;
    }
  }
  return bindingKey({ role: node.role, bid, domain });
}
