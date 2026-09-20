// archs/ —— 每 architectures[0] 的类名 / 路径例外（写在 modeling 之前的过渡表）。
//
// 类名与 _modules 键对标 vLLM/SGLang 每个模型文件手写 class Foo / self.mlp，
// 不从 architectures[0] 剥前缀再拼。checkpoint 绑定按模块路径相等，不再经角色表。

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
//
// 命名不是配方轴。路径默认 self_attn / mlp；类名配方写 HF 全名，没写就用词干
//（Attention / MLP / MoE），不从 architectures[0] 剥前缀再拼。vLLM/SGLang 每个
// 模型文件手写 class Foo，没有这套构词器。attn/ffn 变体清单不进本表。
export const ARCH_RECIPES = {
  DeepseekV4ForCausalLM: { moeClass: "DeepseekV4SparseMoeBlock", hashMoE: true, compressorApe: true },
  // DeepSeek V4.1：与 V4 同族（sqrtsoftplus/noaux_tc 路由、o_lora 分组输出投影、
  // 逐层 compress_ratios、MHC、DSpark 投机头、视觉塔）。差异 = 无 hash 层
  // （config 无 num_hash_layers → numHashLayers=0，moe.js 的 isHashMoe 恒 false，
  // 全部走 sqrtsoftplus routed MoE）+ 每层入口 Engram（engramLayerIds 注入，
  // 见 decoderLayer.js）。类名取自随附 model.py 原生实现（Block/Transformer/MoE）。
  DeepseekV41ForCausalLM: {
    hashMoE: true,
    moeClass: "MoE",
    decoderLayerClass: "Block",
    modelClass: "Transformer",
  },
  Glm5NextForConditionalGeneration: {
    linearAttentionMode: "glm5_next",
    visionInternalMerger: true,
    visionMergerMlp: true,
    visionAttr: "visual",
    moeClass: "Glm5NextTextMoE",
    decoderLayerClass: "Glm5NextTextDecoderLayer",
    mlpClass: "Glm5NextTextMLP",
    rmsNormClass: "Glm5NextTextRMSNorm",
    attentionClass: { linear: "Glm5NextTextLinearAttention", gqa: "Glm5NextTextAttention", qwen35_full: "Glm5NextTextAttention" },
  },
  KimiK25ForConditionalGeneration: {
    linearAttentionMode: "kimi",
    visionAttr: "vision_tower",
    moeClass: "DeepseekV3MoE",
    mlpClass: "DeepseekV3MLP",
    attentionClass: "DeepseekV3Attention",
    decoderLayerClass: "DeepseekV3DecoderLayer",
    rmsNormClass: "DeepseekV3RMSNorm",
    modelClass: "DeepseekV3Model",
  },
  KimiK3ForConditionalGeneration: {
    linearAttentionMode: "kimi_k3",
    sharedExpertsAreFused: true,
    ffn: { moe: "block_sparse_moe" },
    moeClass: "KimiSparseMoeBlock",
    latentMoE: true,
    mlpClass: "KimiMLP",
    attentionClass: { mla: "KimiMLAAttention", linear: "KimiDeltaAttention" },
    decoderLayerClass: "KimiDecoderLayer",
    rmsNormClass: "KimiRMSNorm",
    modelClass: "KimiLinearModel",
    visionAttr: "vision_tower",
  },
  MiniMaxM2ForCausalLM: { ffn: "block_sparse_moe", moeClass: "MiniMaxM2SparseMoeBlock", fusedQkv: true, sigmoidRouter: true },
  Glm4MoeForCausalLM: { fusedQkv: true, sigmoidRouter: true },
  MiniMaxM3SparseForConditionalGeneration: {
    normMode: "gemma_rmsnorm",
    moeClass: "MiniMaxM3VLSparseMoeBlock",
    mlpClass: "MiniMaxM3VLDenseMLP",
    attentionClass: "MiniMaxM3VLAttention",
    decoderLayerClass: "MiniMaxM3VLDecoderLayer",
    rmsNormClass: "MiniMaxM3VLRMSNorm",
    modelClass: "MiniMaxM3VLTextModel",
    visionBlockClass: "MiniMaxM3VLVisionEncoderLayer",
    visionModelClass: "MiniMaxM3VLVisionModel",
    visionAttr: "vision_tower",
    layersAttr: "language_model.layers",
    swigluVariant: "swigluoai",
    sigmoidRouter: true,
  },
  Qwen3_5ForConditionalGeneration: {
    normMode: "gemma_rmsnorm",
    linearAttentionMode: "qwen3_5",
    visionInternalMerger: true,
    visionAttr: "visual",
    attention: { linear: "linear_attn" },
    attentionClass: { linear: "Qwen3_5GatedDeltaNet", qwen35_full: "Qwen3_5Attention" },
    decoderLayerClass: "Qwen3_5DecoderLayer",
    mlpClass: "Qwen3_5MLP",
    rmsNormClass: "Qwen3_5RMSNorm",
    modelClass: "Qwen3_5TextModel",
  },
  Qwen3_5MoeForCausalLM: {
    normMode: "gemma_rmsnorm",
    linearAttentionMode: "qwen3_5",
    attention: { linear: "linear_attn" },
    moeClass: "Qwen3_5MoeSparseMoeBlock",
    attentionClass: { linear: "Qwen3_5MoeGatedDeltaNet", qwen35_full: "Qwen3_5MoeAttention" },
    decoderLayerClass: "Qwen3_5MoeDecoderLayer",
    mlpClass: "Qwen3_5MoeMLP",
    rmsNormClass: "Qwen3_5MoeRMSNorm",
  },
  Qwen3_5MoeForConditionalGeneration: {
    normMode: "gemma_rmsnorm",
    linearAttentionMode: "qwen3_5",
    visionInternalMerger: true,
    visionAttr: "visual",
    attention: { linear: "linear_attn" },
    moeClass: "Qwen3_5MoeSparseMoeBlock",
    attentionClass: { linear: "Qwen3_5MoeGatedDeltaNet", qwen35_full: "Qwen3_5MoeAttention" },
    decoderLayerClass: "Qwen3_5MoeDecoderLayer",
    mlpClass: "Qwen3_5MoeMLP",
    rmsNormClass: "Qwen3_5MoeRMSNorm",
    modelClass: "Qwen3_5MoeTextModel",
  },
  Qwen4ExpForConditionalGeneration: {
    linearAttentionMode: "qwen4_exp",
    visionInternalMerger: true,
    visionAttr: "visual",
    attention: { linear: "linear_attn" },
    moeClass: "Qwen4ExpTextSparseMoeBlock",
    decoderLayerClass: "Qwen4ExpTextDecoderLayer",
    mlpClass: "Qwen4ExpTextMLP",
    rmsNormClass: "Qwen4ExpTextRMSNorm",
    gatedResidualClass: "Qwen4ExpTextGatedResidual",
    layerMix: "hyper_connection",
    pleClass: "Qwen4ExpTextPLELayer",
    attentionClass: { linear: "Qwen4ExpTextGatedDeltaNet", qsa: "Qwen4ExpTextAttention", gqa: "Qwen4ExpTextAttention" },
  },
};

/** 取某架构的配方；未登记的架构返回空配方（走各位的默认值）。 */
export function archRecipe(architecture) {
  return ARCH_RECIPES[String(architecture || "")] || {};
}

// 类名：配方 *Class 写 HF 全名；没写就用词干。不从 architectures[0] 剥前缀再拼
//（vLLM/SGLang 每个模型文件手写 class Foo）。
export function hfNamedClass(normalized, classKey, defaultStem, fallback, { kind } = {}) {
  const override = archRecipe(normalized?.architecture)[classKey];
  if (kind != null && override && typeof override === "object" && override[kind]) return override[kind];
  if (typeof override === "string") return override;
  return fallback || defaultStem || null;
}

// 路径段 = HF self.<name>。默认 self_attn / mlp；配方 attention / ffn 只登记偏离。
export function hfAttentionAttr(normalized, attentionKind) {
  const attention = archRecipe(normalized?.architecture).attention;
  if (attentionKind && attention?.[attentionKind]) return attention[attentionKind];
  return "self_attn";
}

export function hfFfnAttr(normalized, layerKind) {
  const ffn = archRecipe(normalized?.architecture).ffn;
  if (ffn && typeof ffn === "object" && layerKind && ffn[layerKind]) return ffn[layerKind];
  if (typeof ffn === "string") return ffn;
  return "mlp";
}

// 文本层 ModuleList 与视觉塔的 _modules 键。默认 layers（Llama/Qwen/DeepSeek 文本塔）；
// Qwen/GLM VL 的视觉塔叫 visual（modeling_qwen3_5 / modeling_glm5_next / modeling_qwen4_exp）；
// Kimi / MiniMax-M3 叫 vision_tower。MiniMax-M3 文本塔挂在 language_model 下。
export function hfLayersAttr(normalized) {
  return archRecipe(normalized?.architecture).layersAttr || "layers";
}

export function hfVisionAttr(normalized) {
  return archRecipe(normalized?.architecture).visionAttr || "visual";
}

function recipeOf(config) {
  return archRecipe(config?.architecture || (Array.isArray(config?.architectures) ? config.architectures[0] : undefined));
}

/** 配方位：写不出 config 字段判据的家族知识，直接读 ARCH_RECIPES。
 *  对象上已有同名布尔/字符串时用它（测试夹具 / 组网出口显式声明），否则读配方。 */
export function recipeLinearAttentionMode(config) {
  const raw = config?.raw ?? config;
  if (typeof raw?.linearAttentionMode === "string") return raw.linearAttentionMode;
  if (typeof config?.linearAttentionMode === "string") return config.linearAttentionMode;
  return recipeOf(config).linearAttentionMode || "generic";
}

export function recipeNormMode(config) {
  const raw = config?.raw ?? config;
  if (typeof raw?.normMode === "string") return raw.normMode;
  if (typeof config?.normMode === "string") return config.normMode;
  const text = typeof raw?.text_config === "object" && raw.text_config ? raw.text_config : raw;
  if (text?.use_gemma_norm ?? raw?.use_gemma_norm) return "gemma_rmsnorm";
  return recipeOf(config).normMode || "rmsnorm";
}

export function recipeSharedExpertsAreFused(config) {
  const raw = config?.raw ?? config;
  if (typeof raw?.sharedExpertsAreFused === "boolean") return raw.sharedExpertsAreFused;
  if (typeof config?.sharedExpertsAreFused === "boolean") return config.sharedExpertsAreFused;
  return Boolean(recipeOf(config).sharedExpertsAreFused);
}

export function recipeVisionInternalMerger(config) {
  const raw = config?.raw ?? config;
  if (typeof raw?.visionInternalMerger === "boolean") return raw.visionInternalMerger;
  if (typeof config?.visionInternalMerger === "boolean") return config.visionInternalMerger;
  const hasVision = typeof raw?.vision_config === "object" && raw.vision_config
    || raw?.hasVision
    || (typeof raw?.vision_n_layers === "number");
  return Boolean(hasVision && recipeOf(config).visionInternalMerger);
}

export function recipeAttentionOutputGate(config) {
  const raw = config?.raw ?? config;
  if (typeof raw?.attentionOutputGate === "boolean") return raw.attentionOutputGate;
  if (typeof config?.attentionOutputGate === "boolean") return config.attentionOutputGate;
  const text = typeof raw?.text_config === "object" && raw.text_config ? raw.text_config : raw;
  return Boolean(text?.attn_output_gate ?? raw?.attn_output_gate);
}

export function recipeFlag(config, key) {
  return Boolean(recipeOf(config)[key]);
}

export function recipeValue(config, key) {
  return recipeOf(config)[key];
}
