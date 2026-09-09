// plan.js —— 组网方案决定（W3-C，§4.7"config 只供数，方案由组网决定"）。
//
// 从 normalizeConfig 原样搬迁的方案类字段：逐层调度与方案选择不再是"归一层"
// 的输出，而由组装点（builders / cost 消费方）按需从 raw config 派生。
// 逻辑与搬迁前逐字等价（plan parity fixture + 59 模型 spec 树哈希双 oracle）。
//
// 家族名仅出现在"读 config 语义"的探测里（与搬迁前一致）；
// 真正的"模型 → 配方"声明表归 structure/archs/（W3-B2）。
import {
  LAYER_KEYS,
  firstNumber,
} from "./normalize.js";
import { archRecipe } from "../archs/index.js";

const PLAN_CACHE = new WeakMap();

function textConfigOf(config) {
  return typeof config?.text_config === "object" && config.text_config ? config.text_config : config;
}

// 与 normalizeConfig 的 visionConfig 判定一致：嵌套 vision_config 或平铺 vision_n_layers
function hasVisionConfig(config) {
  if (typeof config?.vision_config === "object" && config.vision_config) return true;
  return firstNumber(config, ["vision_n_layers"]) != null;
}

// ---- 以下 helper 自 normalizeConfig 原样搬迁 ----

function explicitLayerSchedule(config, layers) {
  const mlpLayerTypes = config?.mlp_layer_types;
  if (Array.isArray(mlpLayerTypes) && mlpLayerTypes.length > 0) {
    return mlpLayerTypes.map((kind) => (String(kind).toLowerCase().includes("dense") ? "dense" : "moe"));
  }
  const moeFreq = config?.moe_layer_freq;
  if (Array.isArray(moeFreq) && moeFreq.length > 0) {
    return moeFreq.map((value) => (value ? "moe" : "dense"));
  }
  const densePrefix = firstNumber(config, ["first_k_dense_replace"]);
  if (densePrefix !== undefined && layers) {
    return Array.from({ length: layers }, (_, index) => (index < densePrefix ? "dense" : "moe"));
  }
  return undefined;
}

function sparseAttentionSchedule(config, layers) {
  const sparseFreq = config?.sparse_attention_config?.sparse_attention_freq;
  if (!Array.isArray(sparseFreq) || sparseFreq.length === 0) return undefined;
  const schedule = sparseFreq.map((value) => (value ? "sparse" : "gqa"));
  if (!layers || schedule.length >= layers) return schedule;
  return schedule.concat(Array.from({ length: layers - schedule.length }, () => "gqa"));
}

function dsaIndexerSchedule(config, layers) {
  const explicitTypes = config?.indexer_types;
  if (Array.isArray(explicitTypes) && explicitTypes.length > 0) {
    return Array.from({ length: layers || explicitTypes.length }, (_, index) =>
      String(explicitTypes[index] || "full").toLowerCase() === "shared" ? "reuse" : "compute");
  }
  const pattern = config?.index_topk_pattern;
  if (Array.isArray(pattern) && pattern.length > 0) {
    return Array.from({ length: layers || pattern.length }, (_, index) =>
      String(pattern[index] || "").toUpperCase() === "S" ? "reuse" : "compute");
  }
  const frequency = firstNumber(config, ["index_topk_freq"]) ?? 1;
  const offset = firstNumber(config, ["index_skip_topk_offset"]) ?? 2;
  return Array.from({ length: layers || 0 }, (_, index) =>
    Math.max(index - offset + 1, 0) % frequency === 0 ? "compute" : "reuse");
}

function attentionKindForLayerType(layerType, useQsa = false) {
  const kind = String(layerType || "").toLowerCase();
  if (kind.includes("linear") || kind.includes("kda") || kind.includes("delta")) return "linear";
  if (kind.includes("deepseek") || kind.includes("mla") || kind.includes("sparse")) return useQsa ? "qsa" : "mla";
  return kind.includes("full") && useQsa ? "qsa" : "gqa";
}

function explicitAttentionSchedule(config, layers) {
  // W5：三处原本用 model_type 精确/子串比较，全部换成 config 字段判据 ——
  //   compress_ratios 数组存在        -> DeepSeek V4 压缩/滑窗混合（dsv4）
  //   index_topk + kv_lora_rank 存在  -> DSA over MLA（逐层同 kind）
  //   attn_output_gate 为真           -> 带输出门的 full attention（qwen35_full）
  // 判据都能在 59 个内置 config 上机械复现，不含任何家族名。
  if (Array.isArray(config?.compress_ratios) && config.compress_ratios.length > 0) {
    return Array.from({ length: layers || config.compress_ratios.length }, () => "dsv4");
  }
  if (firstNumber(config, ["index_topk"]) != null
    && firstNumber(config, ["kv_lora_rank"]) != null
    && !Array.isArray(config?.layer_types)) {
    return Array.from({ length: layers || 0 }, () => "qsa");
  }
  const hasOutputGate = Boolean(config?.attn_output_gate);
  const layerTypes = config?.layer_types;
  const useQsa = firstNumber(config, ["index_n_heads", "indexer_n_heads"]) != null
    || firstNumber(config, ["index_topk", "indexer_budget"]) != null;
  if (Array.isArray(layerTypes) && layerTypes.length > 0) {
    return layerTypes.map((layerType) => {
      const kind = attentionKindForLayerType(layerType, useQsa);
      return hasOutputGate && kind === "gqa" && String(layerType).toLowerCase().includes("full")
        ? "qwen35_full"
        : kind;
    });
  }
  const linearConfig = config?.linear_attn_config;
  if (linearConfig && layers) {
    const full = new Set(Array.isArray(linearConfig.full_attn_layers) ? linearConfig.full_attn_layers : []);
    const linear = new Set(Array.isArray(linearConfig.kda_layers) ? linearConfig.kda_layers : []);
    return Array.from({ length: layers }, (_, index) => {
      const layerNumber = index + 1;
      if (linear.has(layerNumber)) return "linear";
      if (full.has(layerNumber)) return "mla";
      return "gqa";
    });
  }
  return undefined;
}

// ---- 派生入口 ----

/**
 * 从 raw config 派生组网方案（WeakMap 记忆化，可随意重复调用）。
 * @param {object} config 原始 config（normalized.raw 或顶层 config 均可）
 * @returns {{attentionSchedule, layerSchedule, indexerSchedule, linearAttentionMode, normMode, sharedExpertsAreFused, visionInternalMerger, attentionOutputGate}}
 */
export function deriveBuildPlan(config) {
  // 兼容两种入参：raw config，或 normalizeConfig 的输出（解 .raw）
  const source = config?.raw ?? config;
  if (typeof source !== "object" || !source) {
    return deriveBuildPlan({});
  }
  const cached = PLAN_CACHE.get(source);
  if (cached) return cached;
  const textConfig = textConfigOf(source);
  const modelTypeProbe = String(source?.model_type || textConfig?.model_type || "");
  const layers = firstNumber(textConfig, LAYER_KEYS) ?? firstNumber(source, LAYER_KEYS);
  const recipe = archRecipe(Array.isArray(source?.architectures) ? source.architectures[0] : undefined);

  const plan = {
    // 逐层调度（原 normalizeConfig 的 layerSchedule/attentionSchedule/indexerSchedule）。
    // 显式声明优先：入参对象自带方案字段（旧契约调用方/测试 fixture/未来 archs
    // 声明表）直接采用——HF raw config 不含这些键，派生路径不受影响。
    layerSchedule: Array.isArray(source.layerSchedule)
      ? source.layerSchedule
      : explicitLayerSchedule(textConfig, layers) ?? explicitLayerSchedule(source, layers),
    attentionSchedule: Array.isArray(source.attentionSchedule)
      ? source.attentionSchedule
      : explicitAttentionSchedule(textConfig, layers)
        ?? explicitAttentionSchedule(source, layers)
        ?? sparseAttentionSchedule(textConfig, layers),
    // W5：原按 model_type 子串开关，换字段判据 —— indexer 复用调度只在
    // 「有 index_topk 且没有 kpool 压缩」的 DSA 上存在（indexer_types /
    // index_topk_pattern / index_topk_freq 三个键任一驱动）。
    indexerSchedule: Array.isArray(source.indexerSchedule)
      ? source.indexerSchedule
      : (firstNumber(textConfig, ["index_topk"]) ?? firstNumber(source, ["index_topk"])) != null
        && (firstNumber(textConfig, ["index_kpool"]) ?? firstNumber(source, ["index_kpool"]) ?? 1) <= 1
        // compress_ratios 存在 = DeepSeek V4 的压缩层 indexer，走 dsv4_indexer，
        // 没有 DSA 那套「跨层复用 top-k 索引」的调度。
        && !(Array.isArray(textConfig?.compress_ratios) ? textConfig.compress_ratios.length : (source?.compress_ratios || []).length)
        ? dsaIndexerSchedule(textConfig, layers) ?? dsaIndexerSchedule(source, layers)
        : undefined,
    // W5：四个「没有 config 字段判据」的配方位改读 archs/ 的 ARCH_RECIPES
    // 声明表（key = architectures[0] 原字符串，不做子串匹配）。原实现是
    // modelTypeProbe.includes("qwen3_5"/"kimi_k3"/"glm5_next"/...) —— 把人工
    // 适配伪装成自动推断。反例证据：use_gemma_norm 全库仅 2/59 命中，却有 35 个
    // 模型实际走 gemma norm，字段判据根本写不出来，只能登记。
    normMode: typeof source.normMode === "string"
      ? source.normMode
      : (textConfig?.use_gemma_norm ?? source?.use_gemma_norm)
        ? "gemma_rmsnorm"
        : recipe.normMode ?? "rmsnorm",
    linearAttentionMode: typeof source.linearAttentionMode === "string"
      ? source.linearAttentionMode
      : recipe.linearAttentionMode ?? "generic",
    sharedExpertsAreFused: typeof source.sharedExpertsAreFused === "boolean"
      ? source.sharedExpertsAreFused
      : Boolean(recipe.sharedExpertsAreFused),
    visionInternalMerger: typeof source.visionInternalMerger === "boolean"
      ? source.visionInternalMerger
      : Boolean(hasVisionConfig(source) && recipe.visionInternalMerger),
    attentionOutputGate: typeof source.attentionOutputGate === "boolean"
      ? source.attentionOutputGate
      : Boolean(textConfig?.attn_output_gate ?? source?.attn_output_gate),
  };
  PLAN_CACHE.set(config, plan);
  return plan;
}
