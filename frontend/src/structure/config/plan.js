// plan.js —— 组网按层读 HF 字段（对标 vLLM DecoderLayer.__init__(layer_idx)）。
//
// 不是产品类型。没有八字段 bag。组网 / 身份测试按函数取：
//   layerScheduleOf / attentionScheduleOf / indexerScheduleOf
// 配方旗标（linearAttentionMode / normMode / fused / merger / gate）走 archs/
// recipe*，不经本文件。
import { LAYER_KEYS, firstNumber } from "./normalize.js";

function textConfigOf(config) {
  const source = config?.raw ?? config;
  if (typeof source !== "object" || !source) return {};
  return typeof source.text_config === "object" && source.text_config ? source.text_config : source;
}

function layerCountOf(config) {
  const source = config?.raw ?? config;
  const text = textConfigOf(config);
  return firstNumber(text, LAYER_KEYS) ?? firstNumber(source, LAYER_KEYS);
}

function rawSource(config) {
  const source = config?.raw ?? config;
  return typeof source === "object" && source ? source : {};
}

// ---- 以下 helper 自 normalizeConfig 原样搬迁 ----

export function explicitLayerSchedule(config, layers) {
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

export function sparseAttentionSchedule(config, layers) {
  const sparseFreq = config?.sparse_attention_config?.sparse_attention_freq;
  if (!Array.isArray(sparseFreq) || sparseFreq.length === 0) return undefined;
  const schedule = sparseFreq.map((value) => (value ? "sparse" : "gqa"));
  if (!layers || schedule.length >= layers) return schedule;
  return schedule.concat(Array.from({ length: layers - schedule.length }, () => "gqa"));
}

export function dsaIndexerSchedule(config, layers) {
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

export function explicitAttentionSchedule(config, layers) {
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

/** dense/moe 逐层表。读 mlp_layer_types / moe_layer_freq / first_k_dense_replace。 */
export function layerScheduleOf(config) {
  const source = rawSource(config);
  const text = textConfigOf(config);
  const layers = layerCountOf(config);
  if (Array.isArray(source.layerSchedule)) return source.layerSchedule;
  return explicitLayerSchedule(text, layers) ?? explicitLayerSchedule(source, layers);
}

/** 注意力 kind 逐层表。读 layer_types / compress_ratios / index_topk / linear_attn_config。 */
export function attentionScheduleOf(config) {
  const source = rawSource(config);
  const text = textConfigOf(config);
  const layers = layerCountOf(config);
  if (Array.isArray(source.attentionSchedule)) return source.attentionSchedule;
  return explicitAttentionSchedule(text, layers)
    ?? explicitAttentionSchedule(source, layers)
    ?? sparseAttentionSchedule(text, layers);
}

/** DSA indexer compute/reuse 逐层表。有 compress_ratios 的 V4 压缩层不走这套。 */
export function indexerScheduleOf(config) {
  const source = rawSource(config);
  const text = textConfigOf(config);
  const layers = layerCountOf(config);
  if (Array.isArray(source.indexerSchedule)) return source.indexerSchedule;
  const hasTopk = (firstNumber(text, ["index_topk"]) ?? firstNumber(source, ["index_topk"])) != null;
  const kpool = firstNumber(text, ["index_kpool"]) ?? firstNumber(source, ["index_kpool"]) ?? 1;
  const compressLen = Array.isArray(text?.compress_ratios)
    ? text.compress_ratios.length
    : (source?.compress_ratios || []).length;
  if (!hasTopk || kpool > 1 || compressLen) return undefined;
  return dsaIndexerSchedule(text, layers) ?? dsaIndexerSchedule(source, layers);
}
