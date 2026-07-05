const LAYER_KEYS = ["num_hidden_layers", "num_layers", "n_layer", "n_layers"];
const HIDDEN_KEYS = ["hidden_size", "dim", "d_model"];
const HEAD_KEYS = ["num_attention_heads", "n_heads", "attention_heads"];
const KV_HEAD_KEYS = ["num_key_value_heads", "n_kv_heads", "kv_heads"];
const EXPERT_KEYS = ["num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"];
const EXPERTS_PER_TOKEN_KEYS = ["num_experts_per_tok", "num_experts_per_token", "moe_top_k"];
const CONTEXT_KEYS = ["max_position_embeddings", "seq_length", "max_sequence_length"];

function firstNumber(config, keys) {
  for (const key of keys) {
    const value = config?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() && !Number.isNaN(Number(value))) return Number(value);
  }
  return undefined;
}

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

export function normalizeConfig(config) {
  const textConfig = typeof config?.text_config === "object" && config.text_config ? config.text_config : config;
  const visionConfig = typeof config?.vision_config === "object" && config.vision_config ? config.vision_config : null;
  const layers = firstNumber(textConfig, LAYER_KEYS) ?? firstNumber(config, LAYER_KEYS);
  const visionLayers = visionConfig ? firstNumber(visionConfig, LAYER_KEYS) : undefined;

  return {
    raw: config,
    architecture: Array.isArray(config?.architectures) ? config.architectures[0] : undefined,
    modelType: config?.model_type,
    textConfig,
    visionConfig,
    layers,
    visionLayers,
    hiddenSize: firstNumber(textConfig, HIDDEN_KEYS) ?? firstNumber(config, HIDDEN_KEYS),
    attentionHeads: firstNumber(textConfig, HEAD_KEYS) ?? firstNumber(config, HEAD_KEYS),
    kvHeads: firstNumber(textConfig, KV_HEAD_KEYS) ?? firstNumber(config, KV_HEAD_KEYS),
    experts: firstNumber(textConfig, EXPERT_KEYS) ?? firstNumber(config, EXPERT_KEYS),
    expertsPerToken: firstNumber(textConfig, EXPERTS_PER_TOKEN_KEYS) ?? firstNumber(config, EXPERTS_PER_TOKEN_KEYS),
    contextLength: firstNumber(textConfig, CONTEXT_KEYS) ?? firstNumber(config, CONTEXT_KEYS),
    layerSchedule: explicitLayerSchedule(textConfig, layers) ?? explicitLayerSchedule(config, layers),
    attentionSchedule: sparseAttentionSchedule(textConfig, layers),
  };
}
