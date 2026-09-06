const LAYER_KEYS = ["num_hidden_layers", "num_layers", "n_layer", "n_layers"];
const HIDDEN_KEYS = ["hidden_size", "dim", "d_model"];
const HEAD_KEYS = ["num_attention_heads", "n_heads", "attention_heads"];
const KV_HEAD_KEYS = ["num_key_value_heads", "n_kv_heads", "kv_heads"];
const HEAD_DIM_KEYS = ["head_dim", "attention_head_dim"];
const VALUE_HEAD_DIM_KEYS = ["v_head_dim", "value_head_dim"];
const INTERMEDIATE_KEYS = ["intermediate_size", "ffn_hidden_size"];
const MOE_INTERMEDIATE_KEYS = ["moe_intermediate_size", "expert_intermediate_size"];
const VOCAB_KEYS = ["vocab_size"];
const EXPERT_KEYS = ["num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"];
const EXPERTS_PER_TOKEN_KEYS = ["num_experts_per_tok", "num_experts_per_token", "moe_top_k"];
const SHARED_EXPERT_KEYS = ["num_shared_experts", "n_shared_experts"];
const SHARED_EXPERT_INTERMEDIATE_KEYS = ["shared_expert_intermediate_size", "shared_expert_hidden_size"];
const CONTEXT_KEYS = ["max_position_embeddings", "seq_length", "max_sequence_length"];
const KV_LORA_RANK_KEYS = ["kv_lora_rank", "kv_lora_dim"];
const Q_LORA_RANK_KEYS = ["q_lora_rank", "q_lora_dim"];
const QK_ROPE_HEAD_DIM_KEYS = ["qk_rope_head_dim", "rope_head_dim"];
const LINEAR_KEY_HEADS_KEYS = ["linear_num_key_heads", "linear_key_heads"];
const LINEAR_VALUE_HEADS_KEYS = ["linear_num_value_heads", "linear_value_heads"];
const LINEAR_KEY_DIM_KEYS = ["linear_key_head_dim", "linear_head_dim"];
const LINEAR_VALUE_DIM_KEYS = ["linear_value_head_dim", "linear_head_dim"];

function firstNumber(config, keys) {
  for (const key of keys) {
    const value = config?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() && !Number.isNaN(Number(value))) return Number(value);
  }
  return undefined;
}

function attentionHeadDim(config) {
  const explicit = firstNumber(config, HEAD_DIM_KEYS);
  if (explicit !== undefined) return explicit;
  const qkNope = firstNumber(config, ["qk_nope_head_dim"]);
  const qkRope = firstNumber(config, ["qk_rope_head_dim"]);
  if (qkNope !== undefined && qkRope !== undefined) return qkNope + qkRope;
  return undefined;
}

function derivedHeadDim(hiddenSize, attentionHeads) {
  if (!hiddenSize || !attentionHeads) return undefined;
  return hiddenSize / attentionHeads;
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

function attentionKindForLayerType(layerType, useQsa = false) {
  const kind = String(layerType || "").toLowerCase();
  if (kind.includes("linear") || kind.includes("kda") || kind.includes("delta")) return "linear";
  if (kind.includes("deepseek") || kind.includes("mla") || kind.includes("sparse")) return "mla";
  return kind.includes("full") && useQsa ? "qsa" : "gqa";
}

function explicitAttentionSchedule(config, layers) {
  const layerTypes = config?.layer_types;
  const useQsa = firstNumber(config, ["indexer_n_heads"]) != null;
  if (Array.isArray(layerTypes) && layerTypes.length > 0) {
    return layerTypes.map((layerType) => attentionKindForLayerType(layerType, useQsa));
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

export function normalizeConfig(config) {
  const textConfig = typeof config?.text_config === "object" && config.text_config ? config.text_config : config;
  const visionConfig = typeof config?.vision_config === "object" && config.vision_config ? config.vision_config : null;
  const linearAttentionConfig = typeof textConfig?.linear_attn_config === "object" && textConfig.linear_attn_config
    ? textConfig.linear_attn_config
    : null;
  const layers = firstNumber(textConfig, LAYER_KEYS) ?? firstNumber(config, LAYER_KEYS);
  const visionLayers = visionConfig ? firstNumber(visionConfig, [...LAYER_KEYS, "depth", "vt_num_hidden_layers"]) : undefined;
  const hiddenSize = firstNumber(textConfig, HIDDEN_KEYS) ?? firstNumber(config, HIDDEN_KEYS);
  const attentionHeads = firstNumber(textConfig, HEAD_KEYS) ?? firstNumber(config, HEAD_KEYS);
  const headDim = attentionHeadDim(textConfig) ?? attentionHeadDim(config) ?? derivedHeadDim(hiddenSize, attentionHeads);

  return {
    raw: config,
    architecture: Array.isArray(config?.architectures) ? config.architectures[0] : undefined,
    modelType: config?.model_type,
    textConfig,
    visionConfig,
    layers,
    visionLayers,
    hiddenSize,
    attentionHeads,
    kvHeads: firstNumber(textConfig, KV_HEAD_KEYS) ?? firstNumber(config, KV_HEAD_KEYS),
    headDim,
    kvLoraRank: firstNumber(textConfig, KV_LORA_RANK_KEYS) ?? firstNumber(config, KV_LORA_RANK_KEYS),
    qLoraRank: firstNumber(textConfig, Q_LORA_RANK_KEYS) ?? firstNumber(config, Q_LORA_RANK_KEYS),
    linearKeyHeads: firstNumber(textConfig, LINEAR_KEY_HEADS_KEYS) ?? firstNumber(linearAttentionConfig, ["num_heads"]) ?? firstNumber(config, LINEAR_KEY_HEADS_KEYS),
    linearValueHeads: firstNumber(textConfig, LINEAR_VALUE_HEADS_KEYS) ?? firstNumber(linearAttentionConfig, ["num_heads"]) ?? firstNumber(config, LINEAR_VALUE_HEADS_KEYS),
    linearKeyDim: firstNumber(textConfig, LINEAR_KEY_DIM_KEYS) ?? firstNumber(linearAttentionConfig, ["head_dim"]) ?? firstNumber(config, LINEAR_KEY_DIM_KEYS),
    linearValueDim: firstNumber(textConfig, LINEAR_VALUE_DIM_KEYS) ?? firstNumber(linearAttentionConfig, ["head_dim"]) ?? firstNumber(config, LINEAR_VALUE_DIM_KEYS),
    indexerNHeads: firstNumber(textConfig, ["indexer_n_heads"]) ?? firstNumber(config, ["indexer_n_heads"]),
    indexerKVHeads: firstNumber(textConfig, ["indexer_kv_heads"]) ?? firstNumber(config, ["indexer_kv_heads"]),
    indexerHeadDim: firstNumber(textConfig, ["indexer_head_dim"]) ?? firstNumber(config, ["indexer_head_dim"]),
    indexerBudget: firstNumber(textConfig, ["indexer_budget"]) ?? firstNumber(config, ["indexer_budget"]),
    indexerCompressRatio: firstNumber(textConfig, ["indexer_compress_ratio"]) ?? firstNumber(config, ["indexer_compress_ratio"]),
    qkRopeHeadDim:
      firstNumber(textConfig, QK_ROPE_HEAD_DIM_KEYS) ?? firstNumber(config, QK_ROPE_HEAD_DIM_KEYS),
    valueHeadDim: firstNumber(textConfig, VALUE_HEAD_DIM_KEYS) ?? firstNumber(config, VALUE_HEAD_DIM_KEYS) ?? headDim,
    intermediateSize: firstNumber(textConfig, INTERMEDIATE_KEYS) ?? firstNumber(config, INTERMEDIATE_KEYS),
    moeIntermediateSize: firstNumber(textConfig, MOE_INTERMEDIATE_KEYS) ?? firstNumber(config, MOE_INTERMEDIATE_KEYS),
    vocabSize: firstNumber(textConfig, VOCAB_KEYS) ?? firstNumber(config, VOCAB_KEYS),
    visionHiddenSize: visionConfig ? firstNumber(visionConfig, [...HIDDEN_KEYS, "vt_hidden_size"]) : undefined,
    visionOutputSize: visionConfig
      ? firstNumber(visionConfig, ["out_hidden_size", "vision_hidden_size"]) ?? firstNumber(visionConfig, ["vt_hidden_size", "mm_hidden_size"]) ?? firstNumber(visionConfig, HIDDEN_KEYS)
      : undefined,
    experts: firstNumber(textConfig, EXPERT_KEYS) ?? firstNumber(config, EXPERT_KEYS),
    routedExpertHiddenSize: firstNumber(textConfig, ["routed_expert_hidden_size"]) ?? firstNumber(config, ["routed_expert_hidden_size"]),
    expertsPerToken: firstNumber(textConfig, EXPERTS_PER_TOKEN_KEYS) ?? firstNumber(config, EXPERTS_PER_TOKEN_KEYS),
    sharedExperts: firstNumber(textConfig, SHARED_EXPERT_KEYS) ?? firstNumber(config, SHARED_EXPERT_KEYS),
    sharedExpertIntermediateSize: firstNumber(textConfig, SHARED_EXPERT_INTERMEDIATE_KEYS) ?? firstNumber(config, SHARED_EXPERT_INTERMEDIATE_KEYS),
    sharedExpertGate: firstNumber(textConfig, SHARED_EXPERT_INTERMEDIATE_KEYS) != null && textConfig?.output_gate_type != null,
    hyperConnectionCount: firstNumber(textConfig, ["hc_count"]) ?? firstNumber(config, ["hc_count"]),
    hyperConnectionLowrank: firstNumber(textConfig, ["hc_lowrank"]) ?? firstNumber(config, ["hc_lowrank"]),
    pleLayerIds: Array.isArray(textConfig?.ple_layer_ids) ? textConfig.ple_layer_ids : Array.isArray(config?.ple_layer_ids) ? config.ple_layer_ids : [],
    pleEmbedDim: firstNumber(textConfig, ["ple_embed_dim"]) ?? firstNumber(config, ["ple_embed_dim"]),
    attnResBlockSize: firstNumber(textConfig, ["attn_res_block_size"]) ?? firstNumber(config, ["attn_res_block_size"]),
    mlaUseOutputGate: Boolean(textConfig?.mla_use_output_gate ?? config?.mla_use_output_gate),
    linearAttentionMode: String(config?.model_type || textConfig?.model_type || "").includes("kimi") || String(textConfig?.model_type || "").includes("kimi")
      ? "kimi"
      : String(config?.model_type || textConfig?.model_type || "").includes("qwen4_exp")
        ? "qwen4_exp"
        : "generic",
    contextLength: firstNumber(textConfig, CONTEXT_KEYS) ?? firstNumber(config, CONTEXT_KEYS),
    tieWordEmbeddings: textConfig?.tie_word_embeddings ?? config?.tie_word_embeddings ?? false,
    layerSchedule: explicitLayerSchedule(textConfig, layers) ?? explicitLayerSchedule(config, layers),
    attentionSchedule:
      explicitAttentionSchedule(textConfig, layers)
      ?? explicitAttentionSchedule(config, layers)
      ?? sparseAttentionSchedule(textConfig, layers),
  };
}
