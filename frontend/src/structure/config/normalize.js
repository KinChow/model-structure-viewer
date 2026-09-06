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
const SHARED_EXPERT_INTERMEDIATE_KEYS = ["shared_expert_intermediate_size", "shared_expert_hidden_size", "shared_intermediate_size"];
const CONTEXT_KEYS = ["max_position_embeddings", "seq_length", "max_sequence_length"];
const KV_LORA_RANK_KEYS = ["kv_lora_rank", "kv_lora_dim"];
const Q_LORA_RANK_KEYS = ["q_lora_rank", "q_lora_dim"];
const O_LORA_RANK_KEYS = ["o_lora_rank", "o_lora_dim"];
const O_GROUP_KEYS = ["o_groups", "output_groups"];
const NUM_HASH_LAYER_KEYS = ["num_hash_layers", "hash_moe_layers"];
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
  // GLM-5.3-Flash publishes head_dim=0 as a sentinel.
  if (explicit !== undefined && explicit > 0) return explicit;
  const qkNope = firstNumber(config, ["qk_nope_head_dim"]);
  const qkRope = firstNumber(config, ["qk_rope_head_dim"]);
  if (qkNope !== undefined && qkRope !== undefined && qkNope + qkRope > 0) return qkNope + qkRope;
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
  const modelType = String(config?.model_type || "").toLowerCase();
  if (modelType === "deepseek_v4" && Array.isArray(config?.compress_ratios) && config.compress_ratios.length > 0) {
    return Array.from({ length: layers || config.compress_ratios.length }, () => "dsv4");
  }
  if ((modelType === "deepseek_v32" || modelType === "glm_moe_dsa") && firstNumber(config, ["index_topk"]) != null) {
    return Array.from({ length: layers || 0 }, () => "qsa");
  }
  const isQwen35 = modelType.includes("qwen3_5");
  const layerTypes = config?.layer_types;
  const useQsa = firstNumber(config, ["index_n_heads", "indexer_n_heads"]) != null
    || firstNumber(config, ["index_topk", "indexer_budget"]) != null;
  if (Array.isArray(layerTypes) && layerTypes.length > 0) {
    return layerTypes.map((layerType) => {
      const kind = attentionKindForLayerType(layerType, useQsa);
      return isQwen35 && kind === "gqa" && String(layerType).toLowerCase().includes("full")
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
    oLoraRank: firstNumber(textConfig, O_LORA_RANK_KEYS) ?? firstNumber(config, O_LORA_RANK_KEYS),
    oGroups: firstNumber(textConfig, O_GROUP_KEYS) ?? firstNumber(config, O_GROUP_KEYS),
    numHashLayers: firstNumber(textConfig, NUM_HASH_LAYER_KEYS) ?? firstNumber(config, NUM_HASH_LAYER_KEYS),
    compressRatios: Array.isArray(textConfig?.compress_ratios)
      ? textConfig.compress_ratios.map((value) => Number(value)).filter((value) => Number.isFinite(value))
      : Array.isArray(config?.compress_ratios)
        ? config.compress_ratios.map((value) => Number(value)).filter((value) => Number.isFinite(value))
        : [],
    linearKeyHeads: firstNumber(textConfig, LINEAR_KEY_HEADS_KEYS) ?? firstNumber(linearAttentionConfig, ["num_heads"]) ?? firstNumber(config, LINEAR_KEY_HEADS_KEYS),
    linearValueHeads: firstNumber(textConfig, LINEAR_VALUE_HEADS_KEYS) ?? firstNumber(linearAttentionConfig, ["num_heads"]) ?? firstNumber(config, LINEAR_VALUE_HEADS_KEYS),
    linearKeyDim: firstNumber(textConfig, LINEAR_KEY_DIM_KEYS) ?? firstNumber(linearAttentionConfig, ["head_dim"]) ?? firstNumber(config, LINEAR_KEY_DIM_KEYS),
    linearValueDim: firstNumber(textConfig, LINEAR_VALUE_DIM_KEYS) ?? firstNumber(linearAttentionConfig, ["head_dim"]) ?? firstNumber(config, LINEAR_VALUE_DIM_KEYS),
    linearConvKernelSize: firstNumber(textConfig, ["linear_conv_kernel_dim", "linear_conv_kernel_size"]) ?? firstNumber(linearAttentionConfig, ["short_conv_kernel_size"]) ?? firstNumber(config, ["linear_conv_kernel_dim", "linear_conv_kernel_size"]),
    linearLowerBound: firstNumber(textConfig, ["linear_lower_bound"]) ?? firstNumber(linearAttentionConfig, ["gate_lower_bound"]) ?? firstNumber(config, ["linear_lower_bound"]),
    linearUseFullRankGate: Boolean(textConfig?.linear_attn_config?.use_full_rank_gate ?? config?.linear_attn_config?.use_full_rank_gate),
    indexerNHeads: firstNumber(textConfig, ["index_n_heads", "indexer_n_heads", "index_heads"]) ?? firstNumber(config, ["index_n_heads", "indexer_n_heads", "index_heads"]),
    indexerKVHeads: firstNumber(textConfig, ["indexer_kv_heads"]) ?? firstNumber(config, ["indexer_kv_heads"]),
    indexerHeadDim: firstNumber(textConfig, ["indexer_head_dim", "index_head_dim"]) ?? firstNumber(config, ["indexer_head_dim", "index_head_dim"]),
    indexerBudget: firstNumber(textConfig, ["index_topk", "indexer_budget"]) ?? firstNumber(config, ["index_topk", "indexer_budget"]),
    indexerCompressRatio: firstNumber(textConfig, ["indexer_compress_ratio"]) ?? firstNumber(config, ["indexer_compress_ratio"]),
    sparseIndexHeads: firstNumber(textConfig?.sparse_attention_config, ["sparse_num_index_heads"]) ?? firstNumber(config?.sparse_attention_config, ["sparse_num_index_heads"]),
    sparseIndexDim: firstNumber(textConfig?.sparse_attention_config, ["sparse_index_dim"]) ?? firstNumber(config?.sparse_attention_config, ["sparse_index_dim"]),
    sparseTopkBlocks: firstNumber(textConfig?.sparse_attention_config, ["sparse_topk_blocks"]) ?? firstNumber(config?.sparse_attention_config, ["sparse_topk_blocks"]),
    sparseBlockSize: firstNumber(textConfig?.sparse_attention_config, ["sparse_block_size"]) ?? firstNumber(config?.sparse_attention_config, ["sparse_block_size"]),
    sparseInitBlock: firstNumber(textConfig?.sparse_attention_config, ["sparse_init_block"]) ?? firstNumber(config?.sparse_attention_config, ["sparse_init_block"]),
    sparseLocalBlock: firstNumber(textConfig?.sparse_attention_config, ["sparse_local_block"]) ?? firstNumber(config?.sparse_attention_config, ["sparse_local_block"]),
    sparseScoreType: textConfig?.sparse_attention_config?.sparse_score_type ?? config?.sparse_attention_config?.sparse_score_type,
    sparseDisableIndexValue: Array.isArray(textConfig?.sparse_attention_config?.sparse_disable_index_value)
      ? textConfig.sparse_attention_config.sparse_disable_index_value.map((value) => Boolean(value))
      : Array.isArray(config?.sparse_attention_config?.sparse_disable_index_value)
        ? config.sparse_attention_config.sparse_disable_index_value.map((value) => Boolean(value))
        : [],
    indexerSchedule: (String(config?.model_type || textConfig?.model_type || "").includes("deepseek_v32")
      || String(config?.model_type || textConfig?.model_type || "").includes("glm_moe_dsa"))
      ? dsaIndexerSchedule(textConfig, layers) ?? dsaIndexerSchedule(config, layers)
      : undefined,
    slidingWindow: firstNumber(textConfig, ["sliding_window", "window_size"]) ?? firstNumber(config, ["sliding_window", "window_size"]),
    routedScalingFactor: firstNumber(textConfig, ["routed_scaling_factor"]) ?? firstNumber(config, ["routed_scaling_factor"]),
    swigluLimit: firstNumber(textConfig, ["swiglu_limit"]) ?? firstNumber(config, ["swiglu_limit"]),
    swigluAlpha: firstNumber(textConfig, ["swiglu_alpha"]) ?? firstNumber(config, ["swiglu_alpha"]),
    swigluBeta: firstNumber(textConfig, ["swiglu_beta"]) ?? firstNumber(config, ["swiglu_beta"]),
    normTopkProb: textConfig?.norm_topk_prob ?? config?.norm_topk_prob,
    qkRopeHeadDim:
      firstNumber(textConfig, QK_ROPE_HEAD_DIM_KEYS) ?? firstNumber(config, QK_ROPE_HEAD_DIM_KEYS),
    qkNopeHeadDim:
      firstNumber(textConfig, ["qk_nope_head_dim"]) ?? firstNumber(config, ["qk_nope_head_dim"]),
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
    sharedExperts: firstNumber(textConfig, SHARED_EXPERT_KEYS) ?? firstNumber(config, SHARED_EXPERT_KEYS)
      ?? (String(config?.model_type || textConfig?.model_type || "").includes("qwen3_5_moe")
        && firstNumber(textConfig, SHARED_EXPERT_INTERMEDIATE_KEYS) != null ? 1 : undefined),
    sharedExpertIntermediateSize: firstNumber(textConfig, SHARED_EXPERT_INTERMEDIATE_KEYS) ?? firstNumber(config, SHARED_EXPERT_INTERMEDIATE_KEYS)
      ?? (["kimi_k3", "deepseek_v4"].some((kind) => String(config?.model_type || textConfig?.model_type || "").includes(kind))
        ? (firstNumber(textConfig, MOE_INTERMEDIATE_KEYS) || 0) * (firstNumber(textConfig, SHARED_EXPERT_KEYS) || 0)
        : undefined),
    sharedExpertsAreFused: String(config?.model_type || textConfig?.model_type || "").includes("kimi_k3"),
    sharedExpertGate: firstNumber(textConfig, SHARED_EXPERT_INTERMEDIATE_KEYS) != null
      && (textConfig?.output_gate_type != null || String(config?.model_type || textConfig?.model_type || "").includes("qwen3_5_moe")),
    attentionOutputGate: Boolean(textConfig?.attn_output_gate ?? config?.attn_output_gate),
    outputGateType: String(textConfig?.output_gate_type ?? config?.output_gate_type ?? "silu"),
    partialRotaryFactor: firstNumber(textConfig, ["partial_rotary_factor"])
      ?? firstNumber(textConfig?.rope_parameters, ["partial_rotary_factor"])
      ?? firstNumber(textConfig?.rope_scaling, ["partial_rotary_factor"]),
    rotaryDim: firstNumber(textConfig, ["rotary_dim"]) ?? firstNumber(config, ["rotary_dim"]),
    useQkNorm: Boolean(textConfig?.use_qk_norm ?? config?.use_qk_norm),
    qkNormType: textConfig?.qk_norm_type ?? config?.qk_norm_type,
    normMode: ["qwen3_5", "minimax_m3"].some((kind) => String(config?.model_type || textConfig?.model_type || "").includes(kind))
      || Boolean(textConfig?.use_gemma_norm ?? config?.use_gemma_norm)
      ? "gemma_rmsnorm"
      : "rmsnorm",
    hyperConnectionCount: firstNumber(textConfig, ["hc_count"]) ?? firstNumber(config, ["hc_count"]),
    hyperConnectionLowrank: firstNumber(textConfig, ["hc_lowrank"]) ?? firstNumber(config, ["hc_lowrank"]),
    pleLayerIds: Array.isArray(textConfig?.ple_layer_ids) ? textConfig.ple_layer_ids : Array.isArray(config?.ple_layer_ids) ? config.ple_layer_ids : [],
    pleEmbedDim: firstNumber(textConfig, ["ple_embed_dim"]) ?? firstNumber(config, ["ple_embed_dim"]),
    pleNgramSize: firstNumber(textConfig, ["ngram_size"]) ?? firstNumber(config, ["ngram_size"]),
    pleHeadsPerNgram: firstNumber(textConfig, ["heads_per_ngram"]) ?? firstNumber(config, ["heads_per_ngram"]),
    pleConvKernelSize: firstNumber(textConfig, ["ple_conv_kernel_size"]) ?? firstNumber(config, ["ple_conv_kernel_size"]),
    attnResBlockSize: firstNumber(textConfig, ["attn_res_block_size"]) ?? firstNumber(config, ["attn_res_block_size"]),
    mlaUseOutputGate: Boolean(textConfig?.mla_use_output_gate ?? config?.mla_use_output_gate),
    linearAttentionMode: String(config?.model_type || textConfig?.model_type || "").includes("kimi_k3")
      ? "kimi_k3"
      : String(config?.model_type || textConfig?.model_type || "").includes("kimi") || String(textConfig?.model_type || "").includes("kimi")
        ? "kimi"
      : String(config?.model_type || textConfig?.model_type || "").includes("qwen4_exp")
        ? "qwen4_exp"
        : String(config?.model_type || textConfig?.model_type || "").includes("qwen3_5")
          ? "qwen3_5"
        : String(config?.model_type || textConfig?.model_type || "").includes("glm5_next")
          ? "glm5_next"
        : "generic",
    multiHyperConnection: Boolean(
      textConfig?.mhc
      ?? config?.mhc
      ?? (String(config?.model_type || textConfig?.model_type || "").includes("deepseek_v4")
        && firstNumber(textConfig, ["hc_mult"]) != null),
    ),
    mhcNumResidualStreams: firstNumber(textConfig, ["mhc_num_residual_streams", "hc_mult"]) ?? firstNumber(config, ["mhc_num_residual_streams", "hc_mult"]),
    mhcSinkhornIterations: firstNumber(textConfig, ["mhc_sinkhorn_iterations", "hc_sinkhorn_iters"]) ?? firstNumber(config, ["mhc_sinkhorn_iterations", "hc_sinkhorn_iters"]),
    mhcTau: firstNumber(textConfig, ["mhc_tau"]) ?? firstNumber(config, ["mhc_tau"]),
    mhcEps: firstNumber(textConfig, ["hc_eps", "mhc_eps"]) ?? firstNumber(config, ["hc_eps", "mhc_eps"]),
    mhcPostMultValue: firstNumber(textConfig, ["mhc_post_mult_value"]) ?? firstNumber(config, ["mhc_post_mult_value"])
      ?? (String(config?.model_type || textConfig?.model_type || "").includes("deepseek_v4") ? 2 : undefined),
    contextLength: firstNumber(textConfig, CONTEXT_KEYS) ?? firstNumber(config, CONTEXT_KEYS),
    tieWordEmbeddings: textConfig?.tie_word_embeddings ?? config?.tie_word_embeddings ?? false,
    layerSchedule: explicitLayerSchedule(textConfig, layers) ?? explicitLayerSchedule(config, layers),
    attentionSchedule:
      explicitAttentionSchedule(textConfig, layers)
      ?? explicitAttentionSchedule(config, layers)
      ?? sparseAttentionSchedule(textConfig, layers),
  };
}
