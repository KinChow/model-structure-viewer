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

function quantizationEstimate(config, textConfig) {
  const quantization = config?.quantization_config || textConfig?.quantization_config;
  if (!quantization || typeof quantization !== "object") return {};
  const method = String(quantization.quant_method || quantization.format || "").toLowerCase();
  const group = Object.values(quantization.config_groups || {})[0];
  const weights = group?.weights || {};
  const bits = firstNumber(quantization, ["bits", "num_bits"])
    ?? firstNumber(weights, ["num_bits"])
    ?? (method.includes("fp8") || method === "fp8" ? 8 : undefined);
  if (!bits || bits <= 0) return { quantizationMethod: method || "configured", quantizationBytesPerParameter: undefined };
  return {
    quantizationMethod: method || "configured",
    quantizationBits: bits,
    // This is a model-wide estimate. Per-module exceptions and scale tensors
    // become exact only after safetensors metadata is available.
    quantizationBytesPerParameter: bits / 8,
  };
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

function visionPatchTokenCount(config) {
  const imageSize = firstNumber(config, ["image_size"]);
  const patchSize = firstNumber(config, ["patch_size"]);
  if (!imageSize || !patchSize) {
    const positionCount = firstNumber(config, ["num_position_embeddings"]);
    if (positionCount) return positionCount;
    const height = firstNumber(config, ["init_pos_emb_height"]);
    const width = firstNumber(config, ["init_pos_emb_width"]);
    if (height && width) return height * width;
    return undefined;
  }
  const patches = Math.floor(imageSize / patchSize);
  return patches * patches;
}

function visionMergeSize(config) {
  return firstNumber(config, ["spatial_merge_size"])
    ?? firstNumber(config?.img_token_compression_config, ["spatial_merge_size"])
    ?? (Array.isArray(config?.merge_kernel_size) ? firstNumber({ value: config.merge_kernel_size[0] }, ["value"]) : undefined)
    ?? 1;
}

function visionTokenCount(config) {
  const patchTokens = visionPatchTokenCount(config);
  const merge = visionMergeSize(config);
  return patchTokens ? Math.floor(patchTokens / (merge * merge)) : undefined;
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
  const nestedVisionConfig = typeof config?.vision_config === "object" && config.vision_config ? config.vision_config : null;
  const flatVisionConfig = firstNumber(config, ["vision_n_layers"]) != null
    ? {
      num_hidden_layers: firstNumber(config, ["vision_n_layers"]),
      hidden_size: firstNumber(config, ["vision_dim"]),
      num_attention_heads: firstNumber(config, ["vision_n_heads"]),
      intermediate_size: firstNumber(config, ["vision_inter_dim"]),
      patch_size: firstNumber(config, ["vision_patch_size"]),
    }
    : null;
  const visionConfig = nestedVisionConfig || flatVisionConfig;
  const hasVision = Boolean(visionConfig) && config?.language_model_only !== true;
  const linearAttentionConfig = typeof textConfig?.linear_attn_config === "object" && textConfig.linear_attn_config
    ? textConfig.linear_attn_config
    : null;
  // W0.5（refactor_plan）：收敛重复探测，行为与原
  // `firstNumber(textConfig, K) ?? … ?? firstNumber(config, K)` 逐字等价。
  // pick：textConfig 优先，可选中间源，最后回退顶层 config。
  const pick = (keys, middle) =>
    firstNumber(textConfig, keys)
      ?? (middle ? firstNumber(middle.source, middle.keys) : undefined)
      ?? firstNumber(config, keys);
  // sparse_attention_config 是子对象，两侧来源与 pick 不同，单独收敛。
  const pickSparse = (keys) =>
    firstNumber(textConfig?.sparse_attention_config, keys)
      ?? firstNumber(config?.sparse_attention_config, keys);
  // 主流探测（A 变体）：先顶层后 text_config。
  // 本文件有意保留两种语义不同的变体——
  //   B 变体（仅顶层）：visionInternalMerger / visionMlpGated；
  //   C 变体（A 再兜一层 textConfig）：linearAttentionMode 的 kimi 分支。
  // 它们在"顶层 model_type 与 text_config 不同"时结果不同，统一属功能决策，不在重构范围。
  const modelTypeProbe = String(config?.model_type || textConfig?.model_type || "");
  const layers = pick(LAYER_KEYS);
  const visionLayers = visionConfig ? firstNumber(visionConfig, [...LAYER_KEYS, "depth", "vt_num_hidden_layers"]) : undefined;
  const hiddenSize = pick(HIDDEN_KEYS);
  const attentionHeads = pick(HEAD_KEYS);
  const headDim = attentionHeadDim(textConfig) ?? attentionHeadDim(config) ?? derivedHeadDim(hiddenSize, attentionHeads);
  const quantization = quantizationEstimate(config, textConfig);

  return {
    raw: config,
    architecture: Array.isArray(config?.architectures) ? config.architectures[0] : undefined,
    modelType: config?.model_type,
    textConfig,
    visionConfig,
    hasVision,
    hasVisionProjector: Boolean(nestedVisionConfig),
    layers,
    visionLayers,
    hiddenSize,
    attentionHeads,
    kvHeads: pick(KV_HEAD_KEYS),
    headDim,
    kvLoraRank: pick(KV_LORA_RANK_KEYS),
    qLoraRank: pick(Q_LORA_RANK_KEYS),
    oLoraRank: pick(O_LORA_RANK_KEYS),
    oGroups: pick(O_GROUP_KEYS),
    numHashLayers: pick(NUM_HASH_LAYER_KEYS),
    compressRatios: Array.isArray(textConfig?.compress_ratios)
      ? textConfig.compress_ratios.map((value) => Number(value)).filter((value) => Number.isFinite(value))
      : Array.isArray(config?.compress_ratios)
        ? config.compress_ratios.map((value) => Number(value)).filter((value) => Number.isFinite(value))
        : [],
    linearKeyHeads: pick(LINEAR_KEY_HEADS_KEYS, { source: linearAttentionConfig, keys: ["num_heads"] }),
    linearValueHeads: pick(LINEAR_VALUE_HEADS_KEYS, { source: linearAttentionConfig, keys: ["num_heads"] }),
    linearKeyDim: pick(LINEAR_KEY_DIM_KEYS, { source: linearAttentionConfig, keys: ["head_dim"] }),
    linearValueDim: pick(LINEAR_VALUE_DIM_KEYS, { source: linearAttentionConfig, keys: ["head_dim"] }),
    linearConvKernelSize: pick(["linear_conv_kernel_dim", "linear_conv_kernel_size"], { source: linearAttentionConfig, keys: ["short_conv_kernel_size"] }),
    linearLowerBound: pick(["linear_lower_bound"], { source: linearAttentionConfig, keys: ["gate_lower_bound"] }),
    linearUseFullRankGate: Boolean(textConfig?.linear_attn_config?.use_full_rank_gate ?? config?.linear_attn_config?.use_full_rank_gate),
    indexerNHeads: pick(["index_n_heads", "indexer_n_heads", "index_heads"]),
    indexerKVHeads: pick(["indexer_kv_heads"]),
    indexerHeadDim: pick(["indexer_head_dim", "index_head_dim"]),
    indexerBudget: pick(["index_topk", "indexer_budget"]),
    indexerCompressRatio: pick(["indexer_compress_ratio"]),
    sparseIndexHeads: pickSparse(["sparse_num_index_heads"]),
    sparseIndexDim: pickSparse(["sparse_index_dim"]),
    sparseTopkBlocks: pickSparse(["sparse_topk_blocks"]),
    sparseBlockSize: pickSparse(["sparse_block_size"]),
    sparseInitBlock: pickSparse(["sparse_init_block"]),
    sparseLocalBlock: pickSparse(["sparse_local_block"]),
    sparseScoreType: textConfig?.sparse_attention_config?.sparse_score_type ?? config?.sparse_attention_config?.sparse_score_type,
    sparseDisableIndexValue: Array.isArray(textConfig?.sparse_attention_config?.sparse_disable_index_value)
      ? textConfig.sparse_attention_config.sparse_disable_index_value.map((value) => Boolean(value))
      : Array.isArray(config?.sparse_attention_config?.sparse_disable_index_value)
        ? config.sparse_attention_config.sparse_disable_index_value.map((value) => Boolean(value))
        : [],
    indexerSchedule: (modelTypeProbe.includes("deepseek_v32")
      || modelTypeProbe.includes("glm_moe_dsa"))
      ? dsaIndexerSchedule(textConfig, layers) ?? dsaIndexerSchedule(config, layers)
      : undefined,
    slidingWindow: pick(["sliding_window", "window_size"]),
    routedScalingFactor: pick(["routed_scaling_factor"]),
    swigluLimit: pick(["swiglu_limit"]),
    swigluAlpha: pick(["swiglu_alpha"]),
    swigluBeta: pick(["swiglu_beta"]),
    normTopkProb: textConfig?.norm_topk_prob ?? config?.norm_topk_prob,
    qkRopeHeadDim:
      pick(QK_ROPE_HEAD_DIM_KEYS),
    qkNopeHeadDim:
      pick(["qk_nope_head_dim"]),
    valueHeadDim: pick(VALUE_HEAD_DIM_KEYS) ?? headDim,
    intermediateSize: pick(INTERMEDIATE_KEYS),
    moeIntermediateSize: pick(MOE_INTERMEDIATE_KEYS),
    vocabSize: pick(VOCAB_KEYS),
    visionHiddenSize: visionConfig ? firstNumber(visionConfig, [...HIDDEN_KEYS, "vt_hidden_size"]) : undefined,
    visionOutputSize: visionConfig
      ? firstNumber(visionConfig, ["out_hidden_size", "vision_hidden_size"]) ?? firstNumber(visionConfig, ["vt_hidden_size", "mm_hidden_size"]) ?? firstNumber(visionConfig, HIDDEN_KEYS)
      : undefined,
    visionAttentionHeads: visionConfig ? firstNumber(visionConfig, ["num_heads", "num_attention_heads", "vt_num_attention_heads"]) : undefined,
    visionHeadDim: visionConfig
      ? firstNumber(visionConfig, ["head_dim", "attention_head_dim"])
        ?? derivedHeadDim(firstNumber(visionConfig, [...HIDDEN_KEYS, "vt_hidden_size"]), firstNumber(visionConfig, ["num_heads", "num_attention_heads", "vt_num_attention_heads"]))
      : undefined,
    visionIntermediateSize: visionConfig ? firstNumber(visionConfig, ["intermediate_size", "vt_intermediate_size"]) : undefined,
    visionPatchSize: visionConfig ? firstNumber(visionConfig, ["patch_size"]) : undefined,
    visionTemporalPatchSize: visionConfig ? firstNumber(visionConfig, ["temporal_patch_size"]) : undefined,
    visionChannels: visionConfig ? firstNumber(visionConfig, ["in_channels", "num_channels"]) : undefined,
    visionImageSize: visionConfig ? firstNumber(visionConfig, ["image_size"]) : undefined,
    visionSpatialMergeSize: visionConfig
      ? firstNumber(visionConfig, ["spatial_merge_size"])
        ?? firstNumber(visionConfig?.img_token_compression_config, ["spatial_merge_size"])
        ?? (Array.isArray(visionConfig?.merge_kernel_size) ? firstNumber({ value: visionConfig.merge_kernel_size[0] }, ["value"]) : undefined)
      : undefined,
    visionTokens: visionConfig ? visionTokenCount(visionConfig) : undefined,
    visionPatchTokens: visionConfig ? visionPatchTokenCount(visionConfig) : undefined,
    visionMergeSize: visionConfig ? visionMergeSize(visionConfig) : 1,
        // B 变体（仅顶层 model_type）：与 modelTypeProbe 语义不同，有意保留（W0.5）。
    visionInternalMerger: Boolean(visionConfig && ["qwen3_5", "qwen4_exp", "glm5_next"].some((kind) => String(config?.model_type || "").includes(kind))),
    visionMergerIntermediateSize: visionConfig ? firstNumber(visionConfig, ["projection_intermediate_size"]) : undefined,
    visionMlpGated: visionConfig
      ? String(visionConfig.hidden_act || "").toLowerCase().includes("silu")
        || String(config?.model_type || "").toLowerCase().includes("glm5_next")
      : false,
    ...quantization,
    experts: pick(EXPERT_KEYS),
    routedExpertHiddenSize: pick(["routed_expert_hidden_size"]),
    expertsPerToken: pick(EXPERTS_PER_TOKEN_KEYS),
    sharedExperts: pick(SHARED_EXPERT_KEYS)
      ?? (modelTypeProbe.includes("qwen3_5_moe")
        && firstNumber(textConfig, SHARED_EXPERT_INTERMEDIATE_KEYS) != null ? 1 : undefined),
    sharedExpertIntermediateSize: pick(SHARED_EXPERT_INTERMEDIATE_KEYS)
      ?? (["kimi", "deepseek_v4", "glm4_moe", "glm5_next"].some((kind) => modelTypeProbe.includes(kind))
        ? (firstNumber(textConfig, MOE_INTERMEDIATE_KEYS) || 0) * (firstNumber(textConfig, SHARED_EXPERT_KEYS) || 0)
        : undefined),
    sharedExpertsAreFused: modelTypeProbe.includes("kimi_k3"),
    sharedExpertGate: firstNumber(textConfig, SHARED_EXPERT_INTERMEDIATE_KEYS) != null
      && (textConfig?.output_gate_type != null || modelTypeProbe.includes("qwen3_5_moe")),
    attentionOutputGate: Boolean(textConfig?.attn_output_gate ?? config?.attn_output_gate),
    attentionBias: Boolean(textConfig?.attention_bias ?? config?.attention_bias),
    outputGateType: String(textConfig?.output_gate_type ?? config?.output_gate_type ?? "silu"),
    partialRotaryFactor: firstNumber(textConfig, ["partial_rotary_factor"])
      ?? firstNumber(textConfig?.rope_parameters, ["partial_rotary_factor"])
      ?? firstNumber(textConfig?.rope_scaling, ["partial_rotary_factor"]),
    rotaryDim: pick(["rotary_dim"]),
    useQkNorm: Boolean(textConfig?.use_qk_norm ?? config?.use_qk_norm),
    qkNormType: textConfig?.qk_norm_type ?? config?.qk_norm_type,
    normMode: ["qwen3_5", "minimax_m3"].some((kind) => modelTypeProbe.includes(kind))
      || Boolean(textConfig?.use_gemma_norm ?? config?.use_gemma_norm)
      ? "gemma_rmsnorm"
      : "rmsnorm",
    hyperConnectionCount: pick(["hc_count"]),
    hyperConnectionLowrank: pick(["hc_lowrank"]),
    pleLayerIds: Array.isArray(textConfig?.ple_layer_ids) ? textConfig.ple_layer_ids : Array.isArray(config?.ple_layer_ids) ? config.ple_layer_ids : [],
    pleEmbedDim: pick(["ple_embed_dim"]),
    pleNgramSize: pick(["ngram_size"]),
    pleHeadsPerNgram: pick(["heads_per_ngram"]),
    pleConvKernelSize: pick(["ple_conv_kernel_size"]),
    attnResBlockSize: pick(["attn_res_block_size"]),
    mlaUseOutputGate: Boolean(textConfig?.mla_use_output_gate ?? config?.mla_use_output_gate),
    linearAttentionMode: modelTypeProbe.includes("kimi_k3")
      ? "kimi_k3"
            // C 变体：在 A 之外多兜一层 textConfig，与纯 probe 语义不同，有意保留（W0.5）。
    : modelTypeProbe.includes("kimi") || String(textConfig?.model_type || "").includes("kimi")
        ? "kimi"
      : modelTypeProbe.includes("qwen4_exp")
        ? "qwen4_exp"
        : modelTypeProbe.includes("qwen3_5")
          ? "qwen3_5"
        : modelTypeProbe.includes("glm5_next")
          ? "glm5_next"
        : "generic",
    multiHyperConnection: Boolean(
      textConfig?.mhc
      ?? config?.mhc
      ?? (modelTypeProbe.includes("deepseek_v4")
        && firstNumber(textConfig, ["hc_mult"]) != null),
    ),
    mhcNumResidualStreams: pick(["mhc_num_residual_streams", "hc_mult"]),
    mhcSinkhornIterations: pick(["mhc_sinkhorn_iterations", "hc_sinkhorn_iters"]),
    mhcTau: pick(["mhc_tau"]),
    mhcEps: pick(["hc_eps", "mhc_eps"]),
    mhcPostMultValue: pick(["mhc_post_mult_value"])
      ?? (modelTypeProbe.includes("deepseek_v4") ? 2 : undefined),
    contextLength: pick(CONTEXT_KEYS),
    tieWordEmbeddings: textConfig?.tie_word_embeddings ?? config?.tie_word_embeddings ?? false,
    layerSchedule: explicitLayerSchedule(textConfig, layers) ?? explicitLayerSchedule(config, layers),
    attentionSchedule:
      explicitAttentionSchedule(textConfig, layers)
      ?? explicitAttentionSchedule(config, layers)
      ?? sparseAttentionSchedule(textConfig, layers),
  };
}
