// normalize.js —— config 字段归一层（§4.7：只供数，方案决定权归组网）。
//
// 唯一的例外是 sharedExpertIntermediateSize 的**回退宽度语义**：字段缺失时，
// "模块宽 vs 单专家宽"取决于 checkpoint 是否把多个 shared expert 打包成单张量
// （fused），这是家族知识而非字段判据 —— 判定权归 archs 配方（P3 单源化），
// 本文件只消费该布尔值。archs/ 与 config/ 同为结构栈最底层（见
// __tests__/layering.test.js）。
import { archRecipe } from "../archs/index.js";

export const LAYER_KEYS = ["num_hidden_layers", "num_layers", "n_layer", "n_layers"];
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
// Qwen 系 gate / 个数回退只认这两键。MiniMax 的 shared_intermediate_size 是宽度
// （0 = 无 shared expert），不能拿来当 gate。vLLM qwen3_moe.py:178：`> 0`。
const QWEN_SHARED_EXPERT_WIDTH_KEYS = ["shared_expert_intermediate_size", "shared_expert_hidden_size"];
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

export function firstNumber(config, keys) {
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
  // 扁平 vision 配置（DeepSeek V4 Flash Vision 的 vision_* 顶层字段）没有
  // image_size，无法从 patch 网格推 token 数，但直接给了送进 LLM 的**上限**
  // vision_max_n_token（已过 downsample，不再除 merge²）。优先用它。
  const declared = firstNumber(config, ["max_n_token"]);
  if (declared) return declared;
  const patchTokens = visionPatchTokenCount(config);
  const merge = visionMergeSize(config);
  return patchTokens ? Math.floor(patchTokens / (merge * merge)) : undefined;
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
      // downsample_ratio 就是空间合并因子；max_n_token 是合并后送进 LLM 的
      // 视觉 token 上限 —— 两者都只在扁平配置里出现。
      spatial_merge_size: firstNumber(config, ["vision_downsample_ratio"]),
      max_n_token: firstNumber(config, ["vision_max_n_token"]),
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
  // fused shared expert（多个 shared expert 打包为单个 gate/up/down 张量）决定
  // sharedExpertIntermediateSize 回退语义是"模块宽"而非"单专家宽"。
  // 判定权归 archs 配方（§4.7 —— normalize 只归一字段；
  // ARCH_RECIPES.sharedExpertsAreFused 是写不出字段判据的家族知识）。
  const sharedExpertsFused = Boolean(archRecipe(
    Array.isArray(config?.architectures) ? config.architectures[0] : undefined,
  ).sharedExpertsAreFused);
  const layers = pick(LAYER_KEYS);
  const visionLayers = visionConfig ? firstNumber(visionConfig, [...LAYER_KEYS, "depth", "vt_num_hidden_layers"]) : undefined;
  const hiddenSize = pick(HIDDEN_KEYS);
  const attentionHeads = pick(HEAD_KEYS);
  const headDim = attentionHeadDim(textConfig) ?? attentionHeadDim(config) ?? derivedHeadDim(hiddenSize, attentionHeads);
  const quantization = quantizationEstimate(config, textConfig);
  // MHC 开：显式 mhc，或流数字段存在（V4 只发 hc_mult，GLM-Flash 发 mhc+hc_mult）。
  // vLLM glm5_next 把 hc_mult 别名到 mhc_num_residual_streams；V4 modeling 读 hc_mult。
  const mhcStreams = pick(["mhc_num_residual_streams", "hc_mult"]);
  const mhcOn = Boolean(textConfig?.mhc ?? config?.mhc) || mhcStreams != null;
  const qwenSharedExpertWidth = pick(QWEN_SHARED_EXPERT_WIDTH_KEYS);

  return {
    raw: config,
    architecture: Array.isArray(config?.architectures) ? config.architectures[0] : undefined,
    modelType: config?.model_type,
    textConfig,
    visionConfig,
    hasVision,
    // 只要视觉塔输出宽 ≠ 文本 hidden，就**必然**有一层视觉→文本投影；扁平
    // vision 配置（顶层 vision_*，如 DeepSeek V4 Flash Vision）没有嵌套
    // vision_config，此前一律判成「无投影器」，结构树里整层缺失（权重字节
    // 恒等式因此差 visionOutput·hidden = 4,194,304，2026-09-09 逐层归因抓出）。
    hasVisionProjector: Boolean(nestedVisionConfig)
      || (hasVision && (visionConfig?.hidden_size ?? 0) !== hiddenSize),
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
    scoringFunc: textConfig?.scoring_func ?? config?.scoring_func,
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
    // 稀疏选择分支的字段分族（W2）。此前 index_* 与 indexer_* 被合并成同一组
    // indexer* 字段，导致 DSA（DeepSeek/GLM，index_* 键族）与 QSA（Qwen
    // qwen4_exp，indexer_* 键族）在下游无法区分——四种 indexer 原理不同却共用
    // 一个 operator_id 的根因就在这里。分族后判据变成纯字段存在性：
    //   index_topk + kv_lora_rank        -> DSA over MLA
    //   + index_kpool > 1                -> DSA k-pool 变体（glm5_next）
    //   indexer_budget + indexer_kv_heads-> QSA（qwen4_exp）
    //   sparse_attention_config.*        -> MiniMax 块稀疏（下方 sparse* 字段）
    dsaIndexHeads: pick(["index_n_heads", "index_heads"]),
    dsaIndexHeadDim: pick(["index_head_dim"]),
    dsaIndexTopk: pick(["index_topk"]),
    dsaIndexKpool: pick(["index_kpool"]),
    dsaIndexKpoolSelectTail: Boolean(textConfig?.index_kpool_always_select_tail ?? config?.index_kpool_always_select_tail),
    qsaIndexerHeads: pick(["indexer_n_heads"]),
    qsaIndexerKVHeads: pick(["indexer_kv_heads"]),
    qsaIndexerHeadDim: pick(["indexer_head_dim"]),
    qsaIndexerBudget: pick(["indexer_budget"]),
    qsaIndexerCompressRatio: pick(["indexer_compress_ratio"]),
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
    denseIntermediateSize: pick(["dense_intermediate_size"]),
    moeIntermediateSize: pick(MOE_INTERMEDIATE_KEYS),
    vocabSize: pick(VOCAB_KEYS),
    visionHiddenSize: visionConfig ? firstNumber(visionConfig, [...HIDDEN_KEYS, "vt_hidden_size"]) : undefined,
    visionOutputSize: visionConfig
      ? firstNumber(visionConfig, ["out_hidden_size", "vision_hidden_size"]) ?? firstNumber(visionConfig, ["vt_hidden_size", "mm_hidden_size"]) ?? firstNumber(visionConfig, HIDDEN_KEYS)
      : undefined,
    visionAttentionHeads: visionConfig ? firstNumber(visionConfig, ["num_heads", "num_attention_heads", "vt_num_attention_heads"]) : undefined,
    // M8-V2：Kimi 系 vision 塔的 qkv 宽独立于 hidden（qkv_hidden_size=1536 vs
    // vt_hidden_size=1024），注意力头维 = qkv_hidden_size/heads，不能用 hidden/heads。
    visionQkvHiddenSize: visionConfig ? firstNumber(visionConfig, ["qkv_hidden_size"]) : undefined,
    visionProjectorType: visionConfig?.mm_projector_type || visionConfig?.projector_type || undefined,
    visionHeadDim: visionConfig
      ? firstNumber(visionConfig, ["head_dim", "attention_head_dim"])
        ?? derivedHeadDim(firstNumber(visionConfig, ["qkv_hidden_size"]) ?? firstNumber(visionConfig, [...HIDDEN_KEYS, "vt_hidden_size"]), firstNumber(visionConfig, ["num_heads", "num_attention_heads", "vt_num_attention_heads"]))
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
    visionMergerIntermediateSize: visionConfig ? firstNumber(visionConfig, ["projection_intermediate_size"]) : undefined,
    // gated vision MLP = SwiGLU。判据是 vision hidden_act（silu/swish），
    // 不是 model_type 子串；GLM-5.3-Flash vision_config.hidden_act 已是 silu。
    visionMlpGated: visionConfig
      ? String(visionConfig.hidden_act || "").toLowerCase().includes("silu")
        || String(visionConfig.hidden_act || "").toLowerCase().includes("swish")
      : false,
    ...quantization,
    experts: pick(EXPERT_KEYS),
    routedExpertHiddenSize: pick(["routed_expert_hidden_size"]),
    expertsPerToken: pick(EXPERTS_PER_TOKEN_KEYS),
    // Qwen 系（含 qwen4_exp）checkpoint 只给 shared_expert_intermediate_size、
    // 不给 n_shared_experts。vLLM qwen3_moe / Qwen4ExpTextSparseMoeBlock：
    // width > 0 ⇒ 1 个 shared expert。个数字段优先，否则按 Qwen 宽度补 1。
    sharedExperts: pick(SHARED_EXPERT_KEYS)
      ?? (qwenSharedExpertWidth > 0 ? 1 : undefined),
    sharedExpertIntermediateSize: pick(SHARED_EXPERT_INTERMEDIATE_KEYS)
      // 通用 MoE 回退：shared expert 模块中间维 = moeIntermediateSize；融合形态
      // （sharedExpertsFused，判定归 archs 配方）乘 n_shared 得模块宽。非融合模型的
      // 个数由声明组 count 乘子处理，回退只给单专家宽。
      // 列表式启发式曾两次漏模型（deepseek_v3、glm_moe_dsa）。
      ?? ((firstNumber(textConfig, EXPERT_KEYS) ?? firstNumber(config, EXPERT_KEYS))
        ? (() => {
          const moeI = firstNumber(textConfig, MOE_INTERMEDIATE_KEYS) ?? firstNumber(config, MOE_INTERMEDIATE_KEYS);
          return moeI != null && sharedExpertsFused
            ? moeI * (pick(SHARED_EXPERT_KEYS) ?? 1)
            : moeI;
        })()
        : undefined),
    // vLLM qwen3_moe.py: shared_expert_intermediate_size > 0 时实例化
    // self.shared_expert_gate；Qwen4Exp 同构。output_gate_type 是激活种类，不是有无。
    // MiniMax 有 shared_intermediate_size 但无 gate（modeling 直接加分支）。
    sharedExpertGate: qwenSharedExpertWidth > 0,
    attentionBias: Boolean(textConfig?.attention_bias ?? config?.attention_bias),
    outputGateType: String(textConfig?.output_gate_type ?? config?.output_gate_type ?? "silu"),
    partialRotaryFactor: firstNumber(textConfig, ["partial_rotary_factor"])
      ?? firstNumber(textConfig?.rope_parameters, ["partial_rotary_factor"])
      ?? firstNumber(textConfig?.rope_scaling, ["partial_rotary_factor"]),
    rotaryDim: pick(["rotary_dim"]),
    useQkNorm: Boolean(textConfig?.use_qk_norm ?? config?.use_qk_norm),
    qkNormType: textConfig?.qk_norm_type ?? config?.qk_norm_type,
    hyperConnectionCount: pick(["hc_count"]),
    hyperConnectionLowrank: pick(["hc_lowrank"]),
    pleLayerIds: Array.isArray(textConfig?.ple_layer_ids) ? textConfig.ple_layer_ids : Array.isArray(config?.ple_layer_ids) ? config.ple_layer_ids : [],
    pleEmbedDim: pick(["ple_embed_dim"]),
    pleNgramSize: pick(["ngram_size"]),
    pleHeadsPerNgram: pick(["heads_per_ngram"]),
    pleConvKernelSize: pick(["ple_conv_kernel_size"]),
    ngramVocabSizeBase: pick(["ngram_vocab_size_base"]),
    makeNgramVocabSizeDivisibleBy: pick(["make_ngram_vocab_size_divisible_by"]),
    attnResBlockSize: pick(["attn_res_block_size"]),
    mlaUseOutputGate: Boolean(textConfig?.mla_use_output_gate ?? config?.mla_use_output_gate),
    multiHyperConnection: mhcOn,
    mhcNumResidualStreams: mhcStreams,
    mhcSinkhornIterations: pick(["mhc_sinkhorn_iterations", "hc_sinkhorn_iters"]),
    mhcTau: pick(["mhc_tau"]),
    mhcEps: pick(["hc_eps", "mhc_eps"]),
    // vLLM glm5_next PretrainedConfig 默认 mhc_post_mult_value=2.0；
    // V4 modeling 硬编码 hc_post_alpha=2.0。字段缺失时，开了 MHC 就用该默认。
    mhcPostMultValue: pick(["mhc_post_mult_value"]) ?? (mhcOn ? 2 : undefined),
    // W4：MTP 模块数。三种键名分别来自 DeepSeek/GLM 系、Qwen 系、MiniMax 系；
    // use_mtp 为布尔开关（MiniMax-M2 用），命中时按 1 个模块计。
    // DSpark（vLLM models/deepseek_v4/nvidia/dspark.py）有独立字段，不能把
    // dspark_target_layer_ids 长度写进 MTP 计数。
    mtpModules: pick(["num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_modules"])
      ?? ((textConfig?.use_mtp ?? config?.use_mtp) ? 1 : undefined),
    dsparkTargetLayerIds: Array.isArray(textConfig?.dspark_target_layer_ids)
      ? textConfig.dspark_target_layer_ids
      : Array.isArray(config?.dspark_target_layer_ids) ? config.dspark_target_layer_ids : [],
    dsparkBlockSize: pick(["dspark_block_size"]),
    dsparkMarkovRank: pick(["dspark_markov_rank"]),
    dsparkNoiseTokenId: pick(["dspark_noise_token_id"]),
    // DSpark 草稿块自带 MoE，专家数可独立于主干（V4.1：dspark_n_routed_experts=128 /
    // dspark_num_experts_per_tok=3，主干为 384/6）。model.py get_moe_config(layer_id)
    // 对 layer_id ≥ n_layers 的草稿块返回这组值。V4 系无此字段 → 草稿复用主干专家数。
    dsparkNRoutedExperts: pick(["dspark_n_routed_experts"]),
    dsparkNumExpertsPerTok: pick(["dspark_num_experts_per_tok"]),
    // DeepSeek V4.1 Engram —— n-gram 哈希记忆按门控写回残差流（model.py Engram /
    // Transformer.forward）。engram_layer_ids 是 **0-indexed 主干层号**
    // （`for layer_id in range(n_layers)`，命中即在该层入口对 h 做 gated 写入），
    // 与 ple_layer_ids 的 1-indexed 语义不同。engram_num_embeddings 逐层不同
    // （哈希表行数），engram 模块按 engramLayerIds 下标取对应表宽。
    engramLayerIds: Array.isArray(textConfig?.engram_layer_ids)
      ? textConfig.engram_layer_ids
      : Array.isArray(config?.engram_layer_ids) ? config.engram_layer_ids : [],
    engramNumEmbeddings: Array.isArray(textConfig?.engram_num_embeddings)
      ? textConfig.engram_num_embeddings.map((value) => Number(value)).filter((value) => Number.isFinite(value))
      : Array.isArray(config?.engram_num_embeddings)
        ? config.engram_num_embeddings.map((value) => Number(value)).filter((value) => Number.isFinite(value))
        : [],
    engramMaxNgramSize: pick(["engram_max_ngram_size"]),
    engramNHeads: pick(["engram_n_heads"]),
    engramHeadDim: pick(["engram_head_dim"]),
    contextLength: pick(CONTEXT_KEYS),
    tieWordEmbeddings: textConfig?.tie_word_embeddings ?? config?.tie_word_embeddings ?? false,
  };
}
