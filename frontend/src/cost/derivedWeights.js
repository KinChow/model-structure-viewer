// 离线或无 checkpoint 时的模型级权重参数量 fallback；结果必须标记为 derived。
// 来源：llm-analysis 的 get_num_params_* 公式形态；不包含架构特有 bias/额外 head。

import { deriveBuildPlan } from "../structure/config/plan.js";
import { recipeLinearAttentionMode, recipeSharedExpertsAreFused, recipeVisionInternalMerger } from "../structure/archs/index.js";
import { paramBytes } from "../structure/operators/formulas/paramDtypes.js";

export function derivedWeightParameters(config = {}) {
  const hidden0 = config.hiddenSize || 0;
  const layers0 = config.layers || 0;
  const decoder = decoderParameters(config);
  const embedding = (config.vocabSize || 0) * hidden0;
  const lmHead = config.tieWordEmbeddings ? 0 : embedding;
  const outputResidual = config.attnResBlockSize ? 2 * hidden0 : 0;
  const finalHyperConnection = config.hyperConnectionCount ? hyperConnectionFinalParameters(config) : 0;
  // W4：MTP 参数占显存但不参与每次前向（投机解码默认关闭，结构树里 MTP
  // repeat=0，见 layers/mtp.js）。计入是为支柱②「放得下吗」——
  // 51/59 内置模型带 MTP 字段，此前完全不在参数量里。单模块 = enorm+hnorm+
  // shared_head_norm(3H) + eh_proj(2H²) + 一个 decoder 层（按平均层成本）。
  // 对标 vLLM：MTP 是 registry 的独立注册项，与主干同一 checkpoint。
  const mtp = derivedMtpParameters(config);
  return embedding + decoder + hidden0 + lmHead + outputResidual + finalHyperConnection + mtp + derivedVisionParameters(config);
}

/**
 * 期望侧的**逐层明细**（诊断用）。权重字节恒等式超差时，用它与结构树的逐层叶子
 * 求和对照，把差额定位到「第几层、哪一项」，不必靠代数反推。
 * 消费者：scripts/diff-weight-identity.mjs。
 */
export function derivedDecoderLayerBreakdown(config = {}) {
  const perLayer = [];
  const total = decoderParameters(config, perLayer);
  return { total, perLayer };
}

function decoderParameters(config = {}, perLayerOut = null) {
  const plan = deriveBuildPlan(config?.raw ?? config);
  const linearMode = recipeLinearAttentionMode(config);
  const sharedFused = recipeSharedExpertsAreFused(config);
  const layers = config.layers || 0;
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const kvHeads = config.kvHeads || heads;
  const qDim = config.headDim || 0;
  const vDim = config.valueHeadDim || qDim;
  const denseIntermediate = config.intermediateSize || 0;
  const moeIntermediate = config.moeIntermediateSize || denseIntermediate;
  const experts = config.experts || 0;
  const routedExpertHidden = config.routedExpertHiddenSize || hidden;
  const sharedExperts = config.sharedExperts || 0;
  const sharedIntermediate = config.sharedExpertIntermediateSize || denseIntermediate;
  const schedule = plan.layerSchedule || Array.from({ length: layers }, () => experts ? "moe" : "dense");
  // 逐头 QK-norm 的权重：`RMSNorm(head_dim)`，跨头共享（vLLM qwen3.py:150-151、
  // qwen3_next.py:358-359、MiniMax 的 Gemma 版同形）。两条 = q_norm + k_norm。
  // 2026-09-09 由权重字节逐层归因补上（此前期望侧完全没有这一项）。
  const qkNorm = 2 * qDim;
  // attention_bias（GLM-4.7 等）：q/k/v 三个投影带 bias，o_proj 不带
  //（结构树的 qkv_proj 声明 bias=attentionBias，out_proj 无）。
  const attentionBias = config.attentionBias
    ? heads * qDim + kvHeads * qDim + kvHeads * vDim
    : 0;
  const attention = hidden * (heads * qDim + kvHeads * qDim + kvHeads * vDim + heads * vDim)
    + qkNorm + attentionBias;
  const norms = config.hyperConnectionCount ? 0 : 2 * hidden;
  let decoder = 0;
  for (let i = 0; i < layers; i++) {
    // MLA 模型（qLoraRank+kvLoraRank）在无显式 schedule 时按 MLA 计
    // （kimi_k2/deepseek 系无 layer_types，schedule undefined → 曾误按 gqa 计 attention）
    const attentionKind = plan.attentionSchedule?.[i]
      || (config.qLoraRank && config.kvLoraRank ? "mla" : "gqa");
    let attentionParameters = attention;
    if (attentionKind === "linear") {
      attentionParameters = linearMode === "glm5_next"
        ? glm5NextLinearAttentionParameters(config)
        : linearMode === "kimi_k3"
          ? kimiK3LinearAttentionParameters(config)
      : linearMode === "qwen4_exp"
          ? qwen4ExpLinearAttentionParameters(config)
            : linearMode === "qwen3_5"
              ? qwen35LinearAttentionParameters(config)
            : genericLinearAttentionParameters(config, { hidden, heads, qDim, vDim });
    } else if (attentionKind === "qwen35_full") {
      attentionParameters = qwen35FullAttentionParameters(config);
    // W5：原来这两支用 model_type 白名单
    // （["deepseek_v32","glm_moe_dsa"].includes / === "minimax_m3_vl"），
    // **漏掉了 glm5_next** —— GLM-5.3-Flash 的 11 个 DSA 层因此退回泛化 GQA 公式
    // hidden·(heads·qDim·2 + kvHeads·(qDim+vDim))=2.684e8/层，而真值（latent 压缩）
    // 约 1.24e8/层，单层多算 1.449e8，x11 层 = 1.594e9，正是权重字节恒等式
    // GLM-5.3-Flash 那条残差的来源（实测：单层变体二分 + 逐项复算双向对上）。
    // 判据改为字段存在性：DSA = kv_lora_rank + index_topk；块稀疏 = sparse_attention_config。
    } else if (attentionKind === "qsa" && config.kvLoraRank && config.dsaIndexTopk) {
      attentionParameters = dsaAttentionParameters(config);
    } else if (attentionKind === "sparse" && config.sparseBlockSize) {
      attentionParameters = minimaxSparseAttentionParameters(config, i);
    } else if (attentionKind === "dsv4" && config.qLoraRank && config.oLoraRank) {
      attentionParameters = deepseekV4AttentionParameters(config, i);
    } else if (attentionKind === "mla" && config.qLoraRank && config.kvLoraRank) {
      const ropeDim = config.qkRopeHeadDim || 0;
      const nopeDim = Math.max(0, qDim - ropeDim);
      attentionParameters = hidden * config.qLoraRank
        + config.qLoraRank * heads * qDim
        + hidden * (config.kvLoraRank + ropeDim)
        + config.kvLoraRank * (heads * nopeDim + vDim * (config.kvHeads || heads))
        + hidden * heads * vDim // o_proj（2026-09-07 补：原分支遗漏）
        // q_a_layernorm + kv_a_layernorm 两个低秩 latent 上的 RMSNorm 权重
        //（结构树里是独立的 q_a_norm / kv_a_norm 叶）。2026-09-09 由权重字节
        // 逐层归因补上：Kimi-K2 每层 2,048，61 层 = 124,928。
        + config.qLoraRank + config.kvLoraRank;
      // M8-V2（modeling_kimi_linear.py KimiMLAAttention）：kimi_k3 的 MLA 层
      // 带 full-rank 输出门 g_proj hidden×projection_size（index.json 实锤
      // 全 93 层含 g_proj 88.1M）——DeepSeek MLA 无此项
      if (config.modelType === "kimi_k3") {
        attentionParameters += hidden * heads * (nopeDim + ropeDim);
      }
    }
    const mhcParameters = config.multiHyperConnection ? mhcLayerParameters(config) : 0;
    const hcParameters = config.hyperConnectionCount ? hyperConnectionLayerParameters(config) : 0;
    // fp32 参数的**元素数**（字节宽在 paramDtypes 登记）：GDN/KDA 的 dt_bias+A_log
    // 与 mHC 的 base/scale 标量。权重字节恒等式与容量字节都要把它们按 4B 计，
    // 元素数仍留在参数量里（支柱②的参数量是纯元素计数）。
    let fp32Elements = 0;
    if (attentionKind === "linear") {
      fp32Elements += gdnDecayElements(config, linearMode);
    }
    if (config.multiHyperConnection) {
      const streams = config.mhcNumResidualStreams || 0;
      const mixRows = (2 + streams) * streams;
      const hcDim = streams * hidden;
      // attn/ffn 两套：fn[mix_hc·hc_dim] + base[mix_hc] + scale[3]，全 fp32。
      fp32Elements += 2 * (mixRows * hcDim + mixRows + 3);
    }
    // PLE（Qwen4Exp 的 Position Learning Enhancement）只挂在 ple_layer_ids 指定的层：
    // W_kv[2·pleEmbedDim, hidden] + grouped norm(pleEmbedDim) + short-conv 核
    //（pleEmbedDim·ngram）。ngram 哈希嵌入表两侧都未建模（登记在 operators_reference
    // §4.5 的「已知近似」，取证需要 Qwen modeling，尚未入库）。
    const pleParameters = config.pleLayerIds?.includes(i + 1)
      ? 2 * (config.pleEmbedDim || 0) * hidden
        + (config.pleEmbedDim || 0)
        + (config.pleEmbedDim || 0) * (config.pleNgramSize || 0)
      : 0;
    const before = decoder;
    let routedThisLayer = 0;
    if (schedule[i] === "moe" && experts > 0) {
      const routedExperts = experts * 3 * routedExpertHidden * moeIntermediate;
      routedThisLayer = routedExperts;
      // 哈希路由层没有 router GEMM。tid2eid 查表是 **buffer 不是参数**
      //（2026-09-09 裁决，见 hashRouteCounts 注释）：不进权重字节恒等式，
      // 常驻容量由 derivedBufferBytes 单独计（vocab·k·4B int32）。
      const isHashLayer = i < (config.numHashLayers || 0);
      const routerParameters = isHashLayer ? 0 : hidden * experts;
      // 潜空间 MoE（K3）：down/up 两条投影 + combine 之后 latent 上的一层 RMSNorm
      //（结构树里是 moe.routed_expert_norm 叶，宽 = routed_expert_hidden_size）。
      const latentProjection = routedExpertHidden !== hidden
        ? hidden * routedExpertHidden + routedExpertHidden * hidden + routedExpertHidden
        : 0;
      decoder += attentionParameters + norms + mhcParameters + hcParameters + pleParameters + routerParameters + routedExperts + latentProjection;
      decoder += (sharedFused ? 1 : sharedExperts) * 3 * hidden * sharedIntermediate;
    } else {
      decoder += attentionParameters + norms + mhcParameters + hcParameters + pleParameters + 3 * hidden * denseIntermediate;
    }
    if (config.attnResBlockSize) decoder += 4 * hidden;
    // 逐层明细（可选出参）：权重字节恒等式超差时用它对齐到「哪一层、哪一项」，
    // 不必再靠代数猜。消费者 = scripts/diff-weight-identity.mjs。
    // routed 单列，因为 decode 相位只读 min(k·T,E)/E 份，比对时要同口径缩放。
    if (perLayerOut) {
      perLayerOut.push({
        index: i,
        kind: schedule[i] === "moe" && experts > 0 ? "moe" : "dense",
        attentionKind,
        attention: attentionParameters,
        norms,
        mhc: mhcParameters,
        hc: hcParameters,
        ple: pleParameters,
        routed: routedThisLayer,
        fp32Elements,
        total: decoder - before,
      });
    }
  }
  // M8-V2 登记：GLM-5.3-Flash 的 hc 超连接（全局 ~35.4M）与 DSA indexer
  // （83.4M/DSA 层）未建模——两侧同缺不影响 ratio，影响绝对值（结构缺口
  // 见 details/identity_calibration.md 案例二追加二，M11 落地）
  return decoder;
}

/**
 * MTP 参数量（独立导出）。恒等式的期望侧要把它减掉——MTP 不产生 MAC，
 * 也不产生每次前向的权重读（repeat=0），与 embedding/norm 同属「有参数无算力」项。
 * 直算，不做代数反解：与 derivedWeightParameters 里的 mtp 项同式。
 */
export function derivedMtpParameters(config = {}) {
  const mtpModules = config.mtpModules || 0;
  if (!mtpModules) return 0;
  const hidden = config.hiddenSize || 0;
  const layers = config.layers || 0;
  if (!layers) return 0;
  const decoderOnly = decoderParameters(config);
  return mtpModules * (3 * hidden + 2 * hidden * hidden + decoderOnly / layers);
}

export function derivedVisionParameters(config) {
  const visionInternalMerger = recipeVisionInternalMerger(config);
  // M8-V2：Kimi 系（MoonViT3dEncoder）分支——qkv 宽独立（qkv_hidden_size）、
  // MLP2 两层无 gate、patchmerger 投影（源码：details/models/kimi-k3/）。
  // 每层 = wqkv hidden·3qkv + wo qkv·hidden + MLP2 2·hidden·mlpDim + norms 2·hidden。
  const qkvHidden = config.visionQkvHiddenSize || 0;
  const projectorType = config.visionProjectorType || "";
  if (qkvHidden || projectorType.includes("patchmerger")) {
    const layers = config.visionLayers || 0;
    const hidden = config.visionHiddenSize || 0;
    const heads = config.visionAttentionHeads || 0;
    const qkvHidden = config.visionQkvHiddenSize || hidden; // 未指定 → wqkv 3×hidden
    const mlpDim = config.visionIntermediateSize || 0;
    const patch = config.visionPatchSize || 0;
    const temporalPatch = config.visionTemporalPatchSize || 1;
    const channels = config.visionChannels || 3;
    if (!layers || !hidden || !heads || !qkvHidden || !mlpDim || !patch) return 0;
    const perLayer = hidden * 3 * qkvHidden + qkvHidden * hidden + 2 * hidden * mlpDim + 2 * hidden;
    const patchEmbedding = channels * temporalPatch * patch * patch * hidden;
    // PatchMergerMLP（models/moonshotai/Kimi-K2.5/modeling_kimi_k25.py:737-751）：
    //   pre_norm = LayerNorm(mm_hidden)                  → 2·mm_hidden（含 bias）
    //   proj[0]  = Linear(mm_hidden·merge² → 同宽, bias)  → W + bias
    //   proj[2]  = Linear(mm_hidden·merge² → text_hidden, bias)
    // 关键修正（2026-09-09 逐层归因）：第二层的输出是**文本侧 hidden**
    // （config.text_hidden_size / 模型 hiddenSize），不是 vision 的 out_hidden_size；
    // 原来把 textOutput 取成 visionOutputSize，K2.5 少算 2,760 万参数。
    const mergeKernel = config.visionMergeKernelSize
      || (config.visionMergeSize ? config.visionMergeSize ** 2 : 4);
    const mergerIn = hidden * mergeKernel; // mm_hidden_size × merge_kernel²
    const textOutput = config.hiddenSize || config.visionOutputSize || 7168;
    const preNorm = 2 * hidden;
    const merger = (mergerIn * mergerIn + mergerIn) + (mergerIn * textOutput + textOutput);
    return patchEmbedding + layers * perLayer + preNorm + merger;
  }
  const layers = config.visionLayers || 0;
  const hidden = config.visionHiddenSize || 0;
  const heads = config.visionAttentionHeads || 0;
  const headDim = config.visionHeadDim || (heads ? hidden / heads : 0);
  const intermediate = config.visionIntermediateSize || 0;
  const patch = config.visionPatchSize || 0;
  const temporalPatch = config.visionTemporalPatchSize || 1;
  // channels 缺省按 RGB=3（Kimi 等不含 in_channels 字段，曾一票否决整个推导 → vision 期望侧 7× 低估）
  const channels = config.visionChannels || 3;
  if (!layers || !hidden || !heads || !headDim || !intermediate || !patch) return 0;
  const patchEmbedding = channels * temporalPatch * patch * patch * hidden;
  const attention = hidden * (3 * heads * headDim) + (heads * headDim) * hidden + 2 * hidden;
  const mlp = config.visionMlpGated ? 3 * hidden * intermediate : 2 * hidden * intermediate;
  const output = config.visionOutputSize || hidden;
  const mergeWidth = (config.visionMergeSize || 1) ** 2 * hidden;
  const mergerIntermediate = config.visionMergerIntermediateSize || intermediate;
  const merger = visionInternalMerger
    ? config.modelType === "glm5_next"
      // GLM-5.3-Flash 的 merger（结构树逐叶实证）：norm(output) + proj(mergeWidth→output)
      // + post_norm(output) + SwiGLU 三条 output↔mergerIntermediate。
      // 原式把 `mergeWidth*output` 与 `output*output` 当成两条投影（本模型
      // mergeWidth == output == 4096，等于重复计一次），还另加了一份 downsample，
      // 净多算 33,546,240（2026-09-09 权重字节逐层归因）。
      ? mergeWidth * output + 2 * output + 3 * output * mergerIntermediate
      // Qwen 系 merger：norm(mergeWidth) + fc1(mergeWidth→mergeWidth) + fc2(mergeWidth→output)
      // norm 那一项 2026-09-09 由权重字节逐层归因补上（结构树里是
      // `vision_tower.merger.norm` 叶，0.8B 3,072 / 35B 4,608）。
      : mergeWidth + mergeWidth * mergeWidth + mergeWidth * output
    : output * (config.hiddenSize || hidden);
  // downsample 已并入上面的 merger（mergeWidth→output 那一条就是它）。
  const downsample = 0;
  return patchEmbedding + layers * (attention + mlp) + merger + downsample;
}

/**
 * GDN/KDA 衰减参数（dt_bias + A_log）的**元素数**，按 linearAttentionMode 分家族。
 * 四个家族的公式都必须通过本函数取这两项 —— 字节宽在 paramDtypes（fp32）登记，
 * 元素数与字节宽两侧各只写一遍，锁死不漂移。
 */
export function gdnDecayElements(config, mode) {
  const heads = config.linearKeyHeads || config.attentionHeads || 0;
  const headDim = config.linearKeyDim || config.headDim || 0;
  const valueHeads = config.linearValueHeads || config.attentionHeads || heads;
  switch (mode) {
    case "qwen3_5":
    case "qwen4_exp":
      return 2 * valueHeads; // dt_bias + A_log，各 num_v_heads
    case "glm5_next":
    case "kimi_k3":
      return heads * headDim + heads; // dt_bias = projection_size、A_log = num_heads
    case "generic":
      // 泛化 GDN 模板与 qwen 同形（leaf 的 state_update 对 generic 也发
      // 2·heads 的 dt_bias+A_log），default 不给 —— 显式列出已知形态。
      return 2 * valueHeads;
    default:
      return 0;
  }
}

function qwen35LinearAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const keyHeads = config.linearKeyHeads || 0;
  const valueHeads = config.linearValueHeads || 0;
  const keyDim = config.linearKeyDim || 0;
  const valueDim = config.linearValueDim || 0;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const convDim = 2 * keyProjection + valueProjection;
  const kernel = config.linearConvKernelSize || 0;
  return hidden * (2 * keyProjection + 2 * valueProjection)
    + 2 * hidden * valueHeads
    + convDim * kernel
    + gdnDecayElements(config, "qwen3_5")
    + valueDim
    + valueProjection * hidden;
}

function qwen35FullAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const kvHeads = config.kvHeads || heads;
  const headDim = config.headDim || 0;
  // + 2·headDim = q_norm/k_norm（逐头共享，见 attention 常量处的出处注释）
  return hidden * (2 * heads * headDim + 2 * kvHeads * headDim)
    + hidden * heads * headDim
    + 2 * headDim;
}

function dsaAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const qRank = config.qLoraRank || 0;
  const kvRank = config.kvLoraRank || 0;
  const qkNope = config.qkNopeHeadDim || 0;
  const rope = config.qkRopeHeadDim || 0;
  const qkDim = qkNope + rope;
  const valueDim = config.valueHeadDim || config.headDim || 0;
  const indexHeads = config.dsaIndexHeads ?? config.indexerNHeads ?? 0;
  const indexDim = config.dsaIndexHeadDim ?? config.indexerHeadDim ?? 0;
  return hidden * qRank
    + qRank * heads * qkDim
    + hidden * (kvRank + rope)
    + kvRank * heads * (qkNope + valueDim)
    + qRank * indexHeads * indexDim
    + hidden * (indexDim + indexHeads)
    + hidden * heads * valueDim
    // 三个 RMSNorm 权重（结构树里都是独立叶）：q_a_norm(qRank) + kv_a_norm(kvRank)
    // + indexer 的 k_norm(indexDim，index k 单头)。2026-09-09 逐层归因补上。
    + qRank + kvRank + indexDim;
}

function minimaxSparseAttentionParameters(config, layerIndex = 0) {
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const kvHeads = config.kvHeads || heads;
  const headDim = config.headDim || 0;
  const indexHeads = config.sparseIndexHeads || kvHeads;
  const indexDim = config.sparseIndexDim || headDim;
  // index 分支的 k（以及启用时的 v）是**单头共享**，不是 indexHeads 份：
  //   vLLM linear.py:1409-1414 output_sizes=[q, kv, kv, iq, ik=index_head_size]
  //   SGLang minimax_m3.py:632-643 index_qkv_proj 的 total_num_kv_heads=1，
  //     v_head_size = 0 if disable_index_value else idx_head_dim
  // index value 启用的层还多一个 index_o_proj（SGLang minimax_m3.py:648-658）。
  const indexValueEnabled = config.sparseDisableIndexValue?.[layerIndex] === false;
  const indexProjection = indexHeads * indexDim + indexDim + (indexValueEnabled ? indexDim : 0);
  const indexOutput = indexValueEnabled ? indexHeads * indexDim * hidden : 0;
  // + 2·headDim = q_norm/k_norm；+ 2·indexDim = index_q_norm/index_k_norm
  //（都是逐头共享的 Gemma RMSNorm，index 侧 k 单头，见 ops/index.js 的出处注释）
  return hidden * (heads * headDim + 2 * kvHeads * headDim + indexProjection)
    + indexOutput
    + hidden * heads * headDim
    + 2 * headDim
    + 2 * indexDim;
}

function deepseekV4AttentionParameters(config, layerIndex) {
  const hidden = config.hiddenSize || 0;
  const heads = config.attentionHeads || 0;
  const headDim = config.headDim || 0;
  const qRank = config.qLoraRank || 0;
  const outputRank = config.oLoraRank || 0;
  const groups = config.oGroups || 1;
  const ratio = config.compressRatios?.[layerIndex] ?? 0;
  const qkv = hidden * (qRank + headDim);
  const query = qRank * heads * headDim;
  const output = (heads * headDim) * (groups * outputRank) + (groups * outputRank) * hidden;
  // 两个低秩 latent 上的 RMSNorm 权重（结构树里是 q_norm / kv_norm 两片独立叶，
  // 宽度 = q_lora_rank 与压缩 latent 的单侧宽 headDim）。2026-09-09 逐层归因补上。
  const latentNorms = qRank + headDim;
  if (ratio <= 1) return qkv + query + output + latentNorms;
  const compressor = hidden * 2 * (ratio === 4 ? 2 : 1) * headDim;
  const indexer = ratio === 4
    ? hidden * (config.indexerNHeads || 0) + qRank * (config.indexerNHeads || 0) * (config.indexerHeadDim || 0)
    : 0;
  return qkv + query + output + compressor + indexer + latentNorms;
}

function genericLinearAttentionParameters(config, { hidden, heads, qDim, vDim }) {
  const keyHeads = config.linearKeyHeads || heads;
  const valueHeads = config.linearValueHeads || heads;
  const keyDim = config.linearKeyDim || qDim;
  const valueDim = config.linearValueDim || vDim;
  // GDN 的 dt_bias + A_log（fp32，paramDtypes 登记）—— 与 leaf 的
  // state_update 同源，此前整片缺失（泛化形态此前无现网载体故未暴露）。
  return hidden * (keyHeads * keyDim + valueHeads * valueDim + hidden)
    + gdnDecayElements(config, "generic");
}

// GLM-5.3-Flash uses six-way fused qkvbfg_a plus separate f_b/g_b projections,
// three depthwise causal convolutions, A_log/dt_bias, gated RMSNorm and o_proj.
function glm5NextLinearAttentionParameters(config) {
  // 来源：modeling_glm5_next.py Glm5NextTextLinearAttention（details/models/glm5-next/）
  // q/k/v 各 hidden×qkv_dim；b_proj hidden×heads；**gate 为 low-rank**
  // （g_a hidden→head_dim + g_b head_dim→qkv_dim，非 full-rank）；o_norm head_dim；
  // o_proj qkv_dim×hidden；dt_bias qkv_dim；q/k/v 短卷积 3×qkv_dim×kernel。
  const hidden = config.hiddenSize || 0;
  const heads = config.linearKeyHeads || config.attentionHeads || 0;
  const headDim = config.linearKeyDim || config.headDim || 0;
  const qkvDim = headDim * heads;
  const convKernel = config.linearConvKernelSize || 0;
  // 逐项对齐 vLLM glm5next/nvidia/kda.py:179-254（2026-09-09 重核）：
  //   in_proj_qkvbfg_a = hidden × (3·P + num_heads + head_dim(f_a) + head_dim(g_a))
  //   f_b_proj = head_dim × P     g_b_proj = head_dim × P
  //   q/k/v_conv1d = 3 × conv × P    dt_bias = P    A_log = num_heads
  //   o_norm = head_dim（逐头）      o_proj = P × hidden
  // 补齐前少了 g_a、g_b、A_log、o_norm 四项，共 1,573,056/层。
  return hidden * (3 * qkvDim + heads + 2 * headDim)
    + headDim * qkvDim
    + headDim * qkvDim
    + 3 * qkvDim * convKernel
    + gdnDecayElements(config, "glm5_next")
    + headDim
    + qkvDim * hidden;
}

function kimiK3LinearAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const heads = config.linearKeyHeads || config.attentionHeads || 0;
  const headDim = config.linearKeyDim || config.headDim || 0;
  const projection = heads * headDim;
  const convKernel = config.linearConvKernelSize || 0;
  return hidden * 4 * projection
    + hidden * heads
    + hidden * headDim
    + headDim * projection
    + 3 * projection * convKernel
    + gdnDecayElements(config, "kimi_k3")
    + headDim
    + projection * hidden;
}

function qwen4ExpLinearAttentionParameters(config) {
  const hidden = config.hiddenSize || 0;
  const keyHeads = config.linearKeyHeads || config.attentionHeads || 0;
  const valueHeads = config.linearValueHeads || config.attentionHeads || keyHeads;
  const keyDim = config.linearKeyDim || config.headDim || 0;
  const valueDim = config.linearValueDim || config.valueHeadDim || keyDim;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const convDim = 2 * keyProjection + valueProjection;
  const kernel = config.linearConvKernelSize || 0;
  return hidden * (2 * keyProjection + 2 * valueProjection)
    + 2 * hidden * valueHeads
    + convDim * kernel
    + gdnDecayElements(config, "qwen4_exp")
    + valueDim
    + valueProjection * hidden;
}

function hyperConnectionLayerParameters(config) {
  const streams = config.hyperConnectionCount || 0;
  const hidden = config.hiddenSize || 0;
  const lowrank = config.hyperConnectionLowrank || 0;
  if (!streams || !hidden || !lowrank) return 0;
  const hyperHidden = streams * hidden;
  const oneBranch = hyperHidden + hyperHidden * (lowrank + streams) + lowrank * hyperHidden;
  return 2 * oneBranch;
}

function hyperConnectionFinalParameters(config) {
  const streams = config.hyperConnectionCount || 0;
  const hidden = config.hiddenSize || 0;
  const lowrank = config.hyperConnectionLowrank || 0;
  if (!streams || !hidden || !lowrank) return 0;
  const hyperHidden = streams * hidden;
  return hyperHidden + hyperHidden * lowrank + lowrank * hyperHidden;
}

function mhcLayerParameters(config) {
  const streams = config.mhcNumResidualStreams || 0;
  const hidden = config.hiddenSize || 0;
  if (!streams || !hidden) return 0;
  const mixRows = (2 + streams) * streams;
  const oneProjection = mixRows * streams * hidden + mixRows + 3;
  return 2 * oneProjection;
}

/**
 * 常驻 buffer 的字节（**不是参数**，不进权重字节恒等式，只进显存容量）：
 * - DeepSeek V4 哈希层的 tid2eid 查表：num_hash_layers × vocab × k，int32
 *   （Megatron-Bridge：`tid2eid` buffer, int32；"Buffers are not parameters"）。
 */
export function derivedBufferBytes(config = {}, bytesPerElement = 4) {
  const hashLayers = config.numHashLayers || 0;
  if (!hashLayers) return 0;
  const tableEntries = (config.vocabSize || 0) * (config.expertsPerToken || 0);
  return hashLayers * tableEntries * bytesPerElement;
}

/**
 * fp32 参数的元素数（字节宽见 paramDtypes）。注意：MTP 的期望侧近似
 * （decoderOnly/layers）不含 fp32 修正，本函数只覆盖主干层 —— 对 MTP
 * 一层的份额误差是每家族至多一份 gdn/mhc 标量（KB 级）。
 */
export function derivedFp32Parameters(config = {}) {
  const { perLayer } = derivedDecoderLayerBreakdown(config);
  return perLayer.reduce((sum, row) => sum + (row.fp32Elements || 0), 0);
}

export function derivedWeightBytes(config = {}, bytesPerElement = 2) {
  const params = derivedWeightParameters(config);
  const fp32 = derivedFp32Parameters(config);
  // fp32 参数按 4B 计（paramDtypes 登记值），其余按调用方字节宽。
  return params * bytesPerElement + fp32 * (4 - bytesPerElement);
}
