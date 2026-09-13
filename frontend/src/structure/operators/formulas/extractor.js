// extractor.js —— node → counts ctx 提取器（docs/details/cost_counts.md 提取器规格）。
//
// 原则（principles §3.2 / §4.3）：
// - 查表优先：节点已有 weight_shapes / input_shape / output_shape / attributes；
//   提取器只补 phase 的 T/S、变体参数、expertFraction。
// - 分派只基于 type / attributes.operator_id / 结构化 attributes 与节点路径（结构化 id），
//   禁止显示名（node.name）参与分派。
// - 不 import cost 层；bytesPerElement 由调用方传入（W5 接线点）。
// - 返回单实例 counts；repeat 倍乘由 walker 的 multiplier 处理（与旧链同）。
// - 分派：FORMULAS[operator_id].fromNode 抽 ctx，.counts(ctx) 计价（flop_registry）。
//   type=attention / type=embedding 无 operator_id，仍在 countsForNode 入口处理。

import { scoredPairs, embedGatherCounts } from "./counts.js";
import { paramBytes } from "./paramDtypes.js";
import { FORMULAS } from "./index.js";
import { tensorDims } from "../../config/dims.js";
import { visionDimensions } from "../../config/visionDims.js";
import { recipeLinearAttentionMode } from "../../archs/index.js";

// 路径正则全仓统一处（旧 compute.js/parallel.js 三种变体收敛于此）
export const LAYER_INDEX_RE = /(?:^|\.)(?:layers|language_model|decoder)\.(\d+)(?:\.|$)/;
export const ROUTED_EXPERT_RE = /(?:^|\.)(?<!shared_)(?:experts|expert_mlp)(?:\.|$)/;
export const VISION_PATH_RE = /(?:^|\.)(?:visual|vision_tower)(?:\.|$)/;

export function isVisionPath(path) {
  return VISION_PATH_RE.test(String(path || ""));
}

/** 与旧 tokensFor 逐字等价：decode 1 token；vision 用 visionTokens。 */
export function tokensFor({ batch = 1, sequence = 1, phase = "prefill", vision = false, visionTokens = 1 } = {}) {
  return batch * (vision ? visionTokens : phase === "decode" ? 1 : sequence);
}

export function layerIndexOf(path) {
  const match = String(path || "").match(LAYER_INDEX_RE);
  return match ? Number(match[1]) : null;
}

/** routed expert 叶按 k/E 缩放。dense 层没有 experts 叶，不必再查 layerSchedule。 */
export function expertFractionFor(path, config) {
  const routed = ROUTED_EXPERT_RE.test(String(path || ""));
  return routed && config?.experts && config?.expertsPerToken
    ? config.expertsPerToken / config.experts
    : 1;
}

function productOf(values) {
  return values.reduce((total, value) => total * value, 1);
}

/** 与旧 staticWidth 逐字等价：只保留正有限维并求积；无正维返回 null。 */
function staticWidth(shape) {
  if (!Array.isArray(shape) || shape.length < 1) return null;
  const dimensions = shape.filter((value) => Number.isFinite(value) && value > 0);
  if (dimensions.length === 0) return null;
  return productOf(dimensions);
}

/**
 * 归一化的**权重宽度** = 最后一维。
 *
 * RMSNorm 沿最后一维归一化，权重就是最后一维那么长，跨其余维度共享。
 * 三维 [B,T,H] 时它与 staticWidth 相同；**逐头**的四维 [B,T,heads,headDim] 时
 * staticWidth 会给出 heads·headDim，把权重放大 heads 倍。
 * 上游实证：q_norm/k_norm = `RMSNorm(self.head_dim)`（vLLM qwen3.py:150-151、
 * qwen3_next.py:358-359）；GDN 的输出门 = `RMSNormGated(self.head_v_dim)`
 *（qwen_gdn_linear_attn.py:487-488），都不是全宽。
 */
function normWeightWidth(shape) {
  if (!Array.isArray(shape) || shape.length < 1) return null;
  for (let i = shape.length - 1; i >= 0; i -= 1) {
    if (Number.isFinite(shape[i]) && shape[i] > 0) return shape[i];
  }
  return null;
}

/** 线性逻辑形状：logical_weight_shape 属性优先，其次 weight_shapes 中首个 ≥2 维形状。
 *  packed（qweight）且无逻辑形状 → null（未知，沿用旧链诚实语义）。 */
/** 线性逻辑形状：声明 shape 优先（grouped BMM 如 V4 wo_a 的激活末维乘积 ≠ 权重）。
 *  其次 logical_weight_shape / weight_shapes；都没有才从 input/output 正维积推导。 */
function declaredLinearShape(node) {
  const group = node?.attributes?.weightMatrices?.find((entry) => Array.isArray(entry?.shape) && entry.shape.length >= 2);
  if (group) return [group.out ?? group.shape[0], group.in ?? group.shape.slice(1).reduce((total, value) => total * value, 1)];
  return null;
}

function linearLogicalShape(node) {
  if (node?.weight_shapes?.qweight && !node?.attributes?.logical_weight_shape) return null;
  const logical = declaredLinearShape(node)
    || node?.attributes?.logical_weight_shape
    || Object.values(node?.weight_shapes || {}).find((shape) => Array.isArray(shape) && shape.length >= 2);
  return logical || null;
}

/** 旧 derivedLinearMacs 等价：从 input/output 正维积推导 [out, in]。 */
function derivedLinearShape(node) {
  const inputWidth = staticWidth(node?.input_shape);
  const outputWidth = staticWidth(node?.output_shape);
  if (inputWidth == null || outputWidth == null) return null;
  return [outputWidth, inputWidth];
}

function shapeMatchesPattern(shape, pattern) {
  if (!Array.isArray(shape) || !Array.isArray(pattern) || shape.length !== pattern.length) return false;
  return shape.every((value, index) => pattern[index] === -1 || pattern[index] === value);
}

/** scores / context 的输出 shape 模式（text + vision 两套，来源 dims.js / vision.js）。 */
function attentionShapePatterns(config) {
  const dims = tensorDims(config);
  const v = visionDimensions(config);
  return {
    scores: [dims.attentionScores, v.scores],
    context: [dims.attentionContext, v.context],
  };
}

function linearAttentionDimensions(config = {}) {
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || keyHeads;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  return { keyHeads, valueHeads, keyDim, valueDim, keyProjection: keyHeads * keyDim, valueProjection: valueHeads * valueDim };
}
/** mHC 的混合行数 mix_hc = (2 + hc_mult)·hc_mult（vLLM deepseek_v4 model.py:711）。 */
function mhcMixRows(config) {
  const m = config?.mhcNumResidualStreams || 0;
  return (2 + m) * m;
}
/** mHC 的多流拼接宽 hc_dim = hc_mult·hidden（同上 :712）。 */
function mhcDim(config, hidden) {
  return (config?.mhcNumResidualStreams || 0) * hidden;
}

function gatedDeltaStateCtx(config, options, bytesPerElement, modelKind = "", generic = false) {
  const { keyHeads, valueHeads, keyDim, valueDim } = linearAttentionDimensions(config);
  const heads = valueHeads || keyHeads || 1;
  // qwen GDN：dt_bias 与 A_log 都是 num_v_heads → 2·heads
  // GLM5-Next / K3 KDA：A_log = num_heads、dt_bias = projection_size → heads + heads·valueDim
  // 判据来自节点 attributes.model_kind（模板声明，不是 model_type 子串）。
  const kdaLayout = modelKind === "glm5_next" || modelKind === "kimi_k3";
  return {
    batch: options.batch ?? 1,
    sequence: options.sequence ?? 1,
    phase: options.phase ?? "prefill",
    bytesPerElement,
    keyHeads,
    valueHeads,
    keyDim,
    valueDim,
    convKernelSize: config?.linearConvKernelSize || 0,
    generic,
    gdnScalars: kdaLayout ? heads + heads * (valueDim || 0) : 2 * heads,
  };
}

function attentionKvCtx({ config, vision, kind, tokens, options, bytesPerElement }) {
  const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
  const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
  const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
  const queryTokens = tokens;
  const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
  const latentShared = !vision && kind.includes("mla") && (config?.kvLoraRank || 0) > 0;
  const kvHeads = vision ? heads : (latentShared ? 1 : (config?.kvHeads || heads));
  const kReadWidth = latentShared ? (config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0) : headDim;
  const vReadWidth = latentShared ? (config?.kvLoraRank || 0) : valueDim;
  return { heads, headDim, valueDim, queryTokens, keyTokens, latentShared, kvHeads, kReadWidth, vReadWidth, bytesPerElement };
}
// ---------- 旧链镜像结束 ----------
const FROM_NODE = {
  linear: ({ node, config, path, bytesPerElement, tokens }) => {
          // 与旧 isLinear 的 embed 排除等价：以结构化路径判断（node.name 不参与，§3.2）。
          // TODO(W3): builder 为 embed 投影声明结构化标记后移除路径判断。
          // **只排文本 token 嵌入**（真查表，gather 无 MAC、不读权重矩阵）。
          // 原判据写成 `(patch_)?embed` 把**视觉 patch embedding 也当成查表**了 ——
          // 它是 Conv3d(in_ch, hidden, kernel=(T_p,P,P), stride=kernel, bias=False)
          //（vLLM qwen2_5_vl.py:548-560：view 后 conv 再 view，stride==kernel 即一次
          // GEMM [L, C·T_p·P²]×[C·T_p·P², hidden]），权重 = C·T_p·P²·hidden。
          if (/(^|\.)embed(_tokens)?$/.test(String(node?.id || path))) {
            const hidden = staticWidth(node?.output_shape) || config?.hiddenSize || 0;
            return { embedGather: true, tokens, hidden, bytesPerElement };
          }
          const expertFraction = expertFractionFor(node?.id || path, config);
          const logical = linearLogicalShape(node) || derivedLinearShape(node);
          if (!logical) return null;
          return {
            logicalShape: logical, tokens, bytesPerElement, expertFraction,
            bias: node?.attributes?.bias === true,
          };
  },
  matmul: ({ node, config, options, bytesPerElement, kind, vision, tokens, phase }) => {
          const patterns = attentionShapePatterns(config);
          const output = node?.output_shape;
          const part = patterns.context.some((pattern) => shapeMatchesPattern(output, pattern)) ? "context"
            : patterns.scores.some((pattern) => shapeMatchesPattern(output, pattern)) ? "scores"
            : null;
          if (!part) return null;
          return { part, phase, ...attentionKvCtx({ config, vision, kind, tokens, options, bytesPerElement }) };
  },
  sdpa_attention: ({ config, options, bytesPerElement, kind, vision, tokens, phase }) => ({
          phase, ...attentionKvCtx({ config, vision, kind, tokens, options, bytesPerElement }),
  }),
  qsa_sparse_attention: ({ config, options, bytesPerElement, operatorId, kind, vision, tokens, phase }) => {
          const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
          const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
          const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
          const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
          const budget = (operatorId === "qsa_sparse_attention"
            ? config?.qsaIndexerBudget
            : config?.dsaIndexTopk) ?? keyTokens;
          const selected = Math.min(keyTokens, budget || keyTokens);
          const latentRead = kind !== "qsa" && (config?.kvLoraRank || 0) > 0;
          const kvHeads = latentRead ? 1 : config?.kvHeads || heads;
          const kWidth = latentRead ? (config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0) : headDim;
          const vWidth = latentRead ? (config?.kvLoraRank || 0) : valueDim;
          const kvWrite = latentRead || kind === "dsv4_sparse_mla"
            ? 0
            : kvHeads * tokens * (headDim + valueDim);
          const dsv4Window = kind === "dsv4_sparse_mla"
            ? kvHeads * Math.min(options.sequence || 1, config?.slidingWindow || 128) * headDim
            : 0;
          return {
            heads, tokens, selected, headDim, valueDim, bytesPerElement,
            kvHeads, kWidth, vWidth, latentRead, kvWrite, dsv4Window, phase,
          };
  },
  minimax_sparse_attention: ({ config, options, bytesPerElement, vision, tokens, phase }) => {
          const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
          const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
          const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
          const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
          const selectedTokens = (config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0);
          const size = config?.sparseBlockSize || 1;
          const selected = Math.min(keyTokens, selectedTokens * size);
          return {
            heads, tokens, selected, headDim, valueDim, bytesPerElement,
            kvHeads: config?.kvHeads || heads, phase,
          };
  },
  dsv4_swa_attention: ({ node, config, options, path, bytesPerElement, phase }) => {
          const layerIndex = layerIndexOf(node?.id || path) ?? 0;
          const headDim = config?.headDim || 0;
          return {
            batch: options.batch ?? 1,
            sequence: options.sequence ?? 1,
            phase,
            ratio: config?.compressRatios?.[layerIndex] ?? 0,
            slidingWindow: config?.slidingWindow,
            heads: config?.attentionHeads || 0,
            headDim,
            valueDim: config?.valueHeadDim || headDim,
            kvHeads: config?.kvHeads || 1,
            bytesPerElement,
          };
  },
  dsv4_compressed_attention: ({ node, config, options, path, bytesPerElement, phase }) => {
          const layerIndex = layerIndexOf(node?.id || path) ?? 0;
          const sequence = options.sequence ?? 1;
          const headDim = config?.headDim || 0;
          const slidingWindow = config?.slidingWindow;
          return {
            batch: options.batch ?? 1,
            sequence,
            phase,
            ratio: config?.compressRatios?.[layerIndex] ?? 0,
            slidingWindow,
            indexerBudget: config?.dsaIndexTopk,
            heads: config?.attentionHeads || 0,
            headDim,
            valueDim: config?.valueHeadDim || headDim,
            kvHeads: config?.kvHeads || 1,
            bytesPerElement,
            windowTokens: Math.min(sequence, slidingWindow || 128),
          };
  },
  softmax: ({ config, options, bytesPerElement, vision, tokens, phase }) => {
          const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
          const keyTokens = vision ? config?.visionTokens || 1 : options.sequence ?? 1;
          return { elements: heads * scoredPairs({ phase, queryTokens: tokens, keyTokens }), bytesPerElement };
  },
  rope: ({ node, config, bytesPerElement, vision, tokens }) => {
          const factor = node?.attributes?.partial_rotary_factor ?? config?.partialRotaryFactor ?? 1;
          const ropeHeads = vision
            ? 2 * (config?.visionAttentionHeads || 0)
            : (config?.attentionHeads || 0) + (config?.kvHeads || config?.attentionHeads || 0);
          const ropeDim = (vision ? config?.visionHeadDim || 0 : config?.headDim || 0) * factor;
          return { tokens, ropeDims: ropeHeads * ropeDim, bytesPerElement };
  },
  rmsnorm: ({ node, bytesPerElement, operatorId, tokens }) => ({
            tokens, hidden: staticWidth(node?.input_shape) || 0, bytesPerElement,
            weightOne: operatorId === "gemma_rmsnorm",
            weightWidth: normWeightWidth(node?.input_shape) || undefined,
            affineBias: node?.attributes?.affine_bias === true,
          }),
  gated_rmsnorm: ({ node, bytesPerElement, tokens }) => ({
            tokens, hidden: staticWidth(node?.input_shape) || 0, bytesPerElement, gated: true,
            weightWidth: normWeightWidth(node?.input_shape) || undefined,
          }),
  attention_output_gate: ({ node, bytesPerElement, tokens }) => ({
    tokens, width: staticWidth(node?.output_shape) || 0, bytesPerElement,
  }),
  vision_activation: ({ node, bytesPerElement, tokens }) => ({
    tokens, intermediate: staticWidth(node?.output_shape) || 0, bytesPerElement,
  }),
  vision_position: ({ node, bytesPerElement, tokens }) => ({
    tokens, hidden: staticWidth(node?.output_shape) || 0, bytesPerElement,
  }),
  vision_merge: ({ node, config, bytesPerElement, tokens }) => {
          const inWidth = staticWidth(node?.input_shape) || 0;
          const outWidth = staticWidth(node?.output_shape) || 0;
          const mergeSize = Math.max(config?.visionMergeSize || 1, 1);
          const outTokens = Math.max(1, Math.floor(tokens / (mergeSize * mergeSize)));
          return {
            inElements: inWidth * tokens,
            outElements: outWidth * outTokens,
            bytesPerElement,
          };
  },
  split: () => ({}),
  swiglu: ({ node, bytesPerElement, tokens }) => ({
    tokens, intermediate: staticWidth(node?.output_shape) || 0, bytesPerElement,
  }),
  fused_moe_mlp: ({ node, config, bytesPerElement, tokens }) => ({
            tokens,
            topk: config?.expertsPerToken || 1,
            experts: config?.experts || 0,
            expertHidden: node?.attributes?.latent_size || config?.routedExpertHiddenSize || config?.hiddenSize || 0,
            expertIntermediate: config?.moeIntermediateSize || config?.intermediateSize || 0,
            bytesPerElement,
          }),
  causal_conv1d: ({ config, bytesPerElement, tokens, phase }) => {
          const { keyProjection, valueProjection } = linearAttentionDimensions(config);
          return {
            tokens,
            channels: 2 * keyProjection + valueProjection,
            kernel: config?.linearConvKernelSize || 0,
            bytesPerElement,
            phase,
          };
  },
  linear_attention: ({ node, config, options, path, bytesPerElement, tokens, phase }) => {
          const idPath = String(node?.id || path);
          if (/short_conv|conv/.test(idPath)) {
            const { keyProjection, valueProjection } = linearAttentionDimensions(config);
            return {
              variant: "conv",
              tokens,
              channels: 2 * keyProjection + valueProjection,
              kernel: config?.linearConvKernelSize || 0,
              bytesPerElement,
              phase,
              includeWeights: false,
              includeConvState: false,
            };
          }
          if (/state|recurrent/.test(idPath)) {
            return {
              variant: "state",
              ...gatedDeltaStateCtx(
                config,
                { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase },
                bytesPerElement,
                String(node?.attributes?.model_kind || ""),
                recipeLinearAttentionMode(config) === "generic",
              ),
            };
          }
          return { variant: "zero" };
  },
  gated_delta_attention: ({ node, config, options, bytesPerElement, phase }) => gatedDeltaStateCtx(
    config,
    { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase },
    bytesPerElement,
    String(node?.attributes?.model_kind || ""),
    recipeLinearAttentionMode(config) === "generic",
  ),
  topk: ({ config, bytesPerElement, tokens }) => ({
    tokens, experts: config?.experts || 0, topk: config?.expertsPerToken || 0, bytesPerElement, normTopkProb: config?.normTopkProb ?? true,
  }),
  moe_dispatch: ({ node, config, bytesPerElement, tokens }) => ({
    tokens, hidden: staticWidth(node?.input_shape) || config?.hiddenSize || 0, topk: config?.expertsPerToken || 0, bytesPerElement,
  }),
  moe_combine: ({ node, config, bytesPerElement, tokens }) => ({
    tokens, hidden: staticWidth(node?.input_shape) || config?.hiddenSize || 0, topk: config?.expertsPerToken || 0, bytesPerElement,
  }),
  residual_add: ({ node, config, bytesPerElement, tokens }) => ({
    tokens, hidden: staticWidth(node?.output_shape) || config?.hiddenSize || 0, bytesPerElement,
  }),
  identity: () => ({ copy: false }),
  moe_add: ({ node, config, bytesPerElement, tokens }) => ({
    tokens, hidden: staticWidth(node?.output_shape) || config?.hiddenSize || 0, bytesPerElement,
  }),
  dsv4_hash_route: ({ config, bytesPerElement, tokens }) => ({
            tokens,
            topk: config?.expertsPerToken || 0,
            bytesPerElement,
          }),
  mla_query_compress: ({ config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
    return {
          qa: { logicalShape: [config?.qLoraRank || 0, H], tokens, bytesPerElement },
          norm: { tokens, hidden: config?.qLoraRank || 0, bytesPerElement },
        };
  },
  mla_kv_compress: ({ node, config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
    return {
          proj: {
            logicalShape: [
              staticWidth(node?.output_shape) || (config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0),
              H,
            ],
            tokens,
            bytesPerElement,
          },
        };
  },
  qsa_indexer: ({ config, options, bytesPerElement, tokens, phase }) => ({
          heads: config?.qsaIndexerHeads ?? 0,
          dim: config?.qsaIndexerHeadDim ?? 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: config?.qsaIndexerBudget ?? 0,
          pool: config?.qsaIndexerCompressRatio ?? 1,
          poolStage: (config?.qsaIndexerCompressRatio ?? 1) > 1 ? "key" : "none",
          perHeadWeights: false,
          phase,
          b: bytesPerElement,
        }),
  dsa_indexer: ({ config, options, bytesPerElement, tokens, phase }) => ({
          heads: config?.dsaIndexHeads ?? 0,
          dim: config?.dsaIndexHeadDim ?? 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: config?.dsaIndexTopk ?? 0,
          pool: 1,
          poolStage: "none",
          perHeadWeights: true,
          phase,
          b: bytesPerElement,
        }),
  dsa_kpool_indexer: ({ config, options, bytesPerElement, tokens, phase }) => ({
          heads: config?.dsaIndexHeads ?? 0,
          dim: config?.dsaIndexHeadDim ?? 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: config?.dsaIndexTopk ?? 0,
          pool: config?.dsaIndexKpool ?? 1,
          poolStage: "key",
          perHeadWeights: true,
          phase,
          b: bytesPerElement,
        }),
  dsv4_indexer: ({ config, options, bytesPerElement, tokens, phase }) => ({
          heads: config?.dsaIndexHeads ?? 0,
          dim: config?.dsaIndexHeadDim ?? 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: config?.dsaIndexTopk ?? 0,
          pool: 1,
          poolStage: "none",
          perHeadWeights: true,
          phase,
          b: bytesPerElement,
        }),
  minimax_sparse_indexer: ({ config, options, bytesPerElement, tokens, phase }) => ({
          heads: config?.sparseIndexHeads || 0,
          dim: config?.sparseIndexDim || 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: ((config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0)) * (config?.sparseBlockSize || 1),
          pool: config?.sparseBlockSize || 1,
          poolStage: "score",
          perHeadWeights: false,
          phase,
          b: bytesPerElement,
        }),
  attention_residual: ({ config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
    return {
          norms: { tokens, hidden: H, bytesPerElement },
          scoreProj: { logicalShape: [1, H], tokens, bytesPerElement },
          aggregate: { elements: H * tokens, bytesPerElement },
          mix: { tokens, hidden: H, bytesPerElement },
        };
  },
  hyper_connection: ({ node, config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
          const streams = config?.hyperConnectionCount || 1;
          const lowrank = config?.hyperConnectionLowrank || 0;
          const hyperHidden = streams * H;
          return {
            grouped: { tokens, hidden: hyperHidden, weightOne: true, bytesPerElement },
            mixDown: { logicalShape: [lowrank, hyperHidden], tokens, bytesPerElement },
            silu: { tokens, width: lowrank, bytesPerElement },
            mixUp: { logicalShape: [hyperHidden, lowrank], tokens, bytesPerElement },
            gate: { tokens, width: hyperHidden, bytesPerElement },
            inject: node?.attributes?.hc_use_combine === false
              ? { logicalShape: [streams, hyperHidden], tokens: 0, bytesPerElement, weightsShared: true }
              : { logicalShape: [streams, hyperHidden], tokens, bytesPerElement },
            combine: { tokens, hidden: hyperHidden, bytesPerElement },
          };
  },
  ple: ({ config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
    return {
          kv: { logicalShape: [2 * (config?.pleEmbedDim || 0), H], tokens, bytesPerElement },
          norm: { tokens, hidden: config?.pleEmbedDim || 0, bytesPerElement },
          conv: { tokens, channels: config?.pleEmbedDim || 0, kernel: config?.pleNgramSize || 1, bytesPerElement },
          add: { tokens, hidden: H, bytesPerElement },
        };
  },
  mhc_pre: ({ config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
    return {
          mix: { tokens, width: H, bytesPerElement },
          matrix: { logicalShape: [mhcMixRows(config), mhcDim(config, H)], tokens, bytesPerElement, weightBytesPerElement: paramBytes("mhc_fn") },
          base: { logicalShape: [mhcMixRows(config), 1], tokens: 0, bytesPerElement: paramBytes("mhc_base") },
          scale: { logicalShape: [3, 1], tokens: 0, bytesPerElement: paramBytes("mhc_scale") },
          norm: { tokens, hidden: H, bytesPerElement },
          sinkhorn: { tokens, streams: config?.mhcNumResidualStreams || 0, iterations: config?.mhcSinkhornIterations || 0, bytesPerElement },
          merge: { tokens, hidden: H, bytesPerElement },
        };
  },
  mhc_post: ({ config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
    return {
          combine: { logicalShape: [mhcMixRows(config), mhcDim(config, H)], tokens, bytesPerElement, weightsShared: true },
          inject: { tokens, hidden: H, bytesPerElement },
        };
  },
  mhc_fused_post_pre: ({ config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
    return {
          post: { tokens, width: H, bytesPerElement },
          inject: { tokens, hidden: H, bytesPerElement },
          pre: { tokens, width: H, bytesPerElement },
          matrix: { logicalShape: [mhcMixRows(config), mhcDim(config, H)], tokens, bytesPerElement, weightBytesPerElement: paramBytes("mhc_fn") },
          base: { logicalShape: [mhcMixRows(config), 1], tokens: 0, bytesPerElement: paramBytes("mhc_base") },
          scale: { logicalShape: [3, 1], tokens: 0, bytesPerElement: paramBytes("mhc_scale") },
          norm: { tokens, hidden: H, bytesPerElement },
          sinkhorn: { tokens, streams: config?.mhcNumResidualStreams || 0, iterations: config?.mhcSinkhornIterations || 0, bytesPerElement },
        };
  },
  mhc_contract: ({ config, bytesPerElement, tokens }) => {
    const H = config?.hiddenSize || 0;
    return {
          contract: { tokens, hidden: H, bytesPerElement },
        };
  },
};

FROM_NODE.dsa_sparse_mla = FROM_NODE.qsa_sparse_attention;
FROM_NODE.dsv4_sparse_mla = FROM_NODE.qsa_sparse_attention;
FROM_NODE.gemma_rmsnorm = FROM_NODE.rmsnorm;
FROM_NODE.mla_output_gate = FROM_NODE.attention_output_gate;
FROM_NODE.linear_attention_gate = FROM_NODE.attention_output_gate;
FROM_NODE.shared_expert_gate = FROM_NODE.attention_output_gate;
FROM_NODE.mla_kv_split = FROM_NODE.split;
FROM_NODE.qwen_qkvz_split = FROM_NODE.split;
FROM_NODE.attention_qkv_split = FROM_NODE.split;

/**
 * 计算单个算子节点的动作向量。
 * 分派 = FORMULAS[operator_id].fromNode 抽 ctx，.counts(ctx) 计价（flop_registry）。
 * @returns 动作向量；matrix 无法确定时返回 null（调用方计入 unknownComputePaths）。
 */
export function countsForNode(node, env = {}) {
  const { config, options = {}, path = "", bytesPerElement = 2 } = env;
  const operatorId = String(node?.attributes?.operator_id || "").toLowerCase();
  const type = String(node?.type || "").toLowerCase();
  const kind = String(node?.attributes?.attention_kind || "").toLowerCase();
  const vision = node?.attributes?.modality === "vision";
  const phase = options.phase ?? "prefill";
  const tokens = tokensFor({
    batch: options.batch ?? 1,
    sequence: options.sequence ?? 1,
    phase,
    vision,
    visionTokens: config?.visionTokens || 1,
  });

  // type=attention 是 nn.Module 容器（§2.4：不该挂融合 counts）。打分核在
  // sdpa / sparse / swa / compressed 叶上，走 FORMULAS[operator_id]。
  if (type === "attention") return null;

  // M11 bytes 补齐：embedding gather 是真实访存（每 token 读一行权重、写一行
  // hidden），但它是无 operatorId 的结构节点。gather 无 MACs，matrix 恒 0。
  if (type === "embedding") {
    const hidden = staticWidth(node?.output_shape) || config?.hiddenSize || 0;
    return embedGatherCounts({ tokens, hidden, bytesPerElement });
  }

  // 旧 isLinear 等价：无 operatorId 但 weight_shapes 含 ≥2 维形状的节点按 linear 计
  // （checkpoint 绑定叶的常见形态）；embed 排除以结构化路径判断（§3.2）。
  const isLinearNode = operatorId === "linear"
    || (Object.values(node?.weight_shapes || {}).some((shape) => Array.isArray(shape) && shape.length >= 2)
      && !/(^|\.)(patch_)?embed/.test(String(node?.id || path)));
  const effectiveOperatorId = isLinearNode ? "linear" : operatorId;

  const entry = FORMULAS[effectiveOperatorId];
  const fromNode = entry?.fromNode;
  if (!fromNode) {
    return effectiveOperatorId
      ? null
      : { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
  }
  const ctx = fromNode({ node, config, options, path, bytesPerElement, operatorId: effectiveOperatorId, kind, vision, phase, tokens });
  if (ctx == null) return null;
  return typeof entry.counts === "function" ? entry.counts(ctx) : null;
}

for (const [id, fn] of Object.entries(FROM_NODE)) {
  if (FORMULAS[id]) FORMULAS[id].fromNode = fn;
}
