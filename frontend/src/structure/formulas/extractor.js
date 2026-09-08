// extractor.js —— node → counts ctx 提取器（docs/details/cost_counts.md 提取器规格）。
//
// 原则（principles §3.2 / §4.3）：
// - 查表优先：节点已有 weight_shapes / input_shape / output_shape / attributes；
//   提取器只补 phase 的 T/S、变体参数、expertFraction。
// - 分派只基于 type / attributes.operator_id / 结构化 attributes 与节点路径（结构化 id），
//   禁止显示名（node.name）参与分派。
// - 不 import cost 层；bytesPerElement 由调用方传入（W5 接线点）。
// - 返回单实例 counts；repeat 倍乘由 walker 的 multiplier 处理（与旧链同）。
// - 标注 "旧链镜像" 的分支：为通过差分而逐字复刻旧公式，W5 切换后随旧链一并删除。

import {
  linearCounts,
  attentionCounts,
  softmaxCounts,
  rmsnormCounts,
  gateCounts,
  swigluCounts,
  ropeCounts,
  causalConvCounts,
  linearAttentionStateCounts,
  topkCounts,
  moeDispatchCounts,
  moeCombineCounts,
  addCounts,
  hashRouteCounts,
  rearrangeCounts,
} from "./counts.js";
import { formulaForOperator } from "./index.js";
import { tensorDims } from "../model_executor/dims.js";
import { visionDimensions } from "../model_executor/layers/vision.js";
import { deriveBuildPlan } from "../model_executor/plan.js";
const planOf = (config) => deriveBuildPlan(config?.raw ?? config);

// 路径正则全仓统一处（旧 compute.js/parallel.js 三种变体收敛于此）
export const LAYER_INDEX_RE = /(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/;
export const ROUTED_EXPERT_RE = /(?:^|\.)(?<!shared_)(?:experts|expert_mlp)(?:\.|$)/;

/** 与旧 tokensFor 逐字等价：decode 1 token；vision 用 visionTokens。 */
export function tokensFor({ batch = 1, sequence = 1, phase = "prefill", vision = false, visionTokens = 1 } = {}) {
  return batch * (vision ? visionTokens : phase === "decode" ? 1 : sequence);
}

export function layerIndexOf(path) {
  const match = String(path || "").match(LAYER_INDEX_RE);
  return match ? Number(match[1]) : null;
}

/** 与旧 compute.js:309-312 逐字等价（routed expert 且该层非 dense 时按 k/E 缩放）。 */
export function expertFractionFor(path, config) {
  const layerIndex = layerIndexOf(path);
  const layerKind = layerIndex != null ? planOf(config).layerSchedule?.[layerIndex] : null;
  const routed = ROUTED_EXPERT_RE.test(String(path || ""));
  return routed && layerKind !== "dense" && config?.experts && config?.expertsPerToken
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

/** 线性逻辑形状：logical_weight_shape 属性优先，其次 weight_shapes 中首个 ≥2 维形状。
 *  packed（qweight）且无逻辑形状 → null（未知，沿用旧链诚实语义）。 */
function linearLogicalShape(node) {
  if (node?.weight_shapes?.qweight && !node?.attributes?.logical_weight_shape) return null;
  const logical = node?.attributes?.logical_weight_shape
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

// ---------- 旧链镜像（W5 切换后随旧链删除） ----------

// 旧 attentionMacs：batch·heads·lengthTerm·(D+dv)；不含 vision tokens（旧链现状）。
function legacyAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const lengthTerm = phase === "decode" ? sequence : sequence ** 2;
  return batch * heads * lengthTerm * (qk + value);
}

// 旧 qsaAttentionMacs。
function legacyQsaAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const selected = Math.min(sequence, config?.indexerBudget || sequence);
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  return queryTokens * heads * selected * (qk + value);
}

// 旧 minimaxSparseAttentionMacs。
function legacyMinimaxSparseAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  const selectedBlocks = (config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0);
  const selectedTokens = selectedBlocks * (config?.sparseBlockSize || 1);
  return queryTokens * heads * selectedTokens * (headDim + headDim);
}

// 旧 deepseekV4AttentionMacs（layerIndex 来自节点路径）。
function legacyDeepseekV4AttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill", layerIndex = 0 } = {}) {
  const heads = config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const ratio = config?.compressRatios?.[layerIndex] ?? 0;
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  const available = phase === "decode" ? 1 : sequence;
  const visible = ratio === 0
    ? Math.min(available, config?.slidingWindow || available)
    : ratio === 4
      ? Math.min(Math.ceil(available / ratio) + (config?.slidingWindow || 0), config?.indexerBudget || available)
      : Math.ceil(available / Math.max(ratio, 1));
  return queryTokens * heads * visible * (headDim + headDim);
}

// 旧 linearAttentionMacs（含各 mode 变体的逐字镜像）。
function legacyLinearAttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  if (planOf(config).linearAttentionMode === "glm5_next") return glm5NextLinearStateMacs(config, { batch, sequence, phase });
  if (planOf(config).linearAttentionMode === "kimi_k3") return kimiK3LinearStateMacs(config, { batch, sequence, phase });
  if (planOf(config).linearAttentionMode === "qwen4_exp") return qwen4ExpLinearStateMacs(config, { batch, sequence, phase });
  if (planOf(config).linearAttentionMode === "qwen3_5") return qwen35LinearStateMacs(config, { batch, sequence, phase });
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || 0;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  return tokens * (hidden * (keyHeads * keyDim + valueHeads * valueDim) + keyHeads * valueHeads * keyDim * valueDim);
}
// 旧 linearShortConvolutionMacs。
// 旧 linearAttentionDimensions。

function attentionCoreMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const lengthTerm = phase === "decode" ? sequence : sequence ** 2;
  return batch * heads * lengthTerm * (qk + value);
}
function linearAttentionCoreMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  if (planOf(config).linearAttentionMode === "glm5_next") return glm5NextLinearStateMacs(config, { batch, sequence, phase });
  if (planOf(config).linearAttentionMode === "kimi_k3") return kimiK3LinearStateMacs(config, { batch, sequence, phase });
  if (planOf(config).linearAttentionMode === "qwen4_exp") return qwen4ExpLinearStateMacs(config, { batch, sequence, phase });
  if (planOf(config).linearAttentionMode === "qwen3_5") return qwen35LinearStateMacs(config, { batch, sequence, phase });
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || 0;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  return tokens * (hidden * (keyHeads * keyDim + valueHeads * valueDim) + keyHeads * valueHeads * keyDim * valueDim);
}
function qsaCoreMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const qk = config?.headDim || 0;
  const value = config?.valueHeadDim || qk;
  const selected = Math.min(sequence, config?.indexerBudget || sequence);
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  return queryTokens * heads * selected * (qk + value);
}
function minimaxSparseCoreMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const heads = config?.attentionHeads || 0;
  const headDim = config?.headDim || 0;
  const queryTokens = batch * (phase === "decode" ? 1 : sequence);
  const selectedBlocks = (config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0);
  const selectedTokens = selectedBlocks * (config?.sparseBlockSize || 1);
  return queryTokens * heads * selectedTokens * (headDim + headDim);
}

function qwen35LinearStateMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const keyHeads = config?.linearKeyHeads || 0;
  const valueHeads = config?.linearValueHeads || 0;
  const keyDim = config?.linearKeyDim || 0;
  const valueDim = config?.linearValueDim || 0;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const convDim = 2 * keyProjection + valueProjection;
  const kernel = config?.linearConvKernelSize || 0;
  const qkvzProjection = hidden * (2 * keyProjection + 2 * valueProjection);
  const baProjection = 2 * hidden * valueHeads;
  const shortConvolution = convDim * kernel;
  const recurrentState = 3 * valueHeads * valueDim * keyDim;
  const gatedNorm = 3 * valueProjection;
  const outputProjection = valueProjection * hidden;
  return tokens * (qkvzProjection + baProjection + shortConvolution + recurrentState + gatedNorm + outputProjection);
}
function glm5NextLinearStateMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const heads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const headDim = config?.linearKeyDim || config?.headDim || 0;
  const projection = heads * headDim;
  const convKernel = config?.linearConvKernelSize || 0;
  const fusedProjection = hidden * (3 * projection + heads + 2 * headDim);
  const gateProjections = 2 * headDim * projection;
  const shortConvolution = 3 * projection * convKernel;
  const recurrentState = 3 * heads * headDim * headDim;
  const gatedNorm = 3 * projection;
  const outputProjection = projection * hidden;
  return tokens * (fusedProjection + gateProjections + shortConvolution + recurrentState + gatedNorm + outputProjection);
}
function kimiK3LinearStateMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const heads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const headDim = config?.linearKeyDim || config?.headDim || 0;
  const projection = heads * headDim;
  const convKernel = config?.linearConvKernelSize || 0;
  const fusedQkvg = hidden * 4 * projection;
  const betaProjection = hidden * heads;
  const decayProjection = hidden * headDim + headDim * projection;
  const shortConvolution = 3 * projection * convKernel;
  const recurrentState = 3 * heads * headDim * headDim;
  const gatedNorm = 3 * projection;
  const outputProjection = projection * hidden;
  return tokens * (fusedQkvg + betaProjection + decayProjection + shortConvolution + recurrentState + gatedNorm + outputProjection);
}
function qwen4ExpLinearStateMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const tokens = batch * (phase === "decode" ? 1 : sequence);
  const hidden = config?.hiddenSize || 0;
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || keyHeads;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const convDim = 2 * keyProjection + valueProjection;
  const kernel = config?.linearConvKernelSize || 0;
  const qkvzProjection = hidden * (2 * keyProjection + 2 * valueProjection);
  const baProjection = 2 * hidden * valueHeads;
  const shortConvolution = convDim * kernel;
  const recurrentState = 3 * valueHeads * valueDim * keyDim;
  const gatedNorm = 3 * valueProjection;
  const outputProjection = valueProjection * hidden;
  return tokens * (qkvzProjection + baProjection + shortConvolution + recurrentState + gatedNorm + outputProjection);
}

function linearAttentionDimensions(config = {}) {
  const keyHeads = config?.linearKeyHeads || config?.attentionHeads || 0;
  const valueHeads = config?.linearValueHeads || config?.attentionHeads || keyHeads;
  const keyDim = config?.linearKeyDim || config?.headDim || 0;
  const valueDim = config?.linearValueDim || config?.valueHeadDim || keyDim;
  return { keyHeads, valueHeads, keyDim, valueDim, keyProjection: keyHeads * keyDim, valueProjection: valueHeads * valueDim };
}
// 旧 linearStateUpdateMacs。
function linearStateUpdateMacs(config, { batch = 1, sequence = 1, phase = "prefill" } = {}) {
  const { keyHeads, valueHeads, keyDim, valueDim } = linearAttentionDimensions(config);
  const stateUpdate = planOf(config).linearAttentionMode === "generic"
    ? keyHeads * valueHeads * keyDim * valueDim
    : 3 * valueHeads * valueDim * keyDim;
  return batch * (phase === "decode" ? 1 : sequence) * stateUpdate;
}

/** M11-P0-5：KDA/线性注意力 state 的一阶访存——每次 forward 读+写一遍完整
 * 递归状态（递归矩阵 + conv 环形历史，与 cost/memory.js
 * linearStateElementsPerLayer 同源同式；形状来源：vLLM
 * MambaStateShapeCalculator.kda_state_shape）。状态驻留 HBM，chunk 内不逐
 * token 重读。 */
function stateUpdateCounts(config, options, bytesPerElement) {
  const { keyHeads, valueHeads, keyDim, valueDim } = linearAttentionDimensions(config);
  const kernel = Math.max(0, (config?.linearConvKernelSize || 1) - 1);
  const convElements = keyHeads * keyDim * 2 + valueHeads * valueDim;
  const recurrentElements = valueHeads * valueDim * keyDim;
  const stateBytes = (convElements * kernel + recurrentElements) * bytesPerElement;
  return {
    matrix: linearStateUpdateMacs(config, options),
    vector: 0,
    sfu: 0,
    bytes: { weights: 0, actIn: stateBytes, actOut: stateBytes },
  };
}
// ---------- 旧链镜像结束 ----------

/**
 * 计算单个算子节点的动作向量。
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

  // 注意力模块节点（type === "attention"）：own 值 = 注意力核心公式（投影由独立叶子计费）
  if (type === "attention") {
    const legacyOptions = { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase };
    let matrix = null;
    if (kind === "linear") matrix = linearAttentionCoreMacs(config, legacyOptions);
    else if (kind === "qsa") matrix = qsaCoreMacs(config, legacyOptions);
    else if (kind === "sparse" && config?.modelType === "minimax_m3_vl") matrix = minimaxSparseCoreMacs(config, legacyOptions);
    else if (kind === "dsv4") matrix = legacyDeepseekV4AttentionMacs(config, { ...legacyOptions, layerIndex: layerIndexOf(node?.id || path) ?? 0 });
    else matrix = attentionCoreMacs(config, legacyOptions);
    return { matrix, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
  }

  // M11 bytes 补齐：embedding gather 是真实访存（每 token 读一行权重、写一行
  // hidden），但它是无 operatorId 的结构节点，此前被"非算子零向量"规则计为
  // 零流量（bytes 完整性棘轮实测抓出）。gather 无 MACs，matrix 恒 0。
  if (type === "embedding") {
    const hidden = staticWidth(node?.output_shape) || config?.hiddenSize || 0;
    return {
      matrix: 0,
      vector: 0,
      sfu: 0,
      bytes: { weights: 0, actIn: tokens * hidden * bytesPerElement, actOut: tokens * hidden * bytesPerElement },
    };
  }

  // 旧 isLinear 等价：无 operatorId 但 weight_shapes 含 ≥2 维形状的节点按 linear 计
  // （checkpoint 绑定叶的常见形态）；embed 排除以结构化路径判断（§3.2）。
  const isLinearNode = operatorId === "linear"
    || (Object.values(node?.weight_shapes || {}).some((shape) => Array.isArray(shape) && shape.length >= 2)
      && !/(^|\.)(patch_)?embed/.test(String(node?.id || path)));
  // 旧 isLinear 的 weight-shapes 命中按 linear 分派（改写 operatorId，不递归）
  const effectiveOperatorId = isLinearNode ? "linear" : operatorId;

  switch (effectiveOperatorId) {
    case "linear": {
      // 与旧 isLinear 的 embed 排除等价：以结构化路径判断（node.name 不参与，§3.2）。
      // TODO(W3): builder 为 embed 投影声明结构化标记后移除路径判断。
      if (/(^|\.)(patch_)?embed/.test(String(node?.id || path))) {
        // M11-P0-5：embedding gather——每 token 读一行权重、写一行 hidden
        const hidden = staticWidth(node?.output_shape) || config?.hiddenSize || 0;
        return {
          matrix: 0,
          vector: 0,
          sfu: 0,
          bytes: { weights: 0, actIn: tokens * hidden * bytesPerElement, actOut: tokens * hidden * bytesPerElement },
        };
      }
      const expertFraction = expertFractionFor(node?.id || path, config);
      const logical = linearLogicalShape(node) || derivedLinearShape(node);
      if (!logical) return null;
      return linearCounts({ logicalShape: logical, tokens, bytesPerElement, expertFraction });
    }
    case "matmul": {
      // scores/context 用输出 shape 模式匹配区分（结构化判据，§3.2）。
      const patterns = attentionShapePatterns(config);
      const output = node?.output_shape;
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const queryTokens = tokens;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      // context 模式更具体（含具体 heads/value 维），必须先判；
      // scores 的全 -1 通配模式会吞掉一切 4D 输出。
      const part = patterns.context.some((pattern) => shapeMatchesPattern(output, pattern)) ? "context"
        : patterns.scores.some((pattern) => shapeMatchesPattern(output, pattern)) ? "scores"
        : null;
      if (part === "scores") {
        // M11-P0-5：一阶访存——读 Q、K，写 scores（此前恒 0，F2 KV 流量从未生效）
        return {
          matrix: queryTokens * heads * keyTokens * headDim,
          vector: 0,
          sfu: 0,
          bytes: {
            weights: 0,
            actIn: (queryTokens * heads * headDim + keyTokens * heads * headDim) * bytesPerElement,
            actOut: queryTokens * heads * keyTokens * bytesPerElement,
          },
        };
      }
      if (part === "context") {
        // 读 scores、V，写 context 输出
        return {
          matrix: queryTokens * heads * keyTokens * valueDim,
          vector: 0,
          sfu: 0,
          bytes: {
            weights: 0,
            actIn: (queryTokens * heads * keyTokens + keyTokens * heads * valueDim) * bytesPerElement,
            actOut: queryTokens * heads * valueDim * bytesPerElement,
          },
        };
      }
      return null;
    }
    case "qsa_attention": {
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      const selected = Math.min(keyTokens, config?.indexerBudget || keyTokens);
      // M11-P0-8（方案 A）：F2 整体访存（cost_counts.md F2 融合注意力行：
      // S=indexerBudget、kvHeads 按变体矩阵）。三种 attention_kind 共用本
      // case，读宽/共享度分派：
      // - dsa_sparse_mla（deepseek_v32 / glm_moe_dsa）：MLA latent 共享
      //   （kvHeads=1，K 读宽 kv_lora+rope、V 读宽 kv_lora，FlashMLA-sparse
      //   吸收式核的真实读宽）；新 token 的 latent cache 写回已由 kv_a_proj
      //   （linear actOut）计费 → 不加 kvWrite（防双计）。
      // - dsv4_sparse_mla（deepseek_v4 ratio=4）：config 无 kv_lora_rank →
      //   退 F2 MQA 行（kvHeads=1，config 实发 num_key_value_heads=1），读宽
      //   headDim/valueDim；压缩态写回由 compressor（mla_kv_compress 的
      //   F1 actOut）计费 → 不加 kvWrite。
      // - qsa（逐头 GQA/MHA 模板：qwen4_exp / glm5_next / kimi 预留）：K/V
      //   按实际 KV 头数读；paged cache 写回是模板内未计费的拷贝 → 计
      //   kvWrite（与 minimax_sparse_attention 同口径）。
      // scores/probs 按 A2 写+读各一次（稀疏模板无独立 softmax 叶，4·scores
      // 记此）；top-k 索引由 qsa_indexer 的 topk actOut 写、此处读
      // （tokens·selected，int32 按 2B 计）。取证：/tmp/m11-formulas/qsa.md
      //（16 模型探针明细 + 双计对账）。
      const scores = tokens * heads * selected;
      const context = tokens * heads * valueDim;
      const latentRead = kind !== "qsa" && (config?.kvLoraRank || 0) > 0;
      const kvHeads = latentRead ? 1 : config?.kvHeads || heads;
      const kWidth = latentRead ? (config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0) : headDim;
      const vWidth = latentRead ? (config?.kvLoraRank || 0) : valueDim;
      const kvWrite = latentRead ? 0 : kvHeads * tokens * (headDim + valueDim);
      return {
        matrix: tokens * heads * selected * (headDim + valueDim),
        vector: 0,
        sfu: 0,
        bytes: {
          weights: 0,
          actIn: (tokens * heads * headDim
            + kvHeads * selected * (kWidth + vWidth)
            + tokens * selected
            + 2 * scores) * bytesPerElement,
          actOut: (2 * scores + context + kvWrite) * bytesPerElement,
        },
      };
    }
    case "minimax_sparse_attention": {
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      void keyTokens;
      const selectedTokens = (config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0);
      const size = config?.sparseBlockSize || 1;
      const selected = selectedTokens * size;
      // M11 bytes 补齐（F2 口径，与 dense 分解链的 scores/softmax/context
      // 三节点合计同构）：Q 读 + 选中 KV 读 + scores/probs 中间量读写 +
      // O 写 + KV cache 写回（M3 稀疏注意力为融合算子，cache 写回在
      // attention 内部，dense 侧由 k/v_proj linear 的 actOut 计费）。
      // 依据：modeling_minimax_m3_vl.py（transformers 库版，HF 仓库无
      // modeling，取证件存 models/MiniMaxAI/MiniMax-M3/）+ config sparse_*；
      // 选块 per query token、per KV 组（index_heads=kv_heads）。
      const kvHeads = config?.kvHeads || heads;
      return {
        matrix: tokens * heads * selectedTokens * size * (headDim + valueDim),
        vector: 0,
        sfu: 0,
        bytes: {
          weights: 0,
          actIn: (heads * tokens * headDim
            + kvHeads * selected * (headDim + valueDim)
            + 2 * heads * tokens * selected) * bytesPerElement,
          actOut: (2 * heads * tokens * selected
            + heads * tokens * valueDim
            + kvHeads * tokens * (headDim + valueDim)) * bytesPerElement,
        },
      };
    }
    case "dsv4_swa_attention": {
      // M11 bytes 补齐：F2 一阶访存——MQA（num_key_value_heads=1）+ 滑窗。
      // matrix 维持 legacyDeepseekV4AttentionMacs 镜像（含 decode available=1
      // 的 legacy 行为，本波不动）。swa 缓存每 token 一份 headDim 宽的 KV
      // latent（K/V 共享，依据 memory.js dsv4 分支 + 权重表无 V 扩展投影），
      // 故 KV 读/写宽 = D 而非 2D。取证：/tmp/m11-formulas/dsv4.md
      // （V4 modeling 全网 404，config + 权重 index 实证，降级声明在案）。
      const batch = options.batch ?? 1;
      const sequence = options.sequence ?? 1;
      const layerIndex = layerIndexOf(node?.id || path) ?? 0;
      const ratio = config?.compressRatios?.[layerIndex] ?? 0;
      const heads = config?.attentionHeads || 0;
      const headDim = config?.headDim || 0;
      const valueDim = config?.valueHeadDim || headDim;
      const kvHeads = config?.kvHeads || 1;
      const queryTokens = batch * (phase === "decode" ? 1 : sequence);
      // 与 matrix 的 visible 同口径（decode 的 sequence 即上下文长度）
      const keyTokens = ratio === 0
        ? Math.min(sequence, config?.slidingWindow || sequence)
        : Math.ceil(sequence / Math.max(ratio, 1));
      const scores = heads * queryTokens * keyTokens;
      return {
        matrix: legacyDeepseekV4AttentionMacs(config, { batch, sequence, phase, layerIndex }),
        vector: 0,
        sfu: 0,
        bytes: {
          weights: 0,
          // Q 读 + KV 窗口 latent 读（一份）+ scores/probs 读写（2·scores，A2）
          actIn: (heads * queryTokens * headDim + kvHeads * keyTokens * headDim + 2 * scores) * bytesPerElement,
          // scores/probs（2·scores）+ context 写 + 新 token KV 写回 cache（T·kvH·D）
          actOut: (2 * scores + heads * queryTokens * valueDim + kvHeads * queryTokens * headDim) * bytesPerElement,
        },
      };
    }
    case "dsv4_compressed_attention": {
      // M11 bytes 补齐：同上，但读取对象是压缩缓存（每压缩位 K/V 态各
      // headDim，共 2·headDim，依据 ops 模板 compressor 输出 2·headDim +
      // memory.js (2·headDim)/ratio 摊销）。压缩态的写入由 compressor 叶
      // （mla_kv_compress）计费 → 本叶无 kvWrite，防双计。
      const batch = options.batch ?? 1;
      const sequence = options.sequence ?? 1;
      const layerIndex = layerIndexOf(node?.id || path) ?? 0;
      const ratio = config?.compressRatios?.[layerIndex] ?? 0;
      const heads = config?.attentionHeads || 0;
      const headDim = config?.headDim || 0;
      const valueDim = config?.valueHeadDim || headDim;
      const kvHeads = config?.kvHeads || 1;
      const queryTokens = batch * (phase === "decode" ? 1 : sequence);
      const keyTokens = ratio === 0
        ? Math.min(sequence, config?.slidingWindow || sequence)
        : Math.ceil(sequence / Math.max(ratio, 1));
      const scores = heads * queryTokens * keyTokens;
      return {
        matrix: legacyDeepseekV4AttentionMacs(config, { batch, sequence, phase, layerIndex }),
        vector: 0,
        sfu: 0,
        bytes: {
          weights: 0,
          actIn: (heads * queryTokens * headDim + 2 * kvHeads * keyTokens * headDim + 2 * scores) * bytesPerElement,
          actOut: (2 * scores + heads * queryTokens * valueDim) * bytesPerElement,
        },
      };
    }
    case "softmax": {
      const dims = attentionShapePatterns(config);
      void dims;
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const keyTokens = vision ? config?.visionTokens || 1 : phase === "decode" ? options.sequence ?? 1 : options.sequence ?? 1;
      const queryTokens = tokens;
      return softmaxCounts({ elements: heads * queryTokens * keyTokens, bytesPerElement });
    }
    case "rope": {
      const factor = node?.attributes?.partial_rotary_factor ?? config?.partialRotaryFactor ?? 1;
      return ropeCounts({ tokens, ropeDims: (config?.headDim || 0) * factor, bytesPerElement });
    }
    case "rmsnorm":
    case "gemma_rmsnorm":
      return rmsnormCounts({ tokens, hidden: staticWidth(node?.input_shape) || 0, bytesPerElement, weightOne: operatorId === "gemma_rmsnorm" });
    case "gated_rmsnorm":
      return rmsnormCounts({ tokens, hidden: staticWidth(node?.input_shape) || 0, bytesPerElement, gated: true });
    case "attention_output_gate":
    case "mla_output_gate":
    case "linear_attention_gate":
    case "shared_expert_gate":
      return gateCounts({ tokens, width: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_activation":
      return swigluCounts({ tokens, intermediate: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_position":
      return addCounts({ tokens, hidden: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_merge":
      return rearrangeCounts({ copy: true, inElements: staticWidth(node?.input_shape) || 0, outElements: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "split":
    case "mla_kv_split":
    case "qwen_qkvz_split":
    case "attention_qkv_split":
      // view 语义（strided view 无拷贝）：不产生独立流量，显式登记为零而非漏算。
      return rearrangeCounts();
    case "swiglu": {
      // 旧链：仅 expert 路径的 swiglu 计矩阵（gate/up/down GEMM 语义融合）；
      // 结构化判据用路径（专家目录），W3 换 attributes 标记。
      const idPath = String(node?.id || path);
      const routed = ROUTED_EXPERT_RE.test(idPath);
      if (!routed) return swigluCounts({ tokens, intermediate: staticWidth(node?.output_shape) || 0, bytesPerElement });
      const expertHidden = node?.attributes?.latent_size || config?.routedExpertHiddenSize || config?.hiddenSize || 0;
      const expertIntermediate = config?.moeIntermediateSize || config?.intermediateSize || 0;
      // 压缩的 routed FFN 叶（expert_mlp，无 per-expert repeat）：每 token 激活
      // k 个专家 → 正确计数 = T·k·3·EH·EI。旧链的 ·(k/E) 少乘 E（已知双链 bug）。
      // 若未来出现 per-expert 展开树（祖先 repeat=E），应改回 k/E 并依赖 walker 乘 E。
      const topk = config?.expertsPerToken || 1;
      // M11-P0-5：routed 分支 = GEMM（gate/up/down 融合）+ 逐元素激活两段。
      // 矩阵维持 T·k·3·EH·EI 公式；激活段（vector/sfu/bytes）走 F5 共享实现
      // 补齐流量（此前 bytes 恒 0）。tokens·k 与 per-expert 激活同构。
      const activation = swigluCounts({ tokens: tokens * topk, intermediate: expertIntermediate, bytesPerElement });
      return {
        matrix: tokens * 3 * expertHidden * expertIntermediate * topk,
        vector: activation.vector,
        sfu: activation.sfu,
        bytes: activation.bytes,
      };
    }
    case "causal_conv1d": {
      const { keyProjection, valueProjection } = linearAttentionDimensions(config);
      const kernel = config?.linearConvKernelSize || 0;
      // M11-P0-5：一阶访存——读输入窗口宽度、写同宽输出
      const width = 2 * keyProjection + valueProjection;
      return {
        matrix: tokens * width * kernel,
        vector: 0,
        sfu: 0,
        bytes: { weights: 0, actIn: tokens * width * bytesPerElement, actOut: tokens * width * bytesPerElement },
      };
    }
    case "linear_attention": {
      // 叶级 state/conv 用路径区分（旧链用显示名；W3 换结构化标记）。
      const idPath = String(node?.id || path);
      if (/short_conv|conv/.test(idPath)) {
        const { keyProjection, valueProjection } = linearAttentionDimensions(config);
        const kernel = config?.linearConvKernelSize || 0;
        const width = 2 * keyProjection + valueProjection;
        return {
          matrix: tokens * width * kernel,
          vector: 0,
          sfu: 0,
          bytes: { weights: 0, actIn: tokens * width * bytesPerElement, actOut: tokens * width * bytesPerElement },
        };
      }
      if (/state|recurrent/.test(idPath)) {
        return stateUpdateCounts(config, { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase }, bytesPerElement);
      }
      return { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
    }
    case "gated_delta_attention":
      return stateUpdateCounts(config, { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase }, bytesPerElement);
    case "topk":
      return topkCounts({ tokens, experts: config?.experts || 0, topk: config?.expertsPerToken || 0, bytesPerElement, normTopkProb: config?.normTopkProb ?? true });
    case "moe_dispatch":
      return moeDispatchCounts({ tokens, hidden: config?.hiddenSize || 0, topk: config?.expertsPerToken || 0, bytesPerElement });
    case "moe_combine":
      return moeCombineCounts({ tokens, hidden: config?.hiddenSize || 0, topk: config?.expertsPerToken || 0, bytesPerElement });
    case "moe_add":
      return addCounts({ tokens, hidden: staticWidth(node?.output_shape) || config?.hiddenSize || 0, bytesPerElement });
    case "dsv4_hash_route":
      // M11-P2：tid2eid 路由表 [vocab, num_experts_per_tok] 是真实参数
      // （V4-Flash 权重 index 实证：129280×6 ≈ 775,680 条目/层，此前传
      // tableRows:0 → weights 低估）。int32 索引按 bytesPerElement 计。
      return hashRouteCounts({
        tokens,
        topk: config?.expertsPerToken || 0,
        tableRows: (config?.vocabSize || 0) * (config?.expertsPerToken || 0),
        bytesPerElement,
      });
    default: {
      // 复合节点：调注册表的组合 counts（§3.1 唯一注册点），ctx 按规格构建。
      // 精度为初版（n 流参数用 normalized 近似），恒等式（T4）校准后复核。
      const entry = formulaForOperator(operatorId);
      // 无 operatorId 的结构节点（normalization/module 容器等）= 非算子，零向量；
      // 有 operatorId 但注册表未实现 = 真未实现算子 → null（unknownComputePaths）。
      if (typeof entry?.counts !== "function") {
        return operatorId ? null : { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
      }
      const H = config?.hiddenSize || 0;
      const ctxBuilders = {
        mla_query_compress: () => ({
          // 仅 q_a + norm：q_b 由独立 q_b_proj 叶计（组合里含 qb 会与叶双计，
          // 2026-09-07 审计发现 Kimi/GLM 各 +19M/+25M 参数每层）
          qa: { logicalShape: [config?.qLoraRank || 0, H], tokens, bytesPerElement },
          norm: { tokens, hidden: config?.qLoraRank || 0, bytesPerElement },
        }),
        mla_kv_compress: () => ({
          // out 维以节点自身 output_shape 为权威——模板已按家族声明
          // （MLA latent = kvLoraRank+qkRopeHeadDim；DSV4 压缩 = 2·headDim·k）。
          // config 组合仅作无形状回退。修复 V4 ctx 失配：V4 无 kvLoraRank，
          // 旧式得 out=64，探针实测 compressor macs 差 16-32×（2026-09-08）。
          proj: {
            logicalShape: [
              staticWidth(node?.output_shape) || (config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0),
              H,
            ],
            tokens,
            bytesPerElement,
          },
        }),
        qsa_indexer: () => ({
          score: { heads: config?.indexerNHeads || 0, queryTokens: tokens, keyTokens: options.sequence ?? 1, headDim: config?.indexerHeadDim || 0, valueDim: config?.indexerHeadDim || 0, bytesPerElement },
          topk: { tokens, experts: options.sequence ?? 1, topk: config?.indexerBudget || 0, bytesPerElement },
        }),
        minimax_sparse_indexer: () => ({
          score: { heads: config?.sparseIndexHeads || 0, queryTokens: tokens, keyTokens: options.sequence ?? 1, headDim: config?.sparseIndexDim || 0, valueDim: config?.sparseIndexDim || 0, bytesPerElement },
          topk: { tokens, experts: options.sequence ?? 1, topk: config?.sparseTopkBlocks || 0, bytesPerElement },
        }),
        attention_residual: () => ({
          norms: { tokens, hidden: H, bytesPerElement },
          scoreProj: { logicalShape: [1, H], tokens, bytesPerElement },
          aggregate: { elements: H * tokens, bytesPerElement },
          mix: { tokens, hidden: H, bytesPerElement },
        }),
        hyper_connection: () => ({
          grouped: { tokens, hidden: H, bytesPerElement },
          mix: { tokens, intermediate: H, bytesPerElement },
          mixers: { logicalShape: [H, H], tokens, bytesPerElement },
          gate: { tokens, width: H, bytesPerElement },
          combine: { tokens, hidden: H, bytesPerElement },
        }),
        ple: () => ({
          embed: { tokens, topk: 1, tableRows: 0, bytesPerElement },
          kv: { logicalShape: [2 * (config?.pleEmbedDim || 0), H], tokens, bytesPerElement },
          norm: { tokens, hidden: config?.pleEmbedDim || 0, bytesPerElement },
          conv: { tokens: tokens, channels: config?.pleEmbedDim || 0, kernel: config?.pleNgramSize || 1, bytesPerElement },
          add: { tokens, hidden: H, bytesPerElement },
        }),
        mhc_pre: () => ({
          mix: { tokens, width: H, bytesPerElement },
          matrix: { logicalShape: [H, config?.mhcNumResidualStreams || 1], tokens, bytesPerElement },
          merge: { tokens, hidden: H, bytesPerElement },
        }),
        mhc_post: () => ({
          combine: { logicalShape: [H, config?.mhcNumResidualStreams || 1], tokens, bytesPerElement },
          inject: { tokens, hidden: H, bytesPerElement },
        }),
        mhc_fused_post_pre: () => ({
          post: { tokens, width: H, bytesPerElement },
          inject: { tokens, hidden: H, bytesPerElement },
          pre: { tokens, width: H, bytesPerElement },
          matrix: { logicalShape: [H, config?.mhcNumResidualStreams || 1], tokens, bytesPerElement },
        }),
        mhc_contract: () => ({
          contract: { tokens, hidden: H, bytesPerElement },
        }),
      };
      const builder = ctxBuilders[operatorId];
      if (!builder) return null;
      return { ...entry.counts(builder()) };
    }
  }
}
