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
  linearAttentionStateCounts,
  topkCounts,
  moeDispatchCounts,
  moeCombineCounts,
  addCounts,
  hashRouteCounts,
  rearrangeCounts,
  scoredPairs,
  sinkhornCounts,
  fusedMoeMlpCounts,
} from "./counts.js";
import { paramBytes } from "./paramDtypes.js";
import { formulaForOperator } from "./index.js";
import { tensorDims } from "../config/dims.js";
import { visionDimensions } from "../config/visionDims.js";
import { deriveBuildPlan } from "../config/plan.js";
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

// ---------- 注意力/线性注意力的 matrix 权威实现（活代码，非旧链镜像） ----------
// W5 时本区是新旧双轨的"镜像"区；P0 单源化盘点后：4 个 legacy*Macos 包装
//（旧全量口径）无调用方已删除；余下函数全部被活 case 引用（:396-400 注意力
// 模块解析、dsv4 sparse case、KDA state case、mhc ctx）——它们是权威实现，
// "legacy" 字样仅保留在 deepseekV4 的历史命名里。

// deepseekV4AttentionMacs：DSV4 压缩/滑窗/滑窗预算的 matrix 权威实现（dsv4 sparse case 与注意力模块节点解析在用——"legacy" 前缀是历史遗留，非镜像）。
function deepseekV4AttentionMacs(config, { batch = 1, sequence = 1, phase = "prefill", layerIndex = 0 } = {}) {
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
// W3-⑤：chunked linear attention 的块长（显式近似假设，A6）。业界 chunked
// 实现默认 64；状态与 HBM 的交互次数 = ceil(T / CHUNK)，不是每 token 一次。
const LINEAR_ATTENTION_CHUNK = 64;

/** mHC 的混合行数 mix_hc = (2 + hc_mult)·hc_mult（vLLM deepseek_v4 model.py:711）。 */
function mhcMixRows(config) {
  const m = config?.mhcNumResidualStreams || 0;
  return (2 + m) * m;
}
/** mHC 的多流拼接宽 hc_dim = hc_mult·hidden（同上 :712）。 */
function mhcDim(config, hidden) {
  return (config?.mhcNumResidualStreams || 0) * hidden;
}

function stateUpdateCounts(config, options, bytesPerElement, modelKind = "") {
  const { keyHeads, valueHeads, keyDim, valueDim } = linearAttentionDimensions(config);
  const kernel = Math.max(0, (config?.linearConvKernelSize || 1) - 1);
  const convElements = keyHeads * keyDim * 2 + valueHeads * valueDim;
  const recurrentElements = valueHeads * valueDim * keyDim;
  const stateBytes = (convElements * kernel + recurrentElements) * bytesPerElement;
  // W3-⑤：状态读写次数分相位。此前恒为「一次前向读写一遍状态」，
  // prefill 下漏算了分块次数。chunked 实现每块与状态交互一次：
  //   prefill -> ceil(T / CHUNK)（CHUNK=64，业界 chunked linear attention 默认）
  //   decode  -> 每 token 一次（T=1 即一次，与旧口径一致）
  // 这是显式的近似执行形态假设（A6），记在保留容差清单里。
  const steps = options?.phase === "decode"
    ? Math.max(options?.batch ?? 1, 1)
    : Math.ceil(Math.max(options?.sequence ?? 1, 1) / LINEAR_ATTENTION_CHUNK);
  // W5：vector/sfu 不再恒零 —— 与 F7b（counts.js linearAttentionStateCounts）
  // 同口径：decay 的逐元素乘按 steps 计、每步每头 exp（delta 另加 beta sigmoid）。
  const { keyHeads: kh, valueHeads: vh, keyDim: kd, valueDim: vd } = linearAttentionDimensions(config);
  const recurrentState = (vh || kh || 1) * (kd || 0) * (vd || 0);
  const heads = vh || kh || 1;
  const delta = true; // KDA/GDN 全是 gated delta rule；plain 线性注意力走 linear_attention 分支
  // 递推核自带的标量参数（每次前向都要读一遍，此前记 0）。两族形状不同：
  //   qwen GDN：dt_bias 与 A_log 都是 num_v_heads
  //             （qwen_gdn_linear_attn.py:467-475）→ 2·heads
  //   GLM5-Next / K3 KDA：A_log 是 num_heads、dt_bias 是 projection_size
  //             （glm5next/nvidia/kda.py:205-243、kimi_k3/amd/kda.py:138-195）
  //             → heads + heads·valueDim
  // 判据用**字段存在性**：低秩 decay（f_b_proj）只在 KDA 族出现，配置上等价于
  // 「有 linear_key_head_dim 且 key/value 头数一致的融合 qkvbfg/qkvgfab 布局」。
  // 这里直接用 plan 的 linearAttentionMode（archs 显式登记，不是子串猜测）。
  // modelKind 来自节点自身的 `attributes.model_kind`（模板在发射 state_update 时
  // 声明，archs 显式登记的配方值，不是 model_type 子串猜测）。
  const kdaFamily = modelKind === "glm5_next" || modelKind === "kimi_k3";
  const gdnScalars = kdaFamily ? heads + heads * (vd || 0) : 2 * heads;
  return {
    matrix: linearStateUpdateMacs(config, options),
    vector: steps * recurrentState,
    sfu: steps * heads * (delta ? 3 : 1),
    // dt_bias / A_log 是 vLLM 显式声明的 torch.float32 参数（paramDtypes 登记），
    // 不跟激活字节宽 —— 与 derivedWeights 的 gdnDecayElements 同表锁步。
    bytes: { weights: gdnScalars * paramBytes("gdn_decay"), actIn: stateBytes * steps, actOut: stateBytes * steps },
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
    const macsOptions = { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase };
    let matrix = null;
    if (kind === "linear") matrix = linearAttentionCoreMacs(config, macsOptions);
    else if (kind === "qsa") matrix = qsaCoreMacs(config, macsOptions);
    else if (kind === "sparse" && config?.modelType === "minimax_m3_vl") matrix = minimaxSparseCoreMacs(config, macsOptions);
    else if (kind === "dsv4") matrix = deepseekV4AttentionMacs(config, { ...macsOptions, layerIndex: layerIndexOf(node?.id || path) ?? 0 });
    else matrix = attentionCoreMacs(config, macsOptions);
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
      // **只排文本 token 嵌入**（真查表，gather 无 MAC、不读权重矩阵）。
      // 原判据写成 `(patch_)?embed` 把**视觉 patch embedding 也当成查表**了 ——
      // 它是 Conv3d(in_ch, hidden, kernel=(T_p,P,P), stride=kernel, bias=False)
      //（vLLM qwen2_5_vl.py:548-560：view 后 conv 再 view，stride==kernel 即一次
      // GEMM [L, C·T_p·P²]×[C·T_p·P², hidden]），权重 = C·T_p·P²·hidden。
      // 实测 Qwen3.5-0.8B 因此少 1,179,648 参数（权重字节 0.9987）与
      // 576·1,179,648 MAC（整模型 matrix 0.9956），两条残差同源。
      if (/(^|\.)embed(_tokens)?$/.test(String(node?.id || path))) {
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
      return linearCounts({
        logicalShape: logical, tokens, bytesPerElement, expertFraction,
        bias: node?.attributes?.bias === true,
      });
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
      // W3-①因果：prefill 只算三角，decode 算全长（counts.js scoredPairs 唯一实现）。
      const pairs = heads * scoredPairs({ phase, queryTokens, keyTokens });
      // W3-②KV 读宽：此前 K/V 都按 **query 头数** 读，GQA/MQA/MLA 的共享完全没生效
      //（KV 下界报表实测 DeepSeek-V3.1 超读 71x、Kimi-K3 53x）。修正为：
      //   - 视觉塔 ViT 是 MHA → kvHeads = heads
      //   - MLA/DSA（latent 共享）→ kvHeads=1，K 读宽 kv_lora+rope、V 读宽 kv_lora
      //     （cache 里存的就是 latent，非内核选择）
      //   - 其余 → config.kvHeads
      const latentShared = !vision && kind.includes("mla") && (config?.kvLoraRank || 0) > 0;
      const kvHeads = vision ? heads : (latentShared ? 1 : (config?.kvHeads || heads));
      const kReadWidth = latentShared ? (config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0) : headDim;
      const vReadWidth = latentShared ? (config?.kvLoraRank || 0) : valueDim;
      if (part === "scores") {
        // M11-P0-5：一阶访存——读 Q、K，写 scores（此前恒 0，F2 KV 流量从未生效）
        // W6：`kvRead` 单列 —— 它是 actIn 里**从 KV cache 读的那部分**（不含 Q、
        // 不含 scores 中间量）。KV 读恒等式只能拿这一项跟 cache 容量口径比，
        // 拿 actIn 总量比就只能留松量（原 30%）。kvRead ⊆ actIn，不额外累加。
        const kvRead = keyTokens * kvHeads * kReadWidth * bytesPerElement;
        return {
          matrix: pairs * headDim,
          vector: 0,
          sfu: 0,
          bytes: {
            weights: 0,
            actIn: (queryTokens * heads * headDim + keyTokens * kvHeads * kReadWidth) * bytesPerElement,
            actOut: pairs * bytesPerElement,
            kvRead,
          },
        };
      }
      if (part === "context") {
        // 读 scores、V，写 context 输出。
        // W5 防双计：MLA/DSA 的 K 与 V 是**同一份 latent**（cache 里只存
        // kv_lora+rope 一份），scores 叶已经把它整份流过一遍，context 叶再读
        // 一次就是同一批 cache line 读两遍。KV 读下界报表实测这条让 MLA 系
        // 模型（V3.1 2.12x / K3 2.06x / K2 系 2.01x）整体超读约 2 倍。
        // 非 latent 共享的 GQA/MHA 里 K 与 V 是两个独立张量，照旧各读一次。
        const vRead = latentShared ? 0 : keyTokens * kvHeads * vReadWidth;
        return {
          matrix: pairs * valueDim,
          vector: 0,
          sfu: 0,
          bytes: {
            weights: 0,
            actIn: (pairs + vRead) * bytesPerElement,
            actOut: queryTokens * heads * valueDim * bytesPerElement,
            kvRead: vRead * bytesPerElement,
          },
        };
      }
      return null;
    }
    case "sdpa_attention": {
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const queryTokens = tokens;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      const latentShared = !vision && kind.includes("mla") && (config?.kvLoraRank || 0) > 0;
      const kvHeads = vision ? heads : (latentShared ? 1 : (config?.kvHeads || heads));
      const kReadWidth = latentShared ? (config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0) : headDim;
      const vReadWidth = latentShared ? (config?.kvLoraRank || 0) : valueDim;
      const fused = attentionCounts({
        heads, queryTokens, keyTokens, headDim, valueDim, bytesPerElement, kvHeads, phase,
      });
      // 核边界：scores 不落 HBM（§2.4）。MLA 的 K/V 是同一份 latent，读宽取 max。
      const q = heads * queryTokens * headDim;
      const kRead = kvHeads * keyTokens * kReadWidth;
      const vRead = latentShared ? 0 : kvHeads * keyTokens * vReadWidth;
      const context = heads * queryTokens * valueDim;
      const kvWrite = latentShared ? 0 : kvHeads * queryTokens * (headDim + valueDim);
      fused.bytes = {
        weights: 0,
        actIn: (q + kRead + vRead) * bytesPerElement,
        actOut: (context + kvWrite) * bytesPerElement,
        kvRead: (kRead + vRead) * bytesPerElement,
      };
      return fused;
    }
    case "qsa_sparse_attention":
    // W2：单一 qsa_attention 条目拆成三个 operator_id（算法出处不同不共用条目）。
    // 三者的 counts 仍共用本 case，读宽/共享度按 attention_kind 分派。
    case "dsa_sparse_mla":
    case "dsv4_sparse_mla": {
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      const budget = (effectiveOperatorId === "qsa_sparse_attention"
        ? config?.qsaIndexerBudget
        : config?.dsaIndexTopk) ?? config?.indexerBudget ?? keyTokens;
      const selected = Math.min(keyTokens, budget || keyTokens);
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
      // - qsa（逐头 GQA/MHA 模板：qwen4_exp，Qwen4ExpTextQSAIndexer 实证）：
      //   K/V 按实际 KV 头数读；paged cache 写回是模板内未计费的拷贝 → 计
      //   kvWrite（与 minimax_sparse_attention 同口径）。glm5_next 曾列于此
      //   2026-09-08 证据改判为 DSA（见 qsaAttentionOperatorSpecs）。
      // scores/probs 按 A2 写+读各一次（稀疏模板无独立 softmax 叶，4·scores
      // 记此）；top-k 索引由 qsa_indexer 的 topk actOut 写、此处读
      // （tokens·selected，int32 按 2B 计）。取证：/tmp/m11-formulas/qsa.md
      //（16 模型探针明细 + 双计对账）。
      const scores = heads * scoredPairs({ phase, queryTokens: tokens, keyTokens: selected });
      const context = tokens * heads * valueDim;
      const latentRead = kind !== "qsa" && (config?.kvLoraRank || 0) > 0;
      const kvHeads = latentRead ? 1 : config?.kvHeads || heads;
      const kWidth = latentRead ? (config?.kvLoraRank || 0) + (config?.qkRopeHeadDim || 0) : headDim;
      const vWidth = latentRead ? (config?.kvLoraRank || 0) : valueDim;
      const kvWrite = latentRead || kind === "dsv4_sparse_mla"
        ? 0  // C4 压缩态写入归 compressor 叶、窗口写入已单列（fc99269），防三重计费
        : kvHeads * tokens * (headDim + valueDim);
      // M11 滑窗补记：dsv4_sparse_mla（C4，ratio=4）层与 compressed 层同款
      // 混合读——滑窗 [t-128,t] 全层覆盖（memory.js 容量口径已含），逐头
      // qsa/qwen4_exp 无此窗口。
      const dsv4Window = kind === "dsv4_sparse_mla"
        ? kvHeads * Math.min(options.sequence || 1, config?.slidingWindow || 128) * headDim
        : 0;
      return {
        matrix: scores * (headDim + valueDim),
        vector: 0,
        sfu: 0,
        bytes: {
          weights: 0,
          // W5 防双计：latentRead 时 K/V 是同一份 latent（cache 只存一份），
          // 读宽取 max(kWidth, vWidth) 而非相加 —— 与 dense 分解链里
          // scores/context 两叶的同一处修正对齐（KV 读下界报表实证）。
          actIn: (tokens * heads * headDim
            + kvHeads * selected * (latentRead ? Math.max(kWidth, vWidth) : kWidth + vWidth)
            + tokens * selected
            + dsv4Window
            + 2 * scores) * bytesPerElement,
          actOut: (2 * scores + context + kvWrite + dsv4Window) * bytesPerElement,
          // W6：cache 读的那部分（选中的 KV + dsv4 的原始滑窗），不含 Q / top-k
          // 索引 / scores 中间量。
          kvRead: (kvHeads * selected * (latentRead ? Math.max(kWidth, vWidth) : kWidth + vWidth)
            + dsv4Window) * bytesPerElement,
        },
      };
    }
    case "minimax_sparse_attention": {
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const headDim = vision ? config?.visionHeadDim || 0 : config?.headDim || 0;
      const valueDim = vision ? headDim : config?.valueHeadDim || headDim;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence || 1;
      const selectedTokens = (config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0);
      const size = config?.sparseBlockSize || 1;
      // W3：选中块的 token 总数必须夹到实际可见长度——上下文短于块预算时
      // （如 S=128 而 17 块 x 128 = 2176）不存在那么多 key，否则打分对数虚高。
      // 与 qsa/dsa/dsv4 三个 sparse case 的 Math.min 口径对齐。
      const selected = Math.min(keyTokens, selectedTokens * size);
      // M11 bytes 补齐（F2 口径，与 dense 分解链的 scores/softmax/context
      // 三节点合计同构）：Q 读 + 选中 KV 读 + scores/probs 中间量读写 +
      // O 写 + KV cache 写回（M3 稀疏注意力为融合算子，cache 写回在
      // attention 内部，dense 侧由 k/v_proj linear 的 actOut 计费）。
      // 依据：modeling_minimax_m3_vl.py（transformers 库版，HF 仓库无
      // modeling，取证件存 models/MiniMaxAI/MiniMax-M3/）+ config sparse_*；
      // 选块 per query token、per KV 组（index_heads=kv_heads）。
      const kvHeads = config?.kvHeads || heads;
      // W3-①因果：块稀疏同样只在实际可见的位置上打分（prefill 三角、decode 全长）
      const scores = heads * scoredPairs({ phase, queryTokens: tokens, keyTokens: selected });
      return {
        matrix: scores * (headDim + valueDim),
        vector: 0,
        sfu: 0,
        bytes: {
          weights: 0,
          actIn: (heads * tokens * headDim
            + kvHeads * selected * (headDim + valueDim)
            + 2 * scores) * bytesPerElement,
          actOut: (2 * scores
            + heads * tokens * valueDim
            + kvHeads * tokens * (headDim + valueDim)) * bytesPerElement,
          // W6：cache 读那部分 = 选中块的 K/V。
          kvRead: kvHeads * selected * (headDim + valueDim) * bytesPerElement,
        },
      };
    }
    case "dsv4_swa_attention": {
      // M11 bytes 补齐：F2 一阶访存——MQA（num_key_value_heads=1）+ 滑窗。
      // matrix 维持 deepseekV4AttentionMacs 镜像（含 decode available=1
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
      // W3-①因果：滑窗/压缩层的打分对数同样分相位
      const scores = heads * scoredPairs({ phase, queryTokens, keyTokens });
      return {
        matrix: scores * (headDim + valueDim),
        vector: 0,
        sfu: 0,
        bytes: {
          weights: 0,
          // Q 读 + KV 窗口 latent 读（一份）+ scores/probs 读写（2·scores，A2）
          actIn: (heads * queryTokens * headDim + kvHeads * keyTokens * headDim + 2 * scores) * bytesPerElement,
          // scores/probs（2·scores）+ context 写 + 新 token KV 写回 cache（T·kvH·D）
          actOut: (2 * scores + heads * queryTokens * valueDim + kvHeads * queryTokens * headDim) * bytesPerElement,
          // W6：cache 读那部分 = 窗口内（或压缩后）的 KV latent。
          kvRead: kvHeads * keyTokens * headDim * bytesPerElement,
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
      // M11 滑窗补记（vLLM c128a = 压缩历史 + 原始滑窗 [t-128,t] 混合读，
      // /tmp/m11-formulas/dsv4-sliding-window.md 裁决①）：compressed 层还读
      // 一份未压缩滑窗 KV；新 token 的窗口写入与 swa 层同口径。
      const windowTokens = Math.min(sequence, config?.slidingWindow || 128);
      return {
        matrix: deepseekV4AttentionMacs(config, { batch, sequence, phase, layerIndex }),
        vector: 0,
        sfu: 0,
        bytes: {
          weights: 0,
          actIn: (heads * queryTokens * headDim + 2 * kvHeads * keyTokens * headDim + kvHeads * windowTokens * headDim + 2 * scores) * bytesPerElement,
          actOut: (2 * scores + heads * queryTokens * valueDim + kvHeads * queryTokens * headDim) * bytesPerElement,
          // W6：cache 读那部分 = 压缩历史（K/V 各一份）+ 未压缩滑窗。
          kvRead: (2 * kvHeads * keyTokens * headDim + kvHeads * windowTokens * headDim) * bytesPerElement,
        },
      };
    }
    case "softmax": {
      const dims = attentionShapePatterns(config);
      void dims;
      const heads = vision ? config?.visionAttentionHeads || 0 : config?.attentionHeads || 0;
      const keyTokens = vision ? config?.visionTokens || 1 : options.sequence ?? 1;
      const queryTokens = tokens;
      // W3-①因果：softmax 只作用在实际打分的位置上，与 scores/context 同口径。
      return softmaxCounts({ elements: heads * scoredPairs({ phase, queryTokens, keyTokens }), bytesPerElement });
    }
    case "rope": {
      const factor = node?.attributes?.partial_rotary_factor ?? config?.partialRotaryFactor ?? 1;
      // W5：原来只传单头 head_dim，等于只算了一个头的 rope。实际 q 的全部头与
      // k 的全部 kv 头都要旋转 → 每 token 元素数 = (heads + kvHeads)·D_rope。
      // 视觉塔按 ViT 的 MHA（kvHeads = heads）。
      const ropeHeads = vision
        ? 2 * (config?.visionAttentionHeads || 0)
        : (config?.attentionHeads || 0) + (config?.kvHeads || config?.attentionHeads || 0);
      const ropeDim = (vision ? config?.visionHeadDim || 0 : config?.headDim || 0) * factor;
      return ropeCounts({ tokens, ropeDims: ropeHeads * ropeDim, bytesPerElement });
    }
    case "rmsnorm":
    case "gemma_rmsnorm":
      return rmsnormCounts({
        tokens, hidden: staticWidth(node?.input_shape) || 0, bytesPerElement,
        weightOne: operatorId === "gemma_rmsnorm",
        weightWidth: normWeightWidth(node?.input_shape) || undefined,
        affineBias: node?.attributes?.affine_bias === true,
      });
    case "gated_rmsnorm":
      return rmsnormCounts({
        tokens, hidden: staticWidth(node?.input_shape) || 0, bytesPerElement, gated: true,
        weightWidth: normWeightWidth(node?.input_shape) || undefined,
      });
    case "attention_output_gate":
    case "mla_output_gate":
    case "linear_attention_gate":
    case "shared_expert_gate":
      return gateCounts({ tokens, width: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_activation":
      return swigluCounts({ tokens, intermediate: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_position":
      return addCounts({ tokens, hidden: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "vision_merge": {
      // G1 缺口补齐（2026-09-09）：此前 inElements/outElements 是**单 token 宽度**，
      // 漏乘 token 数。patch merge 是 [V, H] → [V/merge², merge²·H] 的真实拷贝，
      // 两端总元素数相等（V·H），必须各乘自己的 token 数。
      const inWidth = staticWidth(node?.input_shape) || 0;
      const outWidth = staticWidth(node?.output_shape) || 0;
      const mergeSize = Math.max(config?.visionMergeSize || 1, 1);
      const outTokens = Math.max(1, Math.floor(tokens / (mergeSize * mergeSize)));
      return rearrangeCounts({
        copy: true,
        inElements: inWidth * tokens,
        outElements: outWidth * outTokens,
        bytesPerElement,
      });
    }
    case "split":
    case "mla_kv_split":
    case "qwen_qkvz_split":
    case "attention_qkv_split":
      // view 语义（strided view 无拷贝）：不产生独立流量，显式登记为零而非漏算。
      return rearrangeCounts();
    case "swiglu":
      // 纯激活（SiluAndMul）。N2-4 W-A：携带专家 GEMM 的路由专家叶已拆出独立
      // id `fused_moe_mlp`（对标 vLLM FusedMoE），本条只服务 dense/vision 的
      // 门控激活，矩阵恒 0。
      return swigluCounts({ tokens, intermediate: staticWidth(node?.output_shape) || 0, bytesPerElement });
    case "fused_moe_mlp": {
      // N2-4 W-A：MoE 路由专家融合叶（gate/up/down GEMM + SwiGLU 激活），分派
      // 只看 operator_id（W3 的「换 attributes 标记」到此落地——路径正则
      // ROUTED_EXPERT_RE 只剩 expertFractionFor 与 parallel.js 的无声明回退在用）。
      // EH 取 latent_size（K3 潜空间）→ routedExpertHiddenSize → hiddenSize，
      // 与 builder 侧 routedExpertWeightMatrices 的声明同源（锚 1 逐叶对账）。
      const expertHidden = node?.attributes?.latent_size || config?.routedExpertHiddenSize || config?.hiddenSize || 0;
      const expertIntermediate = config?.moeIntermediateSize || config?.intermediateSize || 0;
      return fusedMoeMlpCounts({
        tokens,
        topk: config?.expertsPerToken || 1,
        experts: config?.experts || 0,
        expertHidden,
        expertIntermediate,
        bytesPerElement,
      });
    }
    case "causal_conv1d": {
      const { keyProjection, valueProjection } = linearAttentionDimensions(config);
      const kernel = config?.linearConvKernelSize || 0;
      // M11-P0-5：一阶访存——读输入窗口宽度、写同宽输出
      const width = 2 * keyProjection + valueProjection;
      // W3-⑥：补两项此前恒 0 的分量。
      // (a) 卷积核权重每次前向读一遍（registry F7a 一直声明有，运行时丢了——
      //     文档 G2 登记的双轨差之一）。depthwise：width x kernel。
      // (b) decode 相位独有的 conv state：每步要读回前 kernel-1 个 token 的
      //     通道值并写回滚动窗口。prefill 的窗口在片上滑动，不额外落 HBM。
      const convStateElements = phase === "decode" ? width * Math.max(kernel - 1, 0) : 0;
      return {
        matrix: tokens * width * kernel,
        vector: 0,
        sfu: 0,
        bytes: {
          weights: width * kernel * bytesPerElement,
          actIn: (tokens * width + convStateElements) * bytesPerElement,
          actOut: (tokens * width + convStateElements) * bytesPerElement,
        },
      };
    }
    case "linear_attention": {
      // 叶级 state/conv 用路径区分（旧链用显示名；W3 换结构化标记）。
      const idPath = String(node?.id || path);
      if (/short_conv|conv/.test(idPath)) {
        const { keyProjection, valueProjection } = linearAttentionDimensions(config);
        const kernel = config?.linearConvKernelSize || 0;
        const width = 2 * keyProjection + valueProjection;
        // linear_attention 是 0/59 的通用保留槽位（GDA 家族走 gated_delta_attention
        // /causal_conv1d）：本分支 weights=0 为占位口径——若未来家族的 plain
        // linear-attention 带独立卷积核，权重读在此计并按 P2 判据补声明。
        return {
          matrix: tokens * width * kernel,
          vector: 0,
          sfu: 0,
          bytes: { weights: 0, actIn: tokens * width * bytesPerElement, actOut: tokens * width * bytesPerElement },
        };
      }
      if (/state|recurrent/.test(idPath)) {
        return stateUpdateCounts(config, { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase }, bytesPerElement, String(node?.attributes?.model_kind || ""));
      }
      return { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
    }
    case "gated_delta_attention":
      return stateUpdateCounts(config, { batch: options.batch ?? 1, sequence: options.sequence ?? 1, phase }, bytesPerElement, String(node?.attributes?.model_kind || ""));
    case "topk":
      return topkCounts({ tokens, experts: config?.experts || 0, topk: config?.expertsPerToken || 0, bytesPerElement, normTopkProb: config?.normTopkProb ?? true });
    case "moe_dispatch":
      return moeDispatchCounts({ tokens, hidden: staticWidth(node?.input_shape) || config?.hiddenSize || 0, topk: config?.expertsPerToken || 0, bytesPerElement });
    case "moe_combine":
      return moeCombineCounts({ tokens, hidden: staticWidth(node?.input_shape) || config?.hiddenSize || 0, topk: config?.expertsPerToken || 0, bytesPerElement });
    case "residual_add":
      // W4：每层两处残差加。hidden 取输出宽（与 moe_add 同口径）。
      return addCounts({ tokens, hidden: staticWidth(node?.output_shape) || config?.hiddenSize || 0, bytesPerElement });
    case "identity":
      return rearrangeCounts({ copy: false });
    case "moe_add":
      return addCounts({ tokens, hidden: staticWidth(node?.output_shape) || config?.hiddenSize || 0, bytesPerElement });
    case "dsv4_hash_route":
      // 2026-09-09 分类裁决（取代 M11-P2 的「表算参数」口径）：tid2eid 是
      // **buffer 不是参数**（Megatron-Bridge："Buffers are not parameters"；
      // MaxText 同；出处 Hash Layers, Roller et al. 2021）。表的常驻容量
      // （vocab·k·4B int32）由 derivedBufferBytes 计入显存，不进权重字节；
      // 本叶只计 gather 的真实拷贝流量。
      return hashRouteCounts({
        tokens,
        topk: config?.expertsPerToken || 0,
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
        // W2：四种 indexer 各自一个 operator_id，共用 sparseIndexerCounts 的
        // 参数化实现。参数矩阵见 modules.js sparseIndexerCounts 的 doc。
        qsa_indexer: () => ({
          heads: config?.qsaIndexerHeads ?? config?.indexerNHeads ?? 0,
          dim: config?.qsaIndexerHeadDim ?? config?.indexerHeadDim ?? 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: config?.qsaIndexerBudget ?? config?.indexerBudget ?? 0,
          pool: config?.qsaIndexerCompressRatio ?? 1,
          poolStage: (config?.qsaIndexerCompressRatio ?? 1) > 1 ? "key" : "none",
          perHeadWeights: false,
          phase,
          b: bytesPerElement,
        }),
        dsa_indexer: () => ({
          heads: config?.dsaIndexHeads ?? config?.indexerNHeads ?? 0,
          dim: config?.dsaIndexHeadDim ?? config?.indexerHeadDim ?? 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: config?.dsaIndexTopk ?? config?.indexerBudget ?? 0,
          pool: 1,
          poolStage: "none",
          perHeadWeights: true,
          phase,
          b: bytesPerElement,
        }),
        dsa_kpool_indexer: () => ({
          heads: config?.dsaIndexHeads ?? config?.indexerNHeads ?? 0,
          dim: config?.dsaIndexHeadDim ?? config?.indexerHeadDim ?? 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: config?.dsaIndexTopk ?? config?.indexerBudget ?? 0,
          pool: config?.dsaIndexKpool ?? 1,
          poolStage: "key",
          perHeadWeights: true,
          phase,
          b: bytesPerElement,
        }),
        dsv4_indexer: () => ({
          heads: config?.dsaIndexHeads ?? config?.indexerNHeads ?? 0,
          dim: config?.dsaIndexHeadDim ?? config?.indexerHeadDim ?? 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          budget: config?.dsaIndexTopk ?? config?.indexerBudget ?? 0,
          pool: 1,
          poolStage: "none",
          perHeadWeights: true,
          phase,
          b: bytesPerElement,
        }),
        minimax_sparse_indexer: () => ({
          heads: config?.sparseIndexHeads || 0,
          dim: config?.sparseIndexDim || 0,
          queryTokens: tokens,
          keyTokens: options.sequence ?? 1,
          // 预算按 token 计：选中块数 x 块大小（含 init/local 常驻块）
          budget: ((config?.sparseTopkBlocks || 0) + (config?.sparseInitBlock || 0) + (config?.sparseLocalBlock || 0)) * (config?.sparseBlockSize || 1),
          pool: config?.sparseBlockSize || 1,
          poolStage: "score",
          perHeadWeights: false,
          phase,
          b: bytesPerElement,
        }),
        attention_residual: () => ({
          norms: { tokens, hidden: H, bytesPerElement },
          scoreProj: { logicalShape: [1, H], tokens, bytesPerElement },
          aggregate: { elements: H * tokens, bytesPerElement },
          mix: { tokens, hidden: H, bytesPerElement },
        }),
        hyper_connection: () => {
          // 形状全部来自 vLLM GatedResidual（hyperconnection.py:140-193）：
          // hyper_hidden = hc_count·hidden；norm 覆盖整个 HC×H 布局。
          const streams = config?.hyperConnectionCount || 1;
          const lowrank = config?.hyperConnectionLowrank || 0;
          const hyperHidden = streams * H;
          return {
            grouped: { tokens, hidden: hyperHidden, weightOne: true, bytesPerElement },
            mixDown: { logicalShape: [lowrank, hyperHidden], tokens, bytesPerElement },
            silu: { tokens, width: lowrank, bytesPerElement },
            mixUp: { logicalShape: [hyperHidden, lowrank], tokens, bytesPerElement },
            gate: { tokens, width: hyperHidden, bytesPerElement },
            // use_combine=false 的相位（最终 mixer）没有 block_inject_weight。
            inject: node?.attributes?.hc_use_combine === false
              ? { logicalShape: [streams, hyperHidden], tokens: 0, bytesPerElement, weightsShared: true }
              : { logicalShape: [streams, hyperHidden], tokens, bytesPerElement },
            combine: { tokens, hidden: hyperHidden, bytesPerElement },
          };
        },
        ple: () => ({
          embed: { tokens, topk: 1, bytesPerElement },
          kv: { logicalShape: [2 * (config?.pleEmbedDim || 0), H], tokens, bytesPerElement },
          norm: { tokens, hidden: config?.pleEmbedDim || 0, bytesPerElement },
          conv: { tokens: tokens, channels: config?.pleEmbedDim || 0, kernel: config?.pleNgramSize || 1, bytesPerElement },
          add: { tokens, hidden: H, bytesPerElement },
        }),
        // mHC 的混合矩阵形状取自 vLLM deepseek_v4/amd/model.py:709-752：
        //   mix_hc = (2 + hc_mult)·hc_mult   hc_dim = hc_mult·hidden
        //   hc_{attn,ffn}_fn   : [mix_hc, hc_dim]   （attn 侧归 mhc_pre、
        //   hc_{attn,ffn}_base : [mix_hc]            ffn 侧归 mhc_fused_post_pre）
        //   hc_{attn,ffn}_scale: [3]
        // 此前两处都写成 [H, hc_mult]（28,672），比真值 mix_hc·hc_dim 小 24 倍
        //（V4-Pro 每层少 1,318,910 参数，2026-09-09 权重字节逐层归因抓出）。
        // 注意：上游这些张量是 fp32；本工具统一按激活字节宽计，dtype 差异单列登记。
        mhc_pre: () => ({
          mix: { tokens, width: H, bytesPerElement },
          // hc_{attn,ffn}_fn [mix_hc, hc_dim] 与 base/scale 标量都是 vLLM 显式
          // 声明的 torch.float32（paramDtypes 登记），权重字节宽 4、激活仍跟
          // bytesPerElement（linearCounts 的 weightBytesPerElement）。
          matrix: { logicalShape: [mhcMixRows(config), mhcDim(config, H)], tokens, bytesPerElement, weightBytesPerElement: paramBytes("mhc_fn") },
          base: { logicalShape: [mhcMixRows(config), 1], tokens: 0, bytesPerElement: paramBytes("mhc_base") },
          scale: { logicalShape: [3, 1], tokens: 0, bytesPerElement: paramBytes("mhc_scale") },
          // attn_norm 的 RMSNorm 权重被融进 mhc_pre 内核（vLLM
          // deepseek_v4/amd/model.py:704、816-818 把 attn_norm.weight 传进去），
          // 结构树里没有独立的 input_layernorm 叶 —— 权重记在这里。
          norm: { tokens, hidden: H, bytesPerElement },
          // comb/Sinkhorn 段（scale[2] 分支）：hc_mult×hc_mult tile 逐 token
          // 的 softmax + hc_sinkhorn_iters-1 轮行/列归一化（kernel 取证 2026-09-09）。
          sinkhorn: { tokens, streams: config?.mhcNumResidualStreams || 0, iterations: config?.mhcSinkhornIterations || 0, bytesPerElement },
          merge: { tokens, hidden: H, bytesPerElement },
        }),
        mhc_post: () => ({
          // 最终的 hc_post 复用**最后一层**的 hc_ffn_* 参数再算一遍，然后对
          // hc_mult 条流取均值（vLLM deepseek_v4/amd/model.py:1074-1097：
          // `layer.hc_post(...)` + `.mean(dim=-2)`），没有自己的参数。
          // weightsShared=true：算力照计、权重字节不重复计。
          combine: { logicalShape: [mhcMixRows(config), mhcDim(config, H)], tokens, bytesPerElement, weightsShared: true },
          inject: { tokens, hidden: H, bytesPerElement },
        }),
        mhc_fused_post_pre: () => ({
          post: { tokens, width: H, bytesPerElement },
          inject: { tokens, hidden: H, bytesPerElement },
          pre: { tokens, width: H, bytesPerElement },
          matrix: { logicalShape: [mhcMixRows(config), mhcDim(config, H)], tokens, bytesPerElement, weightBytesPerElement: paramBytes("mhc_fn") },
          base: { logicalShape: [mhcMixRows(config), 1], tokens: 0, bytesPerElement: paramBytes("mhc_base") },
          scale: { logicalShape: [3, 1], tokens: 0, bytesPerElement: paramBytes("mhc_scale") },
          // 同理，ffn_norm 的权重融进 fused post+pre（model.py:705）。
          norm: { tokens, hidden: H, bytesPerElement },
          // fused 的 pre 半同样含 comb/Sinkhorn（scale[1] ffn 分支）
          sinkhorn: { tokens, streams: config?.mhcNumResidualStreams || 0, iterations: config?.mhcSinkhornIterations || 0, bytesPerElement },
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
