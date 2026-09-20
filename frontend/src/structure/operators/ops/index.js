import { formulaForOperator } from "../formulas/index.js";
import { shapeFlow, shapesAndDims } from "../shapes.js";
import { tensorDims } from "../../config/dims.js";
import { indexerScheduleOf } from "../../layers/schedule.js";
import { recipeAttentionOutputGate, recipeFlag, recipeLinearAttentionMode, recipeValue } from "../../archs/index.js";

function cleanAttributes(attributes) {
  return Object.fromEntries(
    Object.entries(attributes).filter(([, value]) => value !== undefined && value !== null),
  );
}

export function operatorSpec(id, name, operatorId, attributes = {}, numericShapes = {}) {
  const formula = formulaForOperator(operatorId);
  return {
    kind: "operator",
    id,
    name,
    operatorId,
    input_shape: numericShapes.input,
    output_shape: numericShapes.output,
    attributes: cleanAttributes({
      formula: formula?.formula,
      explanation: formula?.explanation,
      inputs: formula?.inputs,
      outputs: formula?.outputs,
      ...attributes,
      // P4：归一化族与线性族的权重声明由本工厂按形状自动产出（约 60 处调用点不
      // 逐一手写，出处 = sharding_matrix.md「统一助手按现有形状自动产出」）。
      // 显式传 weightMatrices 的调用点覆盖自动值。
      ...normWeightMatrices(operatorId, numericShapes.input, attributes),
      ...linearWeightMatrices(operatorId, id, numericShapes, attributes),
    }),
  };
}

// ---------------------------------------------------------------------------
// N2-4 W-A（docs/details/sharding_matrix.md 层 1）：weightMatrices 权重声明。
//
// 根因是「每个权重矩阵的 [out, in, 数量, 精度, 分片亲和]」没有单一住址——散在
// counts 闭式公式、nodeWeightBytes 的 weight_shapes、quantBytes 路径匹配、
// parallel.js 路径正则四处互不一致的载体里。层 1 把它种进叶子 attributes：
//
//   weightMatrices: [{ class, out, in, count, matrices }, ...]
//
//   - 每组 = 共享同一量化处理与分片亲和的矩阵集合；class ∈ tp | ep | vocab |
//     replicated（与 parallel.js 的分片轴对应；dtype 不进声明——未量化参数走
//     paramDtypes，量化字节由层 2 消费者按 quant 方案计算）；
//   - count × matrices × out × in = 该组全部元素，与叶 counts.bytes.weights
//     逐位可对账（锚 1，modelIdentities.test.js；权重字节恒等式已锚定叶
//     counts，因此声明写错立即红）；
//   - out/in 从与 numericShapes 同源的 dims 取正维宽度，声明与形状不会漂移；
//   - 无声明的叶子走 parallel.js 规则表回退（行为逐位不变），声明逐步覆盖。
// ---------------------------------------------------------------------------

/** dims 数组的正维宽度（-1/null = 自由/未知维，不参与乘积；与 extractor 的 staticWidth 同口径）。 */
function dimWidth(d) {
  return Array.isArray(d)
    ? d.filter((value) => Number.isFinite(value) && value > 0).reduce((total, value) => total * value, 1)
    : 0;
}

/**
 * 一组权重声明（schema v2，2026-09-10 用户裁决：切分维度建模是真需求 + 可读性
 * 优先 + 量化建模对齐成熟方案，不自己造轮子）：
 *
 *   { class, shape, count, matrices, split, quantizable, param_dtype }
 *
 * - **shape**：张量形状数组（safetensors/state_dict 的成熟表示——向量 [heads]、
 *   卷积核 [width, kernel]、矩阵 [out, in]）。out = shape[0]、in = 其余维乘积
 *  （||1）为派生字段，供现有消费者（sharding/aggregate/锚 1）零改动使用；
 * - **split**：切分维度，成熟命名取自 vLLM 的并行类一一对应——
 *   `MergedColumnParallelLinear`（gate_up/qkv，沿 output 切）→ "output"，
 *   `RowParallelLinear`（down/o_proj，沿 input 切）→ "input"，
 *   replicated（router/norm/向量参数）→ null。出处：vLLM `linear.py`
 *   `create_weights` 的 `ModelWeightParameter(input_dim=1, output_dim=0)` ——
 *   切分轴是权重参数的一级属性；当前消费者只算 ÷tp 总量比例，split 是为
 *   维度级建模（w1/w3 列切、w2 行切）预留的一级字段，不改变现有行为；
 * - **quantizable**：量化方案只作用于 Linear 权重矩阵（HF quantization_config
 *   规范 targets: ["Linear"]，vLLM/SGLang 同）。norm scale/bias/衰减参数等
 *   非 Linear 参数显式标 false；
 * - **param_dtype**：引用 `formulas/paramDtypes.js` 的 FP32_PARAMS 键（不携带
 *   字节数——dtype 知识仍单源在登记表），供锚 1 的 dtype-aware 判据使用。
 */
export function weightMatrixDecl(klass, { shape, out, in: inDim, count = 1, matrices = 1, split = null, quantizable = true, param_dtype = undefined, shared = false }) {
  let resolvedOut = out;
  let resolvedIn = inDim;
  if (Array.isArray(shape)) {
    resolvedOut = shape[0];
    resolvedIn = shape.slice(1).reduce((total, value) => total * value, 1) || 1;
  }
  const group = { class: klass, out: resolvedOut, in: resolvedIn, count, matrices };
  if (Array.isArray(shape)) group.shape = shape;
  if (split) group.split = split;
  if (!quantizable) group.quantizable = false;
  if (param_dtype) group.param_dtype = param_dtype;
  if (shared) group.shared = true;
  return group;
}

// 归一化族：权重是**最后一维**那么长、跨其余维度共享（RMSNorm 沿最后一维归一）。
// 逐头 norm（q_norm/k_norm = RMSNorm(head_dim)、GDN 输出门 = RMSNormGated(head_v_dim)）
// 因此不能用正维乘积，否则放大 heads 倍 —— 与 formulas/extractor.js 的
// normWeightWidth 同判据（该函数是 counts 侧的同一知识，此处不能各写一遍公式，
// 但两侧都从同一个 input_shape 取最后一维，声明与 counts 逐位可对账，锚 1 执法）。
const NORM_OPS = new Set(["rmsnorm", "gemma_rmsnorm", "gated_rmsnorm"]);

/** 归一化族叶的自动声明（replicated：norm 权重每卡各持一份，不切）。 */
function normWeightMatrices(operatorId, inputShape, attributes) {
  if (!NORM_OPS.has(operatorId) || attributes?.weightMatrices) return null;
  const width = Array.isArray(inputShape)
    ? [...inputShape].reverse().find((value) => Number.isFinite(value) && value > 0)
    : undefined;
  if (!width) return null;
  // LayerNorm（affine_bias）有 bias，权重 2×宽度 —— 与 rmsnormCounts 同口径。
  const matrices = attributes?.affine_bias === true ? 2 : 1;
  // norm 的 scale/bias 不是 Linear 权重，量化方案不作用于它（HF quantization_config
  // targets: ["Linear"]，vLLM/SGLang 同）。
  return { weightMatrices: [weightMatrixDecl("replicated", { shape: [width], matrices, quantizable: false })] };
}

// 线性族：ColumnParallel/RowParallel 都按 tp 切，唯一例外是 lm_head/embed（vocab
// 轴，受 vocabParallel 开关支配）与 MoE router（每卡各算一份完整门控，vLLM/SGLang
// 的 gate 不切 —— replicated）。判据用**结构化 id 末段**，不用 display name（§3.2）。
// split 轴与 vLLM 的并行类一一对应（qwen3_moe.py:97,104,289）：gate_up/qkv =
// MergedColumnParallelLinear → "output"；down/o_proj = RowParallelLinear → "input"。
const VOCAB_LINEAR = /(^|\.)(lm_head|output)(\.linear)?$/;
const REPLICATED_LINEAR = /(^|\.)(router|hash_router|main_proj|confidence_head|markov_w2|eh_proj|e_proj|h_proj)$/;
const OUTPUT_SPLIT_LINEAR = /(^|\.)(gate_proj|up_proj|gate_up|q_proj|k_proj|v_proj|qkv_proj|qkvz_proj|in_proj_qkvb|w13|fc_embedding|fc_hidden|fc)$/;
const INPUT_SPLIT_LINEAR = /(^|\.)(down_proj|o_proj|out_proj|w2)$/;

/** 线性叶的自动声明：out/in 取正维乘积（与 extractor 的 derivedLinearShape 同口径）。 */
function linearWeightMatrices(operatorId, id, numericShapes, attributes) {
  if (operatorId !== "linear" || attributes?.weightMatrices) return null;
  // 文本 token 嵌入走 embedding 类型节点，不经本工厂；此处只处理 linear 叶。
  const out = dimWidth(numericShapes?.output);
  const inDim = dimWidth(numericShapes?.input);
  if (!(out > 0) || !(inDim > 0)) return null;
  const path = String(id || "");
  const klass = VOCAB_LINEAR.test(path) ? "vocab" : REPLICATED_LINEAR.test(path) ? "replicated" : "tp";
  const split = klass !== "tp" ? null
    : OUTPUT_SPLIT_LINEAR.test(path) ? "output"
    : INPUT_SPLIT_LINEAR.test(path) ? "input"
    : null;
  // bias 也是权重（out 个），与 linearCounts 同口径 —— 用第二组 [out] 表达；
  // bias 不参与量化（quant config targets: ["Linear"] 只覆盖权重矩阵）。
  const groups = [weightMatrixDecl(klass, { shape: [out, inDim], split })];
  if (attributes?.bias === true) groups.push(weightMatrixDecl(klass, { shape: [out], quantizable: false }));
  return { weightMatrices: groups };
}

/**
 * MoE 路由专家叶的声明：gate/up/down 三矩阵全在叶内（对标 vLLM FusedMoE 打包
 * w13/w2——模块自描述权重），ep 亲和（÷moe_ep，不均衡区间见 expertWeightRange）。
 * EH 的取值链与 extractor 的 fused_moe_mlp case 同源（latent_size →
 * routedExpertHiddenSize → hiddenSize；K3 潜空间为 latent）。
 */
export function routedExpertWeightMatrices(normalized) {
  const expertHidden = normalized.routedExpertHiddenSize || normalized.hiddenSize;
  const expertIntermediate = normalized.moeIntermediateSize || normalized.intermediateSize;
  return [weightMatrixDecl("ep", { shape: [expertIntermediate, expertHidden], count: normalized.experts, matrices: 3, split: "output" })];
}

// SDPA 核（原则 §2.3 / §2.4）：QKᵀ / softmax / PV。不含 RoPE、不含 q/k/v/o 投影。
// FlashAttention 是这个核的实现，写进 implementation。默认折叠；展开才看到三叶。
/** 每 token 驻留 cache 元素（容量，不是这次 forward 的 kvRead）。
 *  ref: vLLM AttentionSpec / MLAAttentionSpec / CompressorStateCache。 */
/** 每 token 驻留 cache 元素（容量，不是这次 forward 的 kvRead）。
 *  ref: vLLM AttentionSpec / MLAAttentionSpec / CompressorStateCache。
 *  - kvElements/indexElements = **全驻留**（含有界滑窗），W5 capacity↔kvRead 对账用，语义不变。
 *  - 带 kvDtype 时额外声明 dsv4 的**边际 + 逐 dtype**口径：growthKvElements/growthIndexElements =
 *    随 token 线性增长的压缩 KV/index（排除有界滑窗），配 cache_kv_dtype/cache_index_dtype（F4/F8）。
 *    仅 residentMemoryFromGraph 的每 token 报告消费；无 kvDtype 的叶（非 dsv4/合成图）行为不变。 */
export function cacheResidentDecl({
  kvElements = 0, indexElements = 0, growthKvElements, growthIndexElements, kvDtype, indexDtype,
} = {}) {
  const out = { cache_kv_elements: kvElements, cache_index_elements: indexElements };
  if (kvDtype) {
    out.cache_kv_dtype = kvDtype;
    out.cache_kv_growth_elements = growthKvElements ?? 0;
    out.cache_index_growth_elements = growthIndexElements ?? 0;
    out.cache_index_dtype = indexDtype ?? kvDtype;
  }
  return out;
}

/** SGLang temporal(ssm) state dtype：默认 fp32；config.mamba_ssm_dtype 可覆盖为 bf16/fp16。 */
function normalizeSsmStateDtype(dtype) {
  const s = String(dtype || "").toLowerCase().replace(/^torch\./, "");
  if (s === "bfloat16" || s === "bf16") return "BF16";
  if (s === "float16" || s === "fp16" || s === "f16") return "F16";
  return "F32"; // 缺省 fp32（vLLM/SGLang 默认；SGLang mamba2_state_dtype temporal 默认 fp32）
}

/** KDA/GDN request state 元素。ref: vLLM MambaStateShapeCalculator.kda_state_shape。
 *  SGLang `mamba2_state_dtype`：conv 恒 bf16(2B)、temporal(recurrent/ssm) 默认 fp32(4B)。
 *  memory lens 按 conv/recurrent 分量逐 dtype 计（Bug2 修复：此前统一 bf16 低估线性 state ~48%）；
 *  state_elements 保留总量给 parallel.js / W5 / 向后兼容。config 覆盖经 normalized.mambaSsmDtype。 */
export function linearStateResidentDecl(normalized) {
  const keyHeads = normalized.linearKeyHeads || normalized.attentionHeads || 0;
  const valueHeads = normalized.linearValueHeads || normalized.attentionHeads || 0;
  const keyDim = normalized.linearKeyDim || normalized.headDim || 0;
  const valueDim = normalized.linearValueDim || normalized.valueHeadDim || keyDim;
  const kernel = Math.max(0, (normalized.linearConvKernelSize || 1) - 1);
  const convElements = keyHeads * keyDim * 2 + valueHeads * valueDim;
  const recurrentElements = valueHeads * valueDim * keyDim;
  const convStateElements = convElements * kernel;
  return {
    state_elements: convStateElements + recurrentElements,
    state_conv_elements: convStateElements,
    state_recurrent_elements: recurrentElements,
    state_recurrent_dtype: normalizeSsmStateDtype(normalized.mambaSsmDtype),
  };
}

export function sdpaAttentionModule(prefix, shapes, dims, { scoresName = "attention scores", scores = {}, context = {}, modality } = {}) {
  const sdpaId = `${prefix}.sdpa`;
  const formula = formulaForOperator("sdpa_attention");
  const children = [
    operatorSpec(`${sdpaId}.scores`, scoresName, "matmul", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, shapes.attentionScores),
      formula: "S = Q K^T / sqrt(d)",
      modality,
      ...scores,
    }, { input: dims.attentionQuery, output: dims.attentionScores }),
    operatorSpec(`${sdpaId}.softmax`, "attention probabilities", "softmax", {
      ...shapeFlow(shapes.attentionScores, shapes.attentionProbabilities),
      modality,
    }, { input: dims.attentionScores, output: dims.attentionProbabilities }),
    operatorSpec(`${sdpaId}.context`, "weighted value", "matmul", {
      ...shapeFlow(`${shapes.attentionProbabilities}, ${shapes.attentionValue}`, shapes.attentionContext),
      formula: "O = P V",
      modality,
      ...context,
    }, { input: dims.attentionProbabilities, output: dims.attentionContext }),
  ];
  return {
    kind: "module",
    id: sdpaId,
    name: "SDPA attention",
    type: "operator",
    attributes: cleanAttributes({
      class: "SDPA",
      operator_id: "sdpa_attention",
      formula: formula?.formula,
      explanation: formula?.explanation,
      inputs: formula?.inputs,
      outputs: formula?.outputs,
      attention_kind: scores.attention_kind || context.attention_kind,
      ...(scores.cacheResident || {}),
      implementation: ["vLLM.Attention", "SGLang.FlashAttentionBackend", "TRT-LLM.GPTAttention"],
      dataflow_edges: [["scores", "softmax"], ["softmax", "context"]],
      modality,
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}, ${shapes.attentionValue}`, shapes.attentionContext),
    }),
    children,
    input_shape: dims.attentionQuery,
    output_shape: dims.attentionContext,
  };
}

// 打分式注意力的公共尾链：rope → SDPA 核 → o_proj。
// 五处调用（GQA / qwen35Full / MLA / minimaxCommon dense / minimaxM2）的节点结构
// 与数值 shape 完全一致，差异全部落在 attributes：
//   rope: { query_shape, key_shape, position_shape, rotary_dim, partial_rotary_factor, implementation }
//   scores: { attention_kind, formula 覆写, name 覆写（MLA "latent attention scores"）, query/key_shape, explanation 等 }
//   context: { attention_kind, explanation 等 }
//   preOutput: 插在 SDPA 核与 o_proj 之间的节点（MLA 的 g_proj）
//   before: 插在 tail 之前的节点（minimax sparse 的 indexer 链）
// 真语义不同的变体（dsa、dsv4、qsa）不并入本 helper。
function scaledDotProductTail(prefix, shapes, dims, { ropeName = "rotary position embedding", rope = {}, scoresName = "attention scores", scores = {}, context = {}, preOutput = [], before = [], cacheResident } = {}) {
  return [
    ...before,
    operatorSpec(`${prefix}.rope`, ropeName, "rope", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, `${shapes.attentionQuery}, ${shapes.attentionKey}`),
      ...rope,
    }, { input: dims.attentionQuery, output: dims.attentionQuery }),
    sdpaAttentionModule(prefix, shapes, dims, { scoresName, scores: { ...scores, cacheResident }, context }),
    ...preOutput,
    operatorSpec(`${prefix}.o_proj`, "output projection", "linear", {
      ...shapeFlow(shapes.attentionContext, shapes.hidden),
      communication_role: "tp_attention_output",
      // N2-4 层 1：RowParallel 输出投影（vLLM RowParallelLinear，沿 input 切）。
      // 声明与 numericShapes 同源（dims 取宽），五处 tail 调用自动跟随。
      weightMatrices: [weightMatrixDecl("tp", { shape: [dimWidth(dims.hidden), dimWidth(dims.attentionContext)], split: "input" })],
    }, { input: dims.attentionContext, output: dims.hidden }),
  ];
}

export function attentionOperatorSpecs(prefix, attentionKind, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  const kvHeads = normalized.kvHeads || normalized.attentionHeads || 0;
  const headDim = normalized.headDim || 0;
  const valueDim = normalized.valueHeadDim || headDim;
  return [
    operatorSpec(`${prefix}.q_proj`, "q projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.attentionQuery),
      weightMatrices: [weightMatrixDecl("tp", { shape: [dimWidth(dims.attentionQuery), dimWidth(dims.hidden)], split: "output" })],
    }, { input: dims.hidden, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.k_proj`, "k projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.attentionKey),
      weightMatrices: [weightMatrixDecl("tp", { shape: [dimWidth(dims.attentionKey), dimWidth(dims.hidden)], split: "output" })],
    }, { input: dims.hidden, output: dims.attentionKey }),
    operatorSpec(`${prefix}.v_proj`, "v projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.attentionValue),
      weightMatrices: [weightMatrixDecl("tp", { shape: [dimWidth(dims.attentionValue), dimWidth(dims.hidden)], split: "output" })],
    }, { input: dims.hidden, output: dims.attentionValue }),
    ...scaledDotProductTail(prefix, shapes, dims, {
      rope: {
        query_shape: shapes.attentionQuery,
        key_shape: shapes.attentionKey,
        position_shape: "[batch, sequence]",
      },
      scores: {
        explanation: "用旋转后的 Q 与 K^T 计算注意力分数。",
        inputs: ["Q", "K"],
        outputs: ["S"],
        attention_kind: attentionKind,
        query_shape: shapes.attentionQuery,
        key_shape: shapes.attentionKey,
      },
      context: {
        explanation: "用注意力概率 P 对 V 做加权聚合。",
        inputs: ["probabilities", "V"],
        outputs: ["O"],
        probabilities_shape: shapes.attentionProbabilities,
        value_shape: shapes.attentionValue,
      },
      cacheResident: cacheResidentDecl({ kvElements: 2 * kvHeads * headDim }),
    }),
  ];
}

// KDA 各模型变体的 linearAttentionMode 值即 canonicalKdaOperatorSpecs 的 modelKind
const KDA_LINEAR_MODES = new Set(["kimi_k3", "kimi", "glm5_next", "qwen4_exp", "qwen3_5"]);

export function linearAttentionOperatorSpecs(prefix, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  const linearMode = recipeLinearAttentionMode(normalized);
  if (KDA_LINEAR_MODES.has(linearMode)) {
    return canonicalKdaOperatorSpecs(prefix, normalized, linearMode);
  }
  return [
    operatorSpec(`${prefix}.in_proj_qkv`, "linear attention qkv projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.in_proj_z`, "linear attention gate projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.in_proj_b`, "linear attention decay projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.short_conv`, "short convolution", "linear_attention", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.state_update`, "linear attention state update", "linear_attention", {
      ...shapeFlow(`${shapes.hidden}, state`, shapes.hidden),
      attention_kind: "linear",
      ...linearStateResidentDecl(normalized),
    }, { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.output_gate`, "linear attention output gate", "linear_attention_gate", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", { ...shapeFlow(shapes.hidden, shapes.hidden), communication_role: "tp_attention_output" }, { input: dims.hidden, output: dims.hidden }),
  ];
}

// KDA is one semantic structure. Framework-specific fused projections remain
// in attributes so vLLM/SGLang implementation details do not duplicate nodes.
function canonicalKdaOperatorSpecs(prefix, normalized, modelKind) {
  const { shapes, dims } = shapesAndDims(normalized);
  const keyHeads = normalized.linearKeyHeads || normalized.attentionHeads || 0;
  const valueHeads = normalized.linearValueHeads || normalized.attentionHeads || keyHeads;
  const keyDim = normalized.linearKeyDim || normalized.headDim || 0;
  const valueDim = normalized.linearValueDim || normalized.valueHeadDim || keyDim;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const qwen = modelKind === "qwen4_exp" || modelKind === "qwen3_5";
  const qkvFlat = qwen ? 2 * keyProjection + 2 * valueProjection : 2 * keyProjection + valueProjection;
  const qkvConvFlat = 2 * keyProjection + valueProjection;
  const qkvShape = qwen
    ? `[batch, sequence, Q/K=${keyHeads}x${keyDim}, V=${valueHeads}x${valueDim}]`
    : `[batch, sequence, linear heads=${keyHeads}, head dimension=${keyDim}]`;
  const fusedFlat = modelKind === "kimi_k3"
    ? 4 * keyProjection + keyDim + keyHeads
    : modelKind === "glm5_next"
      ? 3 * keyProjection + keyHeads + 2 * keyDim
      : qkvFlat;
  const fusedShape = modelKind === "kimi_k3"
    ? `[batch, sequence, fused qkvgfab=${fusedFlat}]`
    : modelKind === "glm5_next"
      ? `[batch, sequence, fused qkvbfg_a=${fusedFlat}]`
      : qwen
        ? `[batch, sequence, fused qkvz=${qkvFlat}]`
        : qkvShape;
  const convShape = qwen
    ? qkvShape
    : `[batch, sequence, qkv channels=${qkvConvFlat}]`;
  const betaShape = `[batch, sequence, value heads=${valueHeads}]`;
  const gateShape = qwen
    ? `[batch, sequence, value heads=${valueHeads}, value dimension=${valueDim}]`
    : qkvShape;
  const stateShape = `[batch, value heads=${valueHeads}, state value dimension=${valueDim}, state key dimension=${keyDim}]`;
  const qkvDims = qwen ? [-1, -1, qkvFlat] : [-1, -1, fusedFlat];
  const qkvConvDims = qwen ? [-1, -1, qkvConvFlat] : [-1, -1, 3 * keyProjection];
  // W3.5 修正：非 qwen 的 KDA，state_update 的输出宽是 value 投影宽
  // （valueHeads·valueDim），**不是**融合输入宽 fusedFlat。此前 glm5_next 的
  // output_gate_norm / out_proj 都按 fusedFlat=24896 计，out_proj 单层多算
  // 6.84e7 元素 x 34 层 = 2.33e9。kimi_k3 早有特判、glm5_next 漏了，现统一。
  // state 的输出是**逐头**的 [.., valueHeads, valueDim]（qwen 与非 qwen 同形；
  // 摊平写法只是同一张量的另一种视图，但会让下游逐头 norm/投影的形状连续性断裂）。
  const outputDims = [-1, -1, valueHeads, valueDim];
  // gated 输出归一化是**逐头**的：`FusedRMSNormGated(self.head_dim)`
  //（vLLM kimi_gdn_linear_attn.py:304、qwen_gdn_linear_attn.py:487-488），
  // 权重只有 valueDim 那么长。声明成四维，normWeightWidth 才能取到最后一维；
  // 摊平成 [-1,-1,valueProjection] 会把权重放大 valueHeads 倍
  //（K3 每层 12,288 而真值 128，2026-09-09 权重字节逐层归因抓出）。
  const gateNormDims = [-1, -1, valueHeads, valueDim];
  const betaDims = [-1, -1, valueHeads];
  const fullRank = modelKind === "kimi_k3";
  const implementation = fullRank
    ? {
      input_projection: "in_proj_qkvgfab",
      beta_projection: "b_proj",
      decay_projection: ["f_a_proj", "f_b_proj"],
      short_convolution: "conv1d",
      output_gate: "in_proj_qkvgfab.g",
    }
    : qwen
      ? {
        input_projection: "in_proj_qkvz",
        beta_projection: "in_proj_ba.b",
        decay_projection: "in_proj_ba.a",
        short_convolution: "conv1d",
        output_gate: "in_proj_qkvz.z",
      }
      : {
      input_projection: "in_proj_qkvbfg_a",
      beta_projection: "in_proj_qkvbfg_a.beta",
      decay_projection: ["in_proj_qkvbfg_a.f_a", "f_b_proj"],
      short_convolution: ["q_conv1d", "k_conv1d", "v_conv1d"],
      output_gate: ["in_proj_qkvbfg_a.g_a", "g_b_proj"],
    };
  const projectionLayout = fullRank ? ["q", "k", "v", "g", "f_a", "beta"] : qwen ? ["q", "k", "v", "z"] : ["q", "k", "v", "beta", "f_a", "g_a"];
  const specs = [
    operatorSpec(`${prefix}.qkv_projection`, "QKV projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      semantic_role: "q_k_v_projection",
      implementation,
      projection_size: qwen ? { qk: keyProjection, v: valueProjection, z: valueProjection } : keyProjection,
      fused_projection_width: fusedFlat,
      fused_projection_layout: projectionLayout,
    }, { input: dims.hidden, output: qkvDims }),
    ...(qwen ? [operatorSpec(`${prefix}.qkvz_split`, "qkvz split", "qwen_qkvz_split", {
      ...shapeFlow(`[batch, sequence, fused qkvz=${qkvFlat}]`, `${qkvShape}, ${qkvShape}, ${qkvShape}, ${gateShape}`),
      split_sizes: [keyProjection, keyProjection, valueProjection, valueProjection],
      implementation: ["vLLM.QwenGatedDeltaNetAttention.fix_query_key_value_ordering", "SGLang.Qwen3_5GatedDeltaNet.fix_query_key_value_ordering"],
    }, { input: qkvDims, output: qkvConvDims })] : []),
    // beta（delta rule 的步长）**只有 qwen GDN 才是独立 GEMM**（in_proj_ba.b）。
    // GLM5-Next 与 K3 的 b 已经在融合投影里：
    //   glm5next/nvidia/kda.py:179-196 in_proj_qkvbfg_a = [q,k,v,b,f_a,g_a]
    //   kimi_k3/amd/kda.py:110-127     in_proj_qkvgfab  = [q,k,v,g,f_a,b]
    // 再发一片独立 beta 叶就是双计（GLM 每层 262,144、K3 每层 688,128 —— 2026-09-09
    // 权重字节逐层归因抓出）。
    ...(qwen ? [operatorSpec(`${prefix}.beta_projection`, "beta projection", "linear", {
      ...shapeFlow(shapes.hidden, betaShape),
      semantic_role: "delta_beta",
      implementation: implementation.beta_projection,
      activation: "sigmoid_in_kda_kernel",
    }, { input: dims.hidden, output: betaDims })] : []),
    // M8-V2（源码 modeling_kimi_linear.py）：kimi_k3 的 decay 走低秩
    // f_a（在融合 qkvgfab 内）+ f_b（独立 head_dim→projection_size）——
    // 独立全宽 decay 叶会与融合内 f_a 重复计数（88M vs 真值 2.5M/层）。
    // W3.5 修正：**glm5_next 同为低秩**（modeling_glm5_next.py 取证：融合
    // qkvbfg_a 已含 b/f_a/g_a，独立叶只有 f_b、g_b，各 head_dim→qkv_dim）。
    // 此前只给 kimi_k3 特判，glm5_next 仍发全宽 hidden×qkv_dim decay 叶，
    // 单层多算 3.2e7 元素 x 34 层 = 1.09e9 —— 权重字节恒等式 decode 1.0958
    // 的主要来源之一。判据改为「非 qwen 的 KDA」。
    ...(qwen ? [operatorSpec(`${prefix}.decay_projection`, "forget/decay gate projection", "linear", {
      ...shapeFlow(shapes.hidden, gateShape),
      semantic_role: "forget_gate_logits",
      implementation: implementation.decay_projection,
      gate_lower_bound: normalized.linearLowerBound,
      projection_size: valueHeads,
    }, { input: dims.hidden, output: qwen ? betaDims : [-1, -1, keyHeads, keyDim] })] : []),
    ...(!qwen ? [operatorSpec(`${prefix}.f_b_proj`, "decay low-rank projection", "linear", {
      ...shapeFlow(`[batch, sequence, head dimension=${keyDim}]`, qkvShape),
      semantic_role: "forget_gate_low_rank_restore",
      implementation: "f_b_proj",
    }, { input: [-1, -1, keyDim], output: [-1, -1, keyProjection] })] : []),
    // 输出门：K3 是**全秩**（g 直接占融合投影的第 4 个 projection_size 分片，
    // kimi_k3/amd/kda.py:110 `qkvg_output_sizes = [projection_size] * 4`），
    // GLM5-Next 是**低秩**（融合内只有 g_a=head_dim，另有独立 g_b_proj
    // head_dim→projection_size，glm5next/nvidia/kda.py:248-254）。
    // 此前 g_b 整片缺失，GLM 每层少 head_dim×projection_size。
    ...(!qwen && !fullRank ? [operatorSpec(`${prefix}.g_b_proj`, "output gate low-rank projection", "linear", {
      ...shapeFlow(`[batch, sequence, head dimension=${keyDim}]`, qkvShape),
      semantic_role: "output_gate_low_rank_restore",
      implementation: "g_b_proj",
    }, { input: [-1, -1, keyDim], output: [-1, -1, keyProjection] })] : []),
    operatorSpec(`${prefix}.short_conv`, "qkv causal short convolution", "causal_conv1d", {
      ...shapeFlow(convShape, convShape),
      semantic_role: "q_k_v_short_convolution",
      implementation: implementation.short_convolution,
      branches: ["q", "k", "v"],
      kernel_size: normalized.linearConvKernelSize,
      activation: "silu",
      channel_layout: qwen ? { q: keyProjection, k: keyProjection, v: valueProjection, z: valueProjection } : undefined,
      // P4-2：depthwise 卷积核（causal-conv1d 自定义 kernel，无 quant_method → 不量化； [channels, kernel]，channels = 2·keyProj + valueProj
      //（extractor causal_conv1d case 的 width 同式）；沿通道（头维）tp 切）
      weightMatrices: [weightMatrixDecl("tp", { shape: [2 * keyProjection + valueProjection, normalized.linearConvKernelSize || 0], split: "output", quantizable: false })],
    }, { input: qkvConvDims, output: qkvConvDims }),
    operatorSpec(`${prefix}.state_update`, "KDA recurrent state", "gated_delta_attention", {
      ...shapeFlow(`${convShape}, ${betaShape}, ${stateShape}`, qwen ? gateShape : qkvShape),
      semantic_role: "gated_delta_recurrent_state",
      model_kind: modelKind,
      // P4-2：递推核标量参数（extractor stateUpdateCounts 的 gdnScalars 项，
      // fp32 走 paramDtypes 的 gdn_decay）。两族形状不同（kimi_gdn_linear_attn.py
      // :237-241,265-268 沿头维 sharded_weight_loader → tp）：
      //   qwen GDN：dt_bias 与 A_log 都是 valueHeads → 一组 [heads] matrices=2
      //   KDA（glm5_next/kimi_k3）：dt_bias = projection_size（heads·valueDim）、
      //     A_log = heads
      weightMatrices: (modelKind === "glm5_next" || modelKind === "kimi_k3")
        ? [
          weightMatrixDecl("tp", { shape: [valueHeads * valueDim], param_dtype: "gdn_decay", quantizable: false, split: "output" }),
          weightMatrixDecl("tp", { shape: [valueHeads], param_dtype: "gdn_decay", quantizable: false, split: "output" }),
        ]
        : [weightMatrixDecl("tp", { shape: [valueHeads], matrices: 2, param_dtype: "gdn_decay", quantizable: false, split: "output" })],
      attention_kind: "linear",
      mode: "chunk_prefill_or_fused_recurrent",
      qk_l2norm: true,
      beta_activation: "sigmoid",
      safe_gate: !qwen,
      decay_activation: qwen ? "softplus" : "bounded_sigmoid",
      gate_lower_bound: normalized.linearLowerBound,
      decay_parameters: ["A_log", "dt_bias"],
      state_shape: stateShape,
      ...linearStateResidentDecl(normalized),
    }, { input: qkvConvDims, output: outputDims }),
    // M8-V2：kimi_k3 的 gated norm / o_proj 输入 = state 输出宽（projection），
    // 非融合聚合宽（源码：o_norm(128 逐头门控) → o_proj 12288→hidden）
    operatorSpec(`${prefix}.output_gate_norm`, "gated RMSNorm", "gated_rmsnorm", {
      ...shapeFlow(qkvShape, gateShape),
      semantic_role: "gated_output_normalization",
      implementation: implementation.output_gate,
      gate_shape: gateShape,
      activation: normalized.outputGateType || "sigmoid",
    }, { input: gateNormDims, output: gateNormDims }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", {
      ...shapeFlow(gateShape, shapes.hidden),
      semantic_role: "attention_output_projection",
      communication_role: "tp_attention_output",
    }, { input: outputDims, output: dims.hidden }),
  ];
  return specs;
}

export function qwen35FullAttentionOperatorSpecs(prefix, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  const qProjection = (normalized.attentionHeads || 0) * (normalized.headDim || 0);
  const kvProjection = (normalized.kvHeads || normalized.attentionHeads || 0) * (normalized.headDim || 0);
  const fusedWidth = 2 * qProjection + 2 * kvProjection;
  const fusedShape = `[batch, sequence, fused qkv+gate=${fusedWidth}]`;
  const qShape = shapes.attentionQuery;
  const kShape = shapes.attentionKey;
  const vShape = shapes.attentionValue;
  const gateShape = qShape;
  const qkvDims = [-1, -1, fusedWidth];
  return [
    operatorSpec(`${prefix}.qkv_gate_proj`, "fused QKV + attention gate projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: ["q", "gate", "k", "v"],
      implementation: ["vLLM.Qwen3NextAttention.qkv_proj", "SGLang.Qwen3_5Attention.qkv_proj"],
    }, { input: dims.hidden, output: qkvDims }),
    operatorSpec(`${prefix}.qkv_gate_split`, "QKV + gate split", "split", {
      ...shapeFlow(fusedShape, `${qShape}, ${gateShape}, ${kShape}, ${vShape}`),
      split_sizes: [qProjection, qProjection, kvProjection, kvProjection],
    }, { input: qkvDims, output: [-1, -1, qProjection] }),
    operatorSpec(`${prefix}.q_norm`, "Q attention Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(qShape, qShape), { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.k_norm`, "K attention Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(kShape, kShape), { input: dims.attentionKey, output: dims.attentionKey }),
    ...scaledDotProductTail(prefix, shapes, dims, {
      rope: { partial_rotary_factor: normalized.partialRotaryFactor },
      scores: { attention_kind: "qwen35_full" },
      context: { attention_kind: "qwen35_full" },
      cacheResident: cacheResidentDecl({ kvElements: 2 * kvProjection }),
      preOutput: [operatorSpec(`${prefix}.output_gate`, "attention output gate", "attention_output_gate", {
        ...shapeFlow(`${shapes.attentionContext}, ${gateShape}`, shapes.attentionContext),
        activation: recipeAttentionOutputGate(normalized) ? "sigmoid" : "none",
        implementation: ["vLLM.fused_sigmoid_mul", "SGLang.fused_sigmoid_mul"],
      }, { input: dims.attentionContext, output: dims.attentionContext })],
    }),
  ];
}

export function mlaAttentionOperatorSpecs(prefix, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  const specs = [];
  if (normalized.qLoraRank != null) {
    // P4-2：q_a + q_a_norm 的组成与 extractor mla_query_compress ctx 同源。
    specs.push(operatorSpec(`${prefix}.q_a_proj`, "query down projection", "mla_query_compress", {
      ...shapeFlow(shapes.hidden, `[batch, sequence, q latent=${normalized.qLoraRank}]`),
      weightMatrices: [
        weightMatrixDecl("tp", { shape: [normalized.qLoraRank, dimWidth(dims.hidden)], split: "output" }),
      ],
    }, { input: dims.hidden, output: [-1, -1, normalized.qLoraRank] }));
    specs.push(operatorSpec(`${prefix}.q_a_norm`, "query latent RMSNorm", "rmsnorm", shapeFlow(`[batch, sequence, q latent=${normalized.qLoraRank}]`, `[batch, sequence, q latent=${normalized.qLoraRank}]`), { input: [-1, -1, normalized.qLoraRank], output: [-1, -1, normalized.qLoraRank] }));
    specs.push(operatorSpec(`${prefix}.q_b_proj`, "query up projection", "linear", shapeFlow(`[batch, sequence, q latent=${normalized.qLoraRank}]`, shapes.attentionQuery), { input: [-1, -1, normalized.qLoraRank], output: dims.attentionQuery }));
  } else {
    specs.push(operatorSpec(`${prefix}.q_proj`, "q projection", "linear", shapeFlow(shapes.hidden, shapes.attentionQuery), { input: dims.hidden, output: dims.attentionQuery }));
  }
  // P4-2：extractor mla_kv_compress ctx 只有一个 proj 项（out 取叶 output 宽）。
  specs.push(operatorSpec(`${prefix}.kv_a_proj`, "KV compression projection", "mla_kv_compress", {
    ...shapeFlow(shapes.hidden, `[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"} + rope=${normalized.qkRopeHeadDim ?? "unknown"}]`),
    weightMatrices: [weightMatrixDecl("tp", { shape: [(normalized.kvLoraRank || 0) + (normalized.qkRopeHeadDim || 0), dimWidth(dims.hidden)], split: "output" })],
  }, { input: dims.hidden, output: [-1, -1, (normalized.kvLoraRank || 0) + (normalized.qkRopeHeadDim || 0)] }));
  specs.push(operatorSpec(`${prefix}.kv_split`, "KV latent and rope split", "mla_kv_split", {
    ...shapeFlow(`[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"} + rope=${normalized.qkRopeHeadDim ?? "unknown"}]`, `[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}], [batch, sequence, rope=${normalized.qkRopeHeadDim ?? "unknown"}]`),
    split_sizes: [normalized.kvLoraRank, normalized.qkRopeHeadDim],
  }, { input: [-1, -1, (normalized.kvLoraRank || 0) + (normalized.qkRopeHeadDim || 0)], output: [-1, -1, normalized.kvLoraRank] }));
  specs.push(operatorSpec(`${prefix}.kv_a_norm`, "KV latent RMSNorm", "rmsnorm", shapeFlow(`[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}]`, `[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}]`), { input: [-1, -1, normalized.kvLoraRank], output: [-1, -1, normalized.kvLoraRank] }));
  // kv_b 真值输出宽 = qk_nope + v_head_dim（128+128=256），非 headDim（nope+rope=192）
  // ——R1 逐项对账审计登记残差（details/cost_counts.md），2026-09-08 修复。
  specs.push(operatorSpec(`${prefix}.kv_b_proj`, "KV expansion projection", "linear", shapeFlow(`[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}]`, `[batch, sequence, kv heads=${normalized.kvHeads ?? "unknown"}, kv expansion=${(normalized.qkNopeHeadDim || 0) + (normalized.valueHeadDim || normalized.headDim || 0)}]`), { input: [-1, -1, normalized.kvLoraRank], output: [-1, -1, normalized.kvHeads, (normalized.qkNopeHeadDim || 0) + (normalized.valueHeadDim || normalized.headDim || 0)] }));
  specs.push(...scaledDotProductTail(prefix, shapes, dims, {
    rope: {
      query_shape: shapes.attentionQuery,
      key_shape: shapes.attentionKey,
    },
    scoresName: "latent attention scores",
    scores: {
      formula: "S = Q K^T / sqrt(d_rope)",
      attention_kind: "mla",
    },
    context: { attention_kind: "mla" },
    cacheResident: cacheResidentDecl({ kvElements: (normalized.kvLoraRank || 0) + (normalized.qkRopeHeadDim || 0) }),    preOutput: [
      ...(normalized.mlaUseOutputGate
        ? [operatorSpec(`${prefix}.g_proj`, "MLA output gate", "mla_output_gate", shapeFlow(shapes.hidden, shapes.attentionContext), { input: dims.hidden, output: dims.attentionContext })]
        : []),
      // M8-V2：kimi_k3 MLA 的 full-rank 输出门（index.json 实锤 88.1M/层）
      ...(recipeLinearAttentionMode(normalized) === "kimi_k3"
        ? [operatorSpec(`${prefix}.mla_gate`, "MLA full-rank output gate", "linear", { ...shapeFlow(shapes.hidden, shapes.attentionQuery), semantic_role: "attention_output_gate" }, { input: dims.hidden, output: dims.attentionQuery })]
        : []),
    ],
  }));
  return specs;
}

// DeepSeek V4 的 vLLM/SGLang 实现使用不同 fused kernel，但语义都是同一条 MLA 链。
// 这里保留一份 canonical 节点，通过 implementation 与 compress_ratio 描述实现差异。
export function deepseekV4AttentionOperatorSpecs(prefix, normalized, layerIndex = 0) {
  const dims = tensorDims(normalized);
  const ratio = normalized.compressRatios?.[layerIndex] ?? 0;
  // V4.1（deepseek_v41）跨层复用：compressor/indexer 权重只在 source 层，其余层复用。
  // checkpoint index 实证：compressor∈kv_source_layer_ids、indexer∈index_source_layer_ids。
  // 缺省（V4-Flash/Pro 无 source_layer_ids）退回 compress_ratio 启发式，行为不变。
  const kvSourceLayerIds = normalized.kvSourceLayerIds;
  const indexSourceLayerIds = normalized.indexSourceLayerIds;
  const emitCompressor = Array.isArray(kvSourceLayerIds) ? kvSourceLayerIds.includes(layerIndex) : ratio > 1;
  // ratio===4（C4 sparse_mla 注意力）必然依赖 indexer，故除 index_source 层外，
  // ratio===4 也强制 emit indexer，避免出现 sparse_mla 无 indexer 的悬挂结构。
  const emitIndexer = (Array.isArray(indexSourceLayerIds) && indexSourceLayerIds.includes(layerIndex)) || ratio === 4;
  // 压缩稀疏 MLA 判据：V4 CSA(ratio=4) 与 V4.1 CSA2(ratio=2) 都是"压缩 KV + indexer 选择"的稀疏注意力，
  // 只有 ratio=128 是 HCA 稠密、ratio=0 是滑窗。此前硬编码 ratio===4 把 V4.1 的 ratio=2 层误判成滑窗 MQA
  // （41/43 层），并使 source 层的 indexer 悬挂。改用 isSparse 后 V4(0/4/128) 行为不变、V4.1(0/2) 修正。
  // 压缩稀疏 MLA 判据：ratio∈{1,2,4} 都是"压缩 KV + indexer 选择"的稀疏注意力（V4.1 CSA2 含 ratio=1 的
  // Full 模式：全长压缩 KV，仍走 indexer top-k），只有 ratio=128 是 HCA 稠密、ratio=0 是纯滑窗。改为
  // ratio>0&&!=128 后：V4-Flash/Pro（0/4/128）不变、V4.1 的 ratio=1 层（真实 layer 20-39）归入 sparse_mla
  // （消除 ratio=1 source 层的 dangling compressor/indexer）。（全 catalog 实测仅 V4.1 含 ratio=1。）
  const isSparse = ratio > 0 && ratio !== 128;
  const qRank = normalized.qLoraRank;
  const headDim = normalized.headDim;
  // dsv4 KV cache 有效 dtype（边际字节口径）：V4.1=F4、V4-Flash/Pro=F8_E4M3。
  const kvDtype = normalized.kvCacheDtype || "BF16";
  // 逐张量实证（docs/details/evidence/structure/deepseek_v41_tensor_identity.md）：V4.1 压缩 KV cache=fp4+E4M3/16、
  // index k_cache=fp4+E8M0/32（含 scale 摊销的有效字节）；V4（F8）两者同 dtype、不变。
  const kvCacheDtype = kvDtype === "F4" ? "F4_E4M3S16" : kvDtype;
  const indexCacheDtype = kvDtype === "F4" ? "F4_E8M0S32" : kvDtype;
  // index **键缓存**（owns k_cache）仅在 kv_source∩index_source（indexer.wk 实证只在 [2,8,14,20]）；
  // 24/28/32/36 有 index 查询但复用共享键、不自带 k_cache。故 index 常驻门控在 emitCompressor && emitIndexer
  // （V4：ratio>1 && ratio===4 = ratio===4，与原 emitIndexer 等价、行为不变）。
  const ownsIndexKey = emitCompressor && emitIndexer;
  const groups = normalized.oGroups;
  const outputRank = normalized.oLoraRank;
  const indexHeads = normalized.dsaIndexHeads;
  const indexDim = normalized.dsaIndexHeadDim;
  const budget = normalized.dsaIndexTopk;
  const qLatent = `[batch, sequence, q latent=${qRank ?? "unknown"}]`;
  const kvLatent = `[batch, sequence, kv latent=${headDim ?? "unknown"}]`;
  const query = `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim ?? "unknown"}]`;
  const outputLatent = `[batch, sequence, output groups=${groups ?? "unknown"}, output rank=${outputRank ?? "unknown"}]`;
  const specs = [
    operatorSpec(`${prefix}.fused_wqa_wkv`, "fused q/kv projection", "linear", {
      ...shapeFlow(shapesForHidden(normalized), `${qLatent}, ${kvLatent}`),
      projection_layout: ["q_lora", "kv"],
      q_lora_rank: qRank,
      kv_head_dim: headDim,
      implementation: ["vLLM.fused_wqa_wkv", "SGLang.wqkv_a or wq_a+wkv"],
    }, { input: dims.hidden, output: [-1, -1, (qRank || 0) + (headDim || 0)] }),
    operatorSpec(`${prefix}.qkv_split`, "q/kv latent split", "split", {
      ...shapeFlow(`${qLatent}, ${kvLatent}`, `${qLatent}, ${kvLatent}`),
      split_sizes: [qRank, headDim],
      implementation: ["vLLM.split_qkv_and_norm", "SGLang._compute_q_a/_compute_kv"],
    }, { input: [-1, -1, (qRank || 0) + (headDim || 0)], output: [-1, -1, qRank || 0] }),
    operatorSpec(`${prefix}.q_norm`, "query latent RMSNorm", "rmsnorm", shapeFlow(qLatent, qLatent), { input: [-1, -1, qRank], output: [-1, -1, qRank] }),
    operatorSpec(`${prefix}.kv_norm`, "KV latent RMSNorm", "rmsnorm", shapeFlow(kvLatent, kvLatent), { input: [-1, -1, headDim], output: [-1, -1, headDim] }),
    operatorSpec(`${prefix}.q_proj`, "query expansion projection", "linear", {
      ...shapeFlow(qLatent, query),
      projection_role: "wq_b",
      implementation: ["vLLM.wq_b", "SGLang.wq_b"],
    }, { input: [-1, -1, qRank], output: [-1, -1, normalized.attentionHeads, headDim] }),
    operatorSpec(`${prefix}.rope`, "query/KV rotary position embedding", "rope", {
      ...shapeFlow(`${query}, ${kvLatent}`, `${query}, ${kvLatent}`),
      qk_rope_head_dim: normalized.qkRopeHeadDim,
      compress_ratio: ratio,
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }),
  ];

  // dsv4 注意力 sink：per-head fp32 标量参数（SGLang models/deepseek_v4.py:708
  // self.attn_sink = nn.Parameter(torch.empty(n_heads, dtype=torch.float32)）——
  // softmax 分母加一项 exp(sink - max)（decode/extend kernel）。checkpoint layers.N.attn.attn_sink。
  // 无条件挂在 dsv4 注意力（V4/V4.1 独占）的打分算子上；非量化、replicated、不进 KV。
  const attnSinkMatrices = [weightMatrixDecl("replicated", {
    shape: [normalized.attentionHeads || 0],
    quantizable: false,
    param_dtype: "attn_sink",
  })];
  // DeepSeek-V4 嵌套 compressor 架构的绝对位置嵌入 ape（checkpoint 名 position_bias）：
  // compressor 层 [ratio, coff·head_dim]（coff = ratio===4?2:1）、indexer 内嵌 compressor
  // [4, 2·index_head_dim]。仅 V4（recipe compressorApe）；V4.1 flat（wkv）无。非量化 fp32、
  // 驻留辅助参数（锚 1 residency-aux 扣除）。
  const hasCompressorApe = recipeFlag(normalized, "compressorApe");
  // SGLang Compressor（compressor.py:28）coff = 1 + (ratio==4)：ratio=4 overlap→2、其余→1。
  // 仅在真实嵌套 compressor 架构（V4，compressorApe）用此宽；V4.1 是 flat wkv（本处 compressor
  // 为近似），保持旧 (ratio>1?2:1) 不动，避免扰动其已标定的 890 KV 锚点。
  const compCoff = hasCompressorApe ? (ratio === 4 ? 2 : 1) : (ratio > 1 ? 2 : 1);

  if (emitCompressor) {
    specs.push(operatorSpec(`${prefix}.compressor`, "compressed KV/state compressor", "mla_kv_compress", {
      ...shapeFlow(shapesForHidden(normalized), `[compressed sequence=ceil(sequence/${ratio}), state dimension]`),
      compress_ratio: ratio,
      implementation: ["vLLM.DeepseekCompressor", "SGLang.Compressor.wkv_gate"],
      cache_role: "compressed_kv_and_score_state",
      // extractor mla_kv_compress ctx：out 以叶 output_shape 为权威（模板声明）。
      weightMatrices: [
        weightMatrixDecl("tp", { shape: [2 * compCoff * headDim, dimWidth(dims.hidden)], split: "output" }),
        ...(hasCompressorApe ? [weightMatrixDecl("replicated", { shape: [ratio, compCoff * headDim], quantizable: false, param_dtype: "compressor_ape" })] : []),
      ],
    }, { input: dims.hidden, output: [-1, -1, 2 * compCoff * headDim] }));
    // SGLang Compressor.norm = RMSNorm(head_dim, fp32)（compressor.py:43）。checkpoint attn.compressor.norm。
    if (hasCompressorApe) {
      specs.push(operatorSpec(`${prefix}.compressor.norm`, "compressor latent RMSNorm", "rmsnorm",
        shapeFlow(`[batch, sequence, head dimension=${headDim}]`, `[batch, sequence, head dimension=${headDim}]`),
        { input: [-1, -1, headDim], output: [-1, -1, headDim] }));
    }
  }

  if (emitIndexer) {
    specs.push(operatorSpec(`${prefix}.indexer.weights_proj`, "indexer weight projection", "linear", {
      ...shapeFlow(shapesForHidden(normalized), `[batch, sequence, index heads=${indexHeads}]`),
      implementation: ["vLLM.DeepseekV4Indexer.weights_proj", "SGLang.C4Indexer"],
    }, { input: dims.hidden, output: [-1, -1, indexHeads] }));
    specs.push(operatorSpec(`${prefix}.indexer.q_proj`, "indexer query projection", "linear", {
      ...shapeFlow(qLatent, `[batch, sequence, index heads=${indexHeads}, index head dimension=${indexDim}]`),
      implementation: ["vLLM.DeepseekV4Indexer.wq_b", "SGLang.C4Indexer"],
    }, { input: [-1, -1, qRank], output: [-1, -1, indexHeads, indexDim] }));
    // indexer 内嵌 Compressor（SGLang indexer.py:1115，ratio=4/overlap→coff=2）：wkv_gate 投影 +
    // norm(RMSNorm index_head_dim) + ape。checkpoint attn.indexer.compressor.{wkv,wgate,norm,ape}。
    if (hasCompressorApe) {
      specs.push(operatorSpec(`${prefix}.indexer.compressor.wkv_gate`, "indexer compressor wkv/gate projection", "linear", {
        ...shapeFlow(shapesForHidden(normalized), `[batch, sequence, ${2 * 2 * (indexDim || 0)}]`),
        implementation: ["SGLang.C4Indexer.compressor.wkv_gate"],
      }, { input: dims.hidden, output: [-1, -1, 2 * 2 * (indexDim || 0)] }));
      specs.push(operatorSpec(`${prefix}.indexer.compressor.norm`, "indexer compressor RMSNorm", "rmsnorm",
        shapeFlow(`[batch, sequence, index head dimension=${indexDim}]`, `[batch, sequence, index head dimension=${indexDim}]`),
        { input: [-1, -1, indexDim], output: [-1, -1, indexDim] }));
    }
    specs.push(operatorSpec(`${prefix}.indexer`, "DeepSeek V4 C4 sparse indexer", "dsv4_indexer", {
      ...shapeFlow(shapesForHidden(normalized), `[batch, sequence, selected=${budget}]`),
      indexer_heads: indexHeads,
      indexer_head_dim: indexDim,
      budget,
      compress_ratio: ratio,
      implementation: ["vLLM.DeepseekV4Indexer", "SGLang.C4Indexer"],
      // indexer 内嵌 compressor 的 ape（SGLang indexer.py:1115 nested Compressor，ratio=4/overlap）。
      ...(hasCompressorApe ? { weightMatrices: [weightMatrixDecl("replicated", { shape: [4, 2 * (indexDim || 0)], quantizable: false, param_dtype: "compressor_ape" })] } : {}),
    }, { input: dims.hidden, output: [-1, -1, budget] }));
  }

  if (isSparse) {
    specs.push(operatorSpec(`${prefix}.attention`, "C4 sparse MLA attention", "dsv4_sparse_mla", {
      ...shapeFlow(`${query}, selected compressed KV`, `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`),
      selected_tokens: budget,
      compress_ratio: ratio,
      attention_kind: "dsv4_sparse_mla",
      ...cacheResidentDecl({
        // 全驻留（W5 capacity↔kvRead 对账，语义不变）：各层滑窗 head_dim + 压缩 KV（仅 kv_source）；
        // index 键缓存仅 owns_k（kv_source∩index_source）。V4（无 source_layer_ids）emitCompressor=ratio>1、
        // ownsIndexKey=ratio===4，与原 emitIndexer 等价（golden 不变）。
        kvElements: headDim + (emitCompressor ? (2 * (isSparse ? 2 : 1) * headDim) / ratio : 0),
        indexElements: ownsIndexKey ? (indexDim || 0) : 0,
        // 边际 + 逐 dtype（对齐官方 Global KV/token）：压缩 KV 单份 head_dim/ratio（仅 kv_source，fp4+E4M3/16）+
        // index 键 index_head_dim/ratio（仅 owns_k，fp4+E8M0/32），滑窗有界不计。实证见 v41_tensor_identity_reconcile.md。
        growthKvElements: emitCompressor ? headDim / ratio : 0,
        growthIndexElements: ownsIndexKey ? (indexDim || 0) / ratio : 0,
        kvDtype: kvCacheDtype,
        indexDtype: indexCacheDtype,
      }),
      implementation: ["vLLM.DeepseekV4FlashMLAAttention", "SGLang.RadixAttention + DSV4 backend"],
      weightMatrices: attnSinkMatrices,
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }));
  } else if (ratio === 128) {
    specs.push(operatorSpec(`${prefix}.attention`, "compressed MLA attention", "dsv4_compressed_attention", {
      ...shapeFlow(`${query}, compressed KV`, `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`),
      compress_ratio: ratio,
      attention_kind: "dsv4_compressed_mla",
      // 全驻留不变（W5）：滑窗 + 压缩 KV。边际：压缩 KV 单份 head_dim/ratio（fp4/fp8），滑窗有界不计。
      ...cacheResidentDecl({
        kvElements: headDim + (2 * 1 * headDim) / ratio,
        growthKvElements: headDim / ratio,
        kvDtype,
      }),
      implementation: ["vLLM.DeepseekV4FlashMLAAttention", "SGLang.MQALayer"],
      weightMatrices: attnSinkMatrices,
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }));
  } else {
    specs.push(operatorSpec(`${prefix}.attention`, "sliding-window MQA", "dsv4_swa_attention", {
      ...shapeFlow(`${query}, KV window`, `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`),
      sliding_window: normalized.slidingWindow,
      compress_ratio: ratio,
      attention_kind: "dsv4_swa_mqa",
      // 全驻留不变（W5）：滑窗 head_dim。边际=0（滑窗有界，不随 token 增长）。
      ...cacheResidentDecl({ kvElements: headDim, growthKvElements: 0, kvDtype }),
      implementation: ["vLLM.DeepseekV4SWACache", "SGLang.RadixAttention"],
      weightMatrices: attnSinkMatrices,
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }));
  }

  specs.push(
    operatorSpec(`${prefix}.inverse_rope`, "inverse output rotary transform", "rope", {
      ...shapeFlow(`[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`, `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`),
      phase: "output_inverse_rope",
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }),
    operatorSpec(`${prefix}.wo_a`, "output low-rank projection", "linear", {
      ...shapeFlow(`[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`, outputLatent),
      projection_role: "wo_a",
      output_groups: groups,
      output_rank: outputRank,
      implementation: ["vLLM.wo_a", "SGLang.wo_a"],
      // vLLM DeepseekV4Attention：ColumnParallelLinear(
      //   n_heads*head_dim/o_groups, o_groups*o_lora_rank)，is_bmm=True。
      // 权重是 [o_groups*o_lora, head_dim*n_heads/o_groups]，不是 grouped
      // 输出维乘积 (o_groups*o_lora) × (n_heads*head_dim)。
      weightMatrices: [weightMatrixDecl("tp", {
        shape: [(groups || 0) * (outputRank || 0), ((normalized.attentionHeads || 0) * (headDim || 0)) / Math.max(groups || 1, 1)],
        split: "output",
      })],
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, groups, outputRank] }),
    operatorSpec(`${prefix}.wo_b`, "output hidden projection", "linear", {
      ...shapeFlow(outputLatent, shapesForHidden(normalized)),
      projection_role: "wo_b",
      communication_role: "tp_attention_output",
      implementation: ["vLLM.wo_b", "SGLang.wo_b"],
    }, { input: [-1, -1, groups, outputRank], output: dims.hidden }),
  );
  return specs;
}

function shapesForHidden(normalized) {
  return `[batch, sequence, hidden size=${normalized.hiddenSize ?? "unknown"}]`;
}

export function qsaAttentionOperatorSpecs(prefix, normalized, layerIndex = 0) {
  // 2026-09-08 证据改判：glm5_next（GLM-5.3-Flash）主注意力是 MLA + DSA
  // indexer（modeling_glm5_next.py:739-741 "DeepSeek Sparse Attention (DSA)
  // indexer with k-pool compression"、:1473 mask 名 deepseek_sparse_attention），
  // 此前误入逐头 QSA 模板。差异：index_kpool=4 池化、qk_rope_head_dim=0。
  if ((normalized.kvLoraRank || 0) > 0) {
    return dsaAttentionOperatorSpecs(prefix, normalized, layerIndex);
  }
  const { shapes, dims } = shapesAndDims(normalized);
  const indexerHeads = normalized.qsaIndexerHeads ?? 0;
  const indexerKVHeads = normalized.qsaIndexerKVHeads ?? 0;
  const indexerDim = normalized.qsaIndexerHeadDim ?? 0;
  const budget = normalized.qsaIndexerBudget ?? 0;
  // 融合 QKV 的宽度：vLLM qwen4_exp/nvidia/qsa.py:233-241
  //   QKVParallelLinear(hidden, head_dim, total_num_heads*(1+attn_output_gate), total_num_kv_heads)
  // ⇒ head_dim·(heads·(1+gate) + 2·kv_heads)。此前只声明了 q 的宽度（heads·head_dim），
  // k/v 两份权重整层漏计（Flash-Next 每 QSA 层少 2·kv_heads·head_dim·hidden = 2,621,440）。
  const heads = normalized.attentionHeads || 0;
  const kvHeads = normalized.kvHeads || heads;
  const headDim = normalized.headDim || 0;
  const gateFactor = normalized.attentionOutputGate ? 2 : 1;
  const fusedWidth = headDim * (heads * gateFactor + 2 * kvHeads);
  const fusedShape = `[batch, sequence, fused qkv${normalized.attentionOutputGate ? " + output gate" : ""}=${fusedWidth}]`;
  return [
    operatorSpec(`${prefix}.qkv_proj`, "QSA qkv and output-gate projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: normalized.attentionOutputGate ? ["q", "gate", "k", "v"] : ["q", "k", "v"],
      implementation: ["vLLM.Qwen4ExpQSAAttention.qkv_proj", "SGLang.qwen4_exp qkv_proj"],
    }, { input: dims.hidden, output: [-1, -1, fusedWidth] }),
    operatorSpec(`${prefix}.q_norm`, "Q attention norm", "rmsnorm", shapeFlow(shapes.attentionQuery, shapes.attentionQuery), { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.k_norm`, "K attention norm", "rmsnorm", shapeFlow(shapes.attentionKey, shapes.attentionKey), { input: dims.attentionKey, output: dims.attentionKey }),
    operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, `${shapes.attentionQuery}, ${shapes.attentionKey}`),
      query_shape: shapes.attentionQuery,
      key_shape: shapes.attentionKey,
    }, { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.indexer`, "QSA indexer", "qsa_indexer", {
      ...shapeFlow(shapes.hidden, `[batch, sequence, selected=${budget}]`),
      indexer_heads: indexerHeads,
      indexer_kv_heads: indexerKVHeads,
      indexer_head_dim: indexerDim,
      budget,
      compress_ratio: normalized.qsaIndexerCompressRatio,
      implementation: ["vLLM.QSAIndexer", "SGLang.qwen4_exp indexer"],
    }, { input: dims.hidden, output: [-1, -1, budget] }),
    operatorSpec(`${prefix}.sparse_attention`, "QSA sparse attention", "qsa_sparse_attention", {
      ...shapeFlow(`${shapes.attentionQuery}, selected K/V`, shapes.attentionContext),
      selected_tokens: budget,
      attention_kind: "qsa",
      ...cacheResidentDecl({
        kvElements: (normalized.kvLoraRank != null && normalized.qkRopeHeadDim != null)
          ? (normalized.kvLoraRank + normalized.qkRopeHeadDim)
          : 2 * kvHeads * headDim,
        indexElements: indexerDim || 0,
      }),
      implementation: ["vLLM.Qwen4ExpQSAAttention", "SGLang.qwen4_exp qsa"],
    }, { input: dims.attentionQuery, output: dims.attentionContext }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", { ...shapeFlow(shapes.attentionContext, shapes.hidden), communication_role: "tp_attention_output" }, { input: dims.attentionContext, output: dims.hidden }),
  ];
}

function minimaxAttentionCommon(prefix, normalized, sparse, layerIndex = 0) {
  const { shapes, dims } = shapesAndDims(normalized);
  const heads = normalized.attentionHeads || 0;
  const kvHeads = normalized.kvHeads || heads;
  const headDim = normalized.headDim || 0;
  const qProjection = heads * headDim;
  const kvProjection = kvHeads * headDim;
  const indexHeads = normalized.sparseIndexHeads || kvHeads;
  const indexDim = normalized.sparseIndexDim || headDim;
  const indexProjection = indexHeads * indexDim;
  const disableIndexValue = normalized.sparseDisableIndexValue?.[layerIndex] ?? true;
  // index_k 是**单头共享**的（每 rank 复制一份），不是 index_heads 份：
  //   vLLM linear.py:1405-1414  output_sizes = [q, kv, kv, iq=heads·dim, ik=index_head_size]
  //   vLLM minimax_m3/amd/model.py:983  index_k = qkv[:, start : start + self.idx_head_dim]
  //   SGLang minimax_m3.py:632-639  index_qkv_proj = QKVParallelLinear(..., total_num_kv_heads=1,
  //                                   v_head_size=(0 if disable_index_value else idx_head_dim))
  // index_v 同为单头，且只在该层 sparse_disable_index_value=0 时才存在。
  const indexKeyProjection = indexDim;
  const indexValueProjection = disableIndexValue ? 0 : indexDim;
  const indexKvShape = `[batch, sequence, index kv heads=1, index head dimension=${indexDim}]`;
  const fusedWidth = qProjection + 2 * kvProjection
    + (sparse ? indexProjection + indexKeyProjection + indexValueProjection : 0);
  const fusedShape = `[batch, sequence, fused main QKV + index QKV=${fusedWidth}]`;
  const indexShape = `[batch, sequence, index heads=${indexHeads}, index head dimension=${indexDim}]`;
  const specs = [
    operatorSpec(`${prefix}.qkv_index_proj`, sparse ? "fused QKV + index projection" : "QKV projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: sparse ? ["q", "k", "v", "index_q", "index_k", ...(disableIndexValue ? [] : ["index_v"])] : ["q", "k", "v"],
      implementation: sparse
        ? ["vLLM.MinimaxM3QKVParallelLinearWithIndexer", "SGLang._FusedQKVIndexProj"]
        : ["vLLM.QKVParallelLinear", "SGLang.qkv_proj"],
      disable_index_value: disableIndexValue,
    }, { input: dims.hidden, output: [-1, -1, fusedWidth] }),
    operatorSpec(`${prefix}.qkv_index_split`, sparse ? "main/index QKV split" : "QKV split", "split", {
      ...shapeFlow(fusedShape, sparse ? `${shapes.attentionQuery}, ${shapes.attentionKey}, ${shapes.attentionValue}, ${indexShape}, ${indexKvShape}` : `${shapes.attentionQuery}, ${shapes.attentionKey}, ${shapes.attentionValue}`),
      split_sizes: sparse ? [qProjection, kvProjection, kvProjection, indexProjection, indexKeyProjection, ...(disableIndexValue ? [] : [indexValueProjection])] : [qProjection, kvProjection, kvProjection],
    }, { input: [-1, -1, fusedWidth], output: [-1, -1, qProjection] }),
    operatorSpec(`${prefix}.q_norm`, "Q Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(shapes.attentionQuery, shapes.attentionQuery), { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.k_norm`, "K Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(shapes.attentionKey, shapes.attentionKey), { input: dims.attentionKey, output: dims.attentionKey }),
  ];
  if (sparse) {
    specs.push(
      operatorSpec(`${prefix}.rope`, "partial rotary position embedding", "rope", {
        ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, `${shapes.attentionQuery}, ${shapes.attentionKey}`),
        partial_rotary_factor: normalized.partialRotaryFactor,
        implementation: ["vLLM.MiniMaxM3Attention.rotary_emb", "SGLang.MiniMaxM3Attention.rotary_emb"],
      }, { input: dims.attentionQuery, output: dims.attentionQuery }),
      operatorSpec(`${prefix}.index_q_norm`, "index Q Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(indexShape, indexShape), { input: [-1, -1, indexHeads, indexDim], output: [-1, -1, indexHeads, indexDim] }),
      // index_k / index_rope 的 k 侧都是那颗**单头共享** index-k（vLLM
      // MiniMAXGemmaRMSNorm(self.idx_head_dim) 作用在 1 x idx_head_dim 上）。
      operatorSpec(`${prefix}.index_k_norm`, "index K Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(indexKvShape, indexKvShape), { input: [-1, -1, 1, indexDim], output: [-1, -1, 1, indexDim] }),
      operatorSpec(`${prefix}.index_rope`, "index partial rotary position embedding", "rope", {
        ...shapeFlow(`${indexShape}, ${indexKvShape}`, `${indexShape}, ${indexKvShape}`),
        partial_rotary_factor: normalized.partialRotaryFactor,
      }, { input: [-1, -1, indexHeads, indexDim], output: [-1, -1, indexHeads, indexDim] }),
      operatorSpec(`${prefix}.indexer`, "MiniMax M3 block indexer", "minimax_sparse_indexer", {
        ...shapeFlow(`${indexShape}, ${indexKvShape}`, `[batch, sequence, selected blocks=${normalized.sparseTopkBlocks}]`),
        index_heads: indexHeads,
        index_head_dim: indexDim,
        topk_blocks: normalized.sparseTopkBlocks,
        block_size: normalized.sparseBlockSize,
        init_blocks: normalized.sparseInitBlock,
        local_blocks: normalized.sparseLocalBlock,
        score_type: normalized.sparseScoreType || "max",
        disable_index_value: disableIndexValue,
        implementation: ["vLLM.MiniMaxM3Indexer", "SGLang.Minimax sparse indexer"],
      }, { input: [-1, -1, indexHeads, indexDim], output: [-1, -1, normalized.sparseTopkBlocks] }),
      operatorSpec(`${prefix}.sparse_attention`, "MiniMax M3 block-sparse GQA", "minimax_sparse_attention", {
        ...shapeFlow(`${shapes.attentionQuery}, selected KV blocks`, shapes.attentionContext),
        topk_blocks: normalized.sparseTopkBlocks,
        block_size: normalized.sparseBlockSize,
        local_blocks: normalized.sparseLocalBlock,
        init_blocks: normalized.sparseInitBlock,
        disable_index_value: disableIndexValue,
        attention_kind: "minimax_m3_sparse_gqa",
        ...cacheResidentDecl({
          kvElements: 2 * kvHeads * (normalized.headDim || 0),
          // index cache 是**单头共享** index-k（K，V 仅在该层 disable_index_value=0 时存），
          // 不是 index_heads 份：SGLang minimax_m3.py index_kv/index_k pool 均 head_num=1、
          // head_dim=idx_head_dim（memory_pool.py:5433-5461）。此前误用 query 侧
          // sparseIndexHeads·sparseIndexDim(=4·128=512)，与本文件 874-881 单头建模自相矛盾。
          indexElements: indexKeyProjection + indexValueProjection,
        }),
        implementation: ["vLLM.MiniMaxM3SparseImpl", "SGLang.minimax_sparse_backend"],
      }, { input: dims.attentionQuery, output: dims.attentionContext }),
    );
    specs.push(operatorSpec(`${prefix}.o_proj`, "output projection", "linear", { ...shapeFlow(shapes.attentionContext, shapes.hidden), communication_role: "tp_attention_output" }, { input: dims.attentionContext, output: dims.hidden }));
  } else {
    specs.push(...scaledDotProductTail(prefix, shapes, dims, {
      ropeName: "partial rotary position embedding",
      rope: {
        partial_rotary_factor: normalized.partialRotaryFactor,
        implementation: ["vLLM.MiniMaxM3Attention.rotary_emb", "SGLang.MiniMaxM3Attention.rotary_emb"],
      },
      cacheResident: cacheResidentDecl({ kvElements: 2 * kvHeads * (normalized.headDim || 0) }),
    }));
  }
  return specs;
}

export function minimaxDenseAttentionOperatorSpecs(prefix, normalized) {
  return minimaxAttentionCommon(prefix, normalized, false);
}

export function minimaxSparseAttentionOperatorSpecs(prefix, normalized, layerIndex = 0) {
  return minimaxAttentionCommon(prefix, normalized, true, layerIndex);
}

export function minimaxM2AttentionOperatorSpecs(prefix, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  const qProjection = (normalized.attentionHeads || 0) * (normalized.headDim || 0);
  const kvProjection = (normalized.kvHeads || normalized.attentionHeads || 0) * (normalized.headDim || 0);
  const fusedWidth = qProjection + 2 * kvProjection;
  const fusedShape = `[batch, sequence, fused qkv=${fusedWidth}]`;
  return [
    operatorSpec(`${prefix}.qkv_proj`, "fused QKV projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: ["q", "k", "v"],
      bias: normalized.attentionBias,
      implementation: ["vLLM fused QKV", "SGLang fused QKV"],
    }, { input: dims.hidden, output: [-1, -1, fusedWidth] }),
    operatorSpec(`${prefix}.qkv_split`, "QKV split", "attention_qkv_split", {
      ...shapeFlow(fusedShape, `${shapes.attentionQuery}, ${shapes.attentionKey}, ${shapes.attentionValue}`),
      split_sizes: [qProjection, kvProjection, kvProjection],
    }, { input: [-1, -1, fusedWidth], output: [-1, -1, qProjection] }),
    operatorSpec(`${prefix}.q_norm`, "Q RMSNorm", "rmsnorm", shapeFlow(shapes.attentionQuery, shapes.attentionQuery), {
      input: dims.attentionQuery,
      output: dims.attentionQuery,
      norm_type: normalized.qkNormType || "per_layer",
      implementation: ["vLLM QK RMSNorm", "SGLang QK RMSNorm"],
    }),
    operatorSpec(`${prefix}.k_norm`, "K RMSNorm", "rmsnorm", shapeFlow(shapes.attentionKey, shapes.attentionKey), {
      input: dims.attentionKey,
      output: dims.attentionKey,
      norm_type: normalized.qkNormType || "per_layer",
      implementation: ["vLLM QK RMSNorm", "SGLang QK RMSNorm"],
    }),
    ...scaledDotProductTail(prefix, shapes, dims, {
      ropeName: "partial rotary position embedding",
      rope: {
        rotary_dim: normalized.rotaryDim,
        partial_rotary_factor: normalized.partialRotaryFactor,
      },
      cacheResident: cacheResidentDecl({ kvElements: 2 * kvProjection }),
    }),
  ];
}

// DeepSeek V3.2/GLM DSA 共用一份 MLA + indexer 语义；vLLM/SGLang 的融合方式只记录在 implementation。
function dsaAttentionOperatorSpecs(prefix, normalized, layerIndex) {
  const { shapes, dims } = shapesAndDims(normalized);
  const heads = normalized.attentionHeads || 0;
  const qRank = normalized.qLoraRank || 0;
  const kvRank = normalized.kvLoraRank || 0;
  const qkNope = normalized.qkNopeHeadDim || 0;
  const ropeDim = normalized.qkRopeHeadDim || 0;
  const qkDim = qkNope + ropeDim;
  const valueDim = normalized.valueHeadDim || normalized.headDim || 0;
  const indexHeads = normalized.dsaIndexHeads ?? 0;
  const indexDim = normalized.dsaIndexHeadDim ?? 0;
  const budget = normalized.dsaIndexTopk ?? 0;
  const kpool = normalized.dsaIndexKpool ?? 1;
  const indexerMode = indexerScheduleOf(normalized)?.[layerIndex] || "compute";
  const qLatentShape = `[batch, sequence, q latent=${qRank}]`;
  const kvLatentShape = `[batch, sequence, kv latent=${kvRank}, rope=${ropeDim}]`;
  const qShape = `[batch, sequence, attention heads=${heads}, head dimension=${qkDim}]`;
  const kShape = `[batch, sequence, attention heads=${heads}, head dimension=${qkDim}]`;
  const vShape = `[batch, sequence, attention heads=${heads}, value head dimension=${valueDim}]`;
  return [
    operatorSpec(`${prefix}.q_a_proj`, "query down projection", "mla_query_compress", {
      ...shapeFlow(shapes.hidden, qLatentShape),
      implementation: ["vLLM.q_a_proj", "SGLang.q_a_proj"],
      // P4-2：只有 q_a 一个矩阵 —— extractor ctx 里的 norm 键未被 counts 函数
      // 消费（q_a_norm 是独立 rmsnorm 叶，memory 教训 3：融合内已有的不再发独立叶
      // 的反向情形：独立 norm 叶在场，组合叶不再计）。
      weightMatrices: [
        weightMatrixDecl("tp", { shape: [qRank, dimWidth(dims.hidden)], split: "output" }),
      ],
    }, { input: dims.hidden, output: [-1, -1, qRank] }),
    operatorSpec(`${prefix}.q_a_norm`, "query latent RMSNorm", "rmsnorm", shapeFlow(qLatentShape, qLatentShape), { input: [-1, -1, qRank], output: [-1, -1, qRank] }),
    operatorSpec(`${prefix}.q_b_proj`, "query up projection", "linear", {
      ...shapeFlow(qLatentShape, qShape),
      implementation: ["vLLM.q_b_proj", "SGLang.q_b_proj"],
    }, { input: [-1, -1, qRank], output: [-1, -1, heads, qkDim] }),
    operatorSpec(`${prefix}.kv_a_proj`, "KV compression projection", "mla_kv_compress", {
      ...shapeFlow(shapes.hidden, kvLatentShape),
      implementation: ["vLLM.kv_a_proj_with_mqa", "SGLang.kv_a_proj_with_mqa"],
      kv_lora_rank: kvRank,
      qk_rope_head_dim: ropeDim,
      weightMatrices: [weightMatrixDecl("tp", { shape: [kvRank + ropeDim, dimWidth(dims.hidden)], split: "output" })],
    }, { input: dims.hidden, output: [-1, -1, kvRank + ropeDim] }),
    operatorSpec(`${prefix}.kv_split`, "KV latent and rope split", "mla_kv_split", {
      ...shapeFlow(kvLatentShape, `[batch, sequence, kv latent=${kvRank}], [batch, sequence, rope=${ropeDim}]`),
      split_sizes: [kvRank, ropeDim],
    }, { input: [-1, -1, kvRank + ropeDim], output: [-1, -1, kvRank] }),
    operatorSpec(`${prefix}.kv_a_norm`, "KV latent RMSNorm", "rmsnorm", shapeFlow(`[batch, sequence, kv latent=${kvRank}]`, `[batch, sequence, kv latent=${kvRank}]`), { input: [-1, -1, kvRank], output: [-1, -1, kvRank] }),
    operatorSpec(`${prefix}.kv_b_proj`, "KV expansion projection", "linear", {
      ...shapeFlow(`[batch, sequence, kv latent=${kvRank}]`, `${kShape}, ${vShape}`),
      implementation: ["vLLM.kv_b_proj", "SGLang.kv_b_proj"],
      qk_nope_head_dim: qkNope,
      value_head_dim: valueDim,
    }, { input: [-1, -1, kvRank], output: [-1, -1, heads, qkNope + valueDim] }),
    operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope", {
      ...shapeFlow(`${qShape}, ${kShape}`, `${qShape}, ${kShape}`),
      qk_rope_head_dim: ropeDim,
      implementation: ["vLLM.DeepseekV32 rotary_emb", "SGLang.Deepseek rotary_emb"],
    }, { input: [-1, -1, heads, qkDim], output: [-1, -1, heads, qkDim] }),
    operatorSpec(`${prefix}.indexer.q_proj`, "indexer query projection", "linear", {
      ...shapeFlow(qLatentShape, `[batch, sequence, index heads=${indexHeads}, index head dimension=${indexDim}]`),
      implementation: ["vLLM.Indexer.wq_b", "SGLang.Indexer.wq_b"],
    }, { input: [-1, -1, qRank], output: [-1, -1, indexHeads, indexDim] }),
    operatorSpec(`${prefix}.indexer.wk_weights_proj`, "indexer key and weight projection", "linear", {
      ...shapeFlow(shapes.hidden, `[batch, sequence, index head dimension=${indexDim}] + [batch, sequence, index heads=${indexHeads}]`),
      projection_layout: ["wk", "weights"],
      implementation: ["vLLM.Indexer.wk_weights_proj", "SGLang.Indexer.wk_weights_proj"],
    }, { input: dims.hidden, output: [-1, -1, indexDim + indexHeads] }),
    operatorSpec(`${prefix}.indexer.k_norm`, "indexer key LayerNorm", "rmsnorm", {
      ...shapeFlow(`[batch, sequence, index head dimension=${indexDim}]`, `[batch, sequence, index head dimension=${indexDim}]`),
      // DSA indexer 的 key norm 在 transformers 真值里是 nn.LayerNorm（weight+bias=2×width），
      // 非 RMSNorm；affine_bias 让声明含 bias，参数量与后端 LayerNorm 一致（结构对账实证）。
      affine_bias: true,
      implementation: ["vLLM.Indexer.k_norm", "SGLang.Indexer.k_norm"],
    }, { input: [-1, -1, indexDim], output: [-1, -1, indexDim] }),
    operatorSpec(`${prefix}.indexer`, kpool > 1 ? "DSA indexer (k-pool)" : "DSA indexer", kpool > 1 ? "dsa_kpool_indexer" : "dsa_indexer", {
      ...shapeFlow(shapes.hidden, `[batch, sequence, selected=${budget}]`),
      indexer_heads: indexHeads,
      indexer_head_dim: indexDim,
      budget,
      index_kpool: kpool > 1 ? kpool : undefined,
      indexer_mode: indexerMode,
      reuse_previous_indices: indexerMode === "reuse",
      implementation: kpool > 1
        ? ["vLLM.SparseAttnIndexerKpool", "SGLang.dsa_indexer kpool"]
        : ["vLLM.SparseAttnIndexer", "SGLang.dsa_indexer"],
    }, { input: dims.hidden, output: [-1, -1, budget] }),
    operatorSpec(`${prefix}.sparse_attention`, "DSA sparse MLA attention", "dsa_sparse_mla", {
      ...shapeFlow(`${qShape}, selected ${kShape}, selected ${vShape}`, `[batch, sequence, attention heads=${heads}, value head dimension=${valueDim}]`),
      selected_tokens: budget,
      attention_kind: "dsa_sparse_mla",
      indexer_mode: indexerMode,
      ...cacheResidentDecl({
        kvElements: kvRank + ropeDim,
        indexElements: indexDim || 0,
      }),
      implementation: ["vLLM.DeepseekV32MLAAttention", "SGLang.RadixAttention + DSA backend"],
    }, { input: [-1, -1, heads, qkDim], output: [-1, -1, heads, valueDim] }),
    operatorSpec(`${prefix}.o_proj`, "output projection", "linear", {
      ...shapeFlow(`[batch, sequence, attention heads=${heads}, value head dimension=${valueDim}]`, shapes.hidden),
      implementation: ["vLLM.o_proj", "SGLang.o_proj"],
      communication_role: "tp_attention_output",
    }, { input: [-1, -1, heads, valueDim], output: dims.hidden }),
  ];
}

// W4：残差加。此前每层两处 `h = x + sublayer(x)` 完全没有算子位——文档
// B-layer-res 已自登记「2TH·b x2/层未计」。上游对应 SGLang
// `srt/layers/attn_residual.py` 与各 model 文件里的 `hidden_states + residual`
// 融合入口（vLLM 走 RMSNorm 的 residual 形参做 add+norm 融合）。
export function residualAddSpec(id, normalized, label) {
  const { shapes, dims } = shapesAndDims(normalized);
  return operatorSpec(id, `${label} residual add`, "residual_add", {
    ...shapeFlow(`${shapes.hidden}, ${shapes.hidden}`, shapes.hidden),
    residual_of: label,
    implementation: ["vLLM.RMSNorm(residual=...) 融合 add", "SGLang.attn_residual"],
  }, { input: dims.hidden, output: dims.hidden });
}

export function layerInSpec(id, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  return operatorSpec(id, "layer input", "identity", {
    ...shapeFlow(shapes.hidden, shapes.hidden),
  }, { input: dims.hidden, output: dims.hidden });
}

export function mlpOperatorSpecs(prefix, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  // N2-4 层 1：dense MLP 三投影的权重声明（tp 亲和）。shared expert 复用本函数。
  return [
    operatorSpec(`${prefix}.gate_proj`, "gate projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.intermediate),
      weightMatrices: [weightMatrixDecl("tp", { shape: [dimWidth(dims.intermediate), dimWidth(dims.hidden)], split: "output" })],
    }, { input: dims.hidden, output: dims.intermediate }),
    operatorSpec(`${prefix}.up_proj`, "up projection", "linear", {
      ...shapeFlow(shapes.hidden, shapes.intermediate),
      weightMatrices: [weightMatrixDecl("tp", { shape: [dimWidth(dims.intermediate), dimWidth(dims.hidden)], split: "output" })],
    }, { input: dims.hidden, output: dims.intermediate }),
    operatorSpec(`${prefix}.swiglu`, "SwiGLU activation", "swiglu", {
      ...shapeFlow(`${shapes.intermediate}, ${shapes.intermediate}`, shapes.intermediate),
      gate_shape: shapes.intermediate,
      up_shape: shapes.intermediate,
      activation: recipeValue(normalized, "swigluVariant"),
      swiglu_alpha: normalized.swigluAlpha,
      swiglu_beta: normalized.swigluBeta,
      swiglu_limit: normalized.swigluLimit,
    }, { input: dims.intermediate, output: dims.intermediate }),
    operatorSpec(`${prefix}.down_proj`, "down projection", "linear", {
      ...shapeFlow(shapes.intermediate, shapes.hidden),
      communication_role: "tp_mlp_output",
      weightMatrices: [weightMatrixDecl("tp", { shape: [dimWidth(dims.hidden), dimWidth(dims.intermediate)], split: "input" })],
    }, { input: dims.intermediate, output: dims.hidden }),
  ];
}

// MoE 分组受限 top-k（vLLM/SGLang grouped_topk）：n_group>1 时组内选 topk_group 组再选专家。
// n_group 缺省或=1 → 普通 top-k（不加标注，结构哈希不变）。
function groupedTopkAttrs(normalized) {
  const nGroup = normalized.numExpertGroup;
  if (!(nGroup > 1)) return {};
  return {
    topk_method: "group_limited_topk",
    num_expert_group: nGroup,
    topk_group: normalized.topkGroup,
    implementation: ["vLLM.grouped_topk", "SGLang.biased_grouped_topk"],
  };
}

// 路由（gate）叶权重：router linear 的 gate 权重 [experts, hidden]（replicated，与
// linearWeightMatrices 对 router 的自动声明逐位一致——dims.routerLogits=[..,experts]、
// dims.hidden=[..,hidden]）＋ noaux_tc 负载均衡的 e_score_correction_bias（[experts] fp32，
// SGLang models/deepseek_v2.py:492；checkpoint gate.bias → gate.e_score_correction_bias，
// 用于 (biased_)grouped_topk 分数修正）＋ 视觉路由 bias bias_vl（checkpoint
// layers.N.ffn.gate.bias_vl，仅路由修正 bias × 视觉塔 × 存在 compress_ratios 时出现；SGLang 源码无此符号 = 文本推理路径不消费）。
// 非 noaux 模型返回 undefined → 交回 linearWeightMatrices 自动声明（行为不变）。
// noaux 模型显式声明时必须重放 gate 权重，否则会顶掉自动声明、漏计 gate 矩阵。
// bias 均为非量化 replicated 一维参数，进参数量与 checkpoint 张量对账，不进量化字节 / KV。
function routerWeightMatrices(normalized) {
  if (!normalized.routerCorrectionBias) return undefined;
  const experts = normalized.experts || 0;
  const hidden = normalized.hiddenSize || 0;
  const groups = [
    weightMatrixDecl("replicated", { shape: [experts, hidden] }),
    weightMatrixDecl("replicated", { shape: [experts], quantizable: false, param_dtype: "router_correction_bias" }),
  ];
  if (normalized.routerBiasVl) {
    groups.push(weightMatrixDecl("replicated", { shape: [experts], quantizable: false, param_dtype: "router_bias_vl" }));
  }
  return groups;
}

export function moeOperatorSpecs(prefix, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  const isSigmoidRouter = String(normalized.scoringFunc || "").toLowerCase() === "sigmoid"
    || recipeFlag(normalized, "sigmoidRouter");
  return [
    operatorSpec(`${prefix}.router`, "router logits", "linear", {
      ...shapeFlow(shapes.hidden, shapes.routerLogits),
      scoring_func: isSigmoidRouter ? "sigmoid" : undefined,
      routing_bias: isSigmoidRouter ? true : undefined,
      implementation: isSigmoidRouter ? ["vLLM.GateLinear fp32 router", "SGLang.GateLinear fp32 router"] : undefined,
      weightMatrices: routerWeightMatrices(normalized),
    }, { input: dims.hidden, output: dims.routerLogits }),
    operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk", {
      ...shapeFlow(shapes.routerLogits, `${shapes.topExperts}, ${shapes.topExperts}`),
      expert_ids_shape: shapes.topExperts,
      expert_weights_shape: shapes.topExperts,
      scoring_func: isSigmoidRouter ? "sigmoid" : undefined,
      ...groupedTopkAttrs(normalized),
    }, { input: dims.routerLogits, output: dims.topExperts }),
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.topExperts}`, shapes.expertInput),
      token_shape: shapes.hidden,
      expert_ids_shape: shapes.topExperts,
      communication_role: "ep_dispatch",
    }, { input: dims.hidden, output: dims.expertInput }),
    operatorSpec(`${prefix}.expert_mlp`, "expert MLP", "fused_moe_mlp", {
      ...shapeFlow(shapes.expertInput, shapes.expertInput),
      intermediate_shape: shapes.moeIntermediate,
      activation: recipeValue(normalized, "swigluVariant") ? `${recipeValue(normalized, "swigluVariant")}_uninterleave` : undefined,
      swiglu_alpha: normalized.swigluAlpha,
      swiglu_beta: normalized.swigluBeta,
      swiglu_limit: normalized.swigluLimit,
      // N2-4 W-A：专家 gate/up/down 三矩阵在叶内声明（ep 组）。此前该叶与纯激活
      // 共用 swiglu id、权重对量化/投影/容量不可见——正是 N2-4 的枚举缺口。
      weightMatrices: routedExpertWeightMatrices(normalized),
    }, { input: dims.expertInput, output: dims.expertInput }),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine", {
      ...shapeFlow(`${shapes.expertInput}, ${shapes.topExperts}`, shapes.hidden),
      expert_output_shape: shapes.expertInput,
      expert_weights_shape: shapes.topExperts,
      communication_role: "ep_combine",
    }, { input: dims.expertInput, output: dims.hidden }),
  ];
}

// DeepSeek V4 的 hash 层和普通 MoE 共用同一 routed/shared expert 语义；差异只在路由节点。
export function deepseekV4MoeOperatorSpecs(prefix, normalized, isHashMoe = false) {
  const { shapes, dims } = shapesAndDims(normalized);
  const specs = isHashMoe
    ? [operatorSpec(`${prefix}.hash_router`, "input-id hash expert routing", "dsv4_hash_route", {
      ...shapeFlow("[batch, sequence] input_ids", shapes.topExperts),
      num_hash_layers: normalized.numHashLayers,
      hash_table_shape: `[vocab size=${normalized.vocabSize}, experts per token=${normalized.expertsPerToken}]`,
      // tid2eid 是 buffer 不是参数（Megatron-Bridge）。每层一张 vocab×k 表。
      buffer_elements: (normalized.vocabSize || 0) * (normalized.expertsPerToken || 0),
      implementation: ["vLLM.gate.tid2eid + fused_topk_bias", "SGLang DeepSeek V4 hash routing"],
    }, { input: [-1, -1], output: dims.topExperts })]
    : [
      operatorSpec(`${prefix}.router`, "router logits", "linear", {
        ...shapeFlow(shapes.hidden, shapes.routerLogits),
        scoring_func: "sqrtsoftplus",
        routed_scaling_factor: normalized.routedScalingFactor,
        implementation: ["vLLM.GateLinear + fused_topk_bias", "SGLang fused_moe"],
        weightMatrices: routerWeightMatrices(normalized),
      }, { input: dims.hidden, output: dims.routerLogits }),
      operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk", {
        ...shapeFlow(shapes.routerLogits, `${shapes.topExperts}, ${shapes.topExperts}`),
        expert_ids_shape: shapes.topExperts,
        expert_weights_shape: shapes.topExperts,
        scoring_func: "sqrtsoftplus",
        renormalize: normalized.normTopkProb,
        ...groupedTopkAttrs(normalized),
      }, { input: dims.routerLogits, output: dims.topExperts }),
    ];
  specs.push(
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.topExperts}`, shapes.expertInput),
      token_shape: shapes.hidden,
      expert_ids_shape: shapes.topExperts,
      implementation: ["vLLM.FusedMoE", "SGLang fused_moe"],
      communication_role: "ep_dispatch",
    }, { input: dims.hidden, output: dims.expertInput }),
    operatorSpec(`${prefix}.expert_mlp`, "expert SwiGLU", "fused_moe_mlp", {
      ...shapeFlow(shapes.expertInput, shapes.expertInput),
      intermediate_shape: shapes.moeIntermediate,
      swiglu_limit: normalized.swigluLimit,
      implementation: ["vLLM.DeepseekV4MegaMoEExperts", "SGLang fused_moe"],
      weightMatrices: routedExpertWeightMatrices(normalized),
    }, { input: dims.expertInput, output: dims.expertInput }),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine", {
      ...shapeFlow(`${shapes.expertInput}, ${shapes.topExperts}`, shapes.hidden),
      expert_output_shape: shapes.expertInput,
      expert_weights_shape: shapes.topExperts,
      routed_scaling_factor: normalized.routedScalingFactor,
      communication_role: "ep_combine",
    }, { input: dims.expertInput, output: dims.hidden }),
  );
  return specs;
}

export function kimiK3MoeOperatorSpecs(prefix, normalized) {
  const { shapes, dims } = shapesAndDims(normalized);
  const latent = normalized.routedExpertHiddenSize;
  const latentShape = `[tokens_per_expert, routed expert hidden size=${latent}]`;
  const latentDims = [-1, latent];
  return [
    operatorSpec(`${prefix}.router`, "router logits", "linear", { ...shapeFlow(shapes.hidden, shapes.routerLogits), weightMatrices: routerWeightMatrices(normalized) }, { input: dims.hidden, output: dims.routerLogits }),
    operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk", {
      ...shapeFlow(shapes.routerLogits, `${shapes.topExperts}, ${shapes.topExperts}`),
      expert_ids_shape: shapes.topExperts,
      expert_weights_shape: shapes.topExperts,
      ...groupedTopkAttrs(normalized),
    }, { input: dims.routerLogits, output: dims.topExperts }),
    operatorSpec(`${prefix}.routed_expert_down_proj`, "routed expert latent down projection", "linear", {
      ...shapeFlow(shapes.hidden, latentShape),
      latent_size: latent,
      semantic_role: "latent_moe_compress",
    }, { input: dims.hidden, output: [-1, latent] }),
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch", {
      ...shapeFlow(`${latentShape}, ${shapes.topExperts}`, latentShape),
      token_shape: latentShape,
      expert_ids_shape: shapes.topExperts,
      communication_role: "ep_dispatch",
    }, { input: latentDims, output: latentDims }),
    operatorSpec(`${prefix}.expert_mlp`, "latent expert MLP", "fused_moe_mlp", {
      ...shapeFlow(latentShape, latentShape),
      intermediate_shape: shapes.moeIntermediate,
      latent_size: latent,
      weightMatrices: routedExpertWeightMatrices(normalized),
    }, { input: latentDims, output: latentDims }),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine", {
      ...shapeFlow(`${latentShape}, ${shapes.topExperts}`, latentShape),
      expert_output_shape: latentShape,
      expert_weights_shape: shapes.topExperts,
      communication_role: "ep_combine",
    }, { input: latentDims, output: latentDims }),
    operatorSpec(`${prefix}.routed_expert_norm`, "routed expert latent RMSNorm", "rmsnorm", {
      ...shapeFlow(latentShape, latentShape),
      semantic_role: "latent_moe_reduce_norm",
    }, { input: latentDims, output: latentDims }),
    operatorSpec(`${prefix}.routed_expert_up_proj`, "routed expert latent up projection", "linear", {
      ...shapeFlow(latentShape, shapes.hidden),
      latent_size: latent,
      semantic_role: "latent_moe_expand",
    }, { input: latentDims, output: dims.hidden }),
    operatorSpec(`${prefix}.shared_expert_add`, "shared expert branch add", "moe_add", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.hidden}`, shapes.hidden),
      shared_experts: normalized.sharedExperts,
    }, { input: dims.hidden, output: dims.hidden }),
  ];
}
