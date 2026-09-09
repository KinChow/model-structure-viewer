// modules.js —— 模块层（本体 L1/L2）。
//
// 定位与对标：本层对应 vLLM 的 nn.Module 类，命名直接借上游。
// - 共享模块 = `vllm/vllm/model_executor/layers/`（linear / layernorm / activation /
//   conv / rotary_embedding / fused_moe / vocab_parallel_embedding / mla /
//   lightning_attn / mhc / sparse_attn_indexer / sparse_attn_indexer_kpool）
// - 注意力形态 = `model_executor/layers/attention/`（Attention / MLAAttention /
//   SparseMLAAttention / ChunkedLocalAttention / RSWAAttention / StaticSinkAttention）
// - 每模型私有模块 = `vllm/vllm/models/<model_type>/`
// **内核不建模**（vLLM `v1/attention/backends/*`、`*Impl`/`*Backend`；SGLang
// `srt/layers/attention/*_backend.py`），只在 `source` 字段记出处。
//
// 每条目形态：
//   { id, title, source, fused(p), decompose(p), residentIntermediates(p), notes }
// - fused：闭式融合公式。**镜像当前实现口径**（委托 counts.js，不复制逻辑），
//   这样恒等式报表里的差额就等于「现状 − 第一性原理」，即待修清单。
// - decompose：原子序列（只允许 atoms.js 的 18 个 id），第一性原理口径。
// - residentIntermediates：融合收益的显式清单——分解下会落 HBM、融合下留在
//   寄存器/SRAM 的中间量，逐项给元素数。护栏据此核对 bytes 差额是否可解释。
//
// 护栏判据（__tests__/identities.test.js，W1 warn 模式 / W4 转 error）：
//   fused.matrix|vector|sfu === Σ decompose（整数相等）
//   Σ decompose.bytes − fused.bytes === Σ 2·residentIntermediates.elements·b

import { evaluateDecomposition } from "./atoms.js";
/** 与 formulas/index.js 的 sumCounts 同口径：逐分量相加。复合模块用。 */
const sumCounts = (...parts) => parts.reduce((total, part) => ({
  matrix: total.matrix + part.matrix,
  vector: total.vector + part.vector,
  sfu: total.sfu + part.sfu,
  bytes: {
    weights: total.bytes.weights + part.bytes.weights,
    actIn: total.bytes.actIn + part.bytes.actIn,
    actOut: total.bytes.actOut + part.bytes.actOut,
  },
}), { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } });
import {
  addCounts,
  attentionCounts,
  causalConvCounts,
  gateCounts,
  hashRouteCounts,
  rearrangeCounts,
  linearCounts,
  linearAttentionStateCounts,
  rmsnormCounts,
  ropeCounts,
  softmaxCounts,
  swigluCounts,
  topkCounts,
  scoredPairs,
  causalDensity,
} from "./counts.js";

/**
 * 相位相关的打分对数（per head）。**唯一实现在 counts.js `scoredPairs`**，
 * 本处只做签名适配（modules 的 p 是扁平参数），不复制逻辑。
 */
export function scorePairs(phase, queryTokens, keyTokens) {
  return scoredPairs({ phase, queryTokens, keyTokens });
}

/** W3 前的口径（两相位通吃 T·S）。仅报表对比用，不参与计费。 */
export function scorePairsLegacy(queryTokens, keyTokens) {
  return queryTokens * keyTokens;
}

/** 因果/稀疏密度：实际打分对数 / 稠密对数。喂给 atoms.matmul 的 density。 */
export function scoreDensity(phase, queryTokens, keyTokens) {
  return causalDensity({ phase, queryTokens, keyTokens });
}

// ===========================================================================
// 模块注册表（W1 首批：覆盖 16 结构类的公共叶模块）
// ===========================================================================

// ===========================================================================
// 可复用的原子分解片段
//
// 复合模块（mla_query_compress = linear + rmsnorm 等）直接拼这些片段，
// 而不是把 linear/rmsnorm 的分解体抄第二遍 —— 抄一遍就会漂移一次。
// ===========================================================================

/** linear 的原子分解（p: {tokens, inDim, out, b, bias, expertFraction}）。 */
function linearDecompose(p) {
  return [
    { atom: "matmul", args: { batch: 1, m: p.tokens * (p.expertFraction ?? 1), k: p.inDim, n: p.out, bytesPerElement: p.b, rhs: "weight", outElements: p.tokens * p.out } },
    ...(p.bias ? [{ atom: "add", args: { elements: p.tokens * p.out, bytesPerElement: p.b, weightElements: p.out } }] : []),
  ];
}
function linearResident(p) {
  return p.bias ? [{ name: "GEMM 输出在 epilogue 内加 bias", elements: p.tokens * p.out }] : [];
}
function linearCompulsory(p) {
  return (p.tokens * p.inDim + p.tokens * p.out) * p.b + (p.out * p.inDim + (p.bias ? p.out : 0)) * p.b;
}

/** rmsnorm 的原子分解（p: {tokens, hidden, b, weightOne, gated}）。 */
function rmsnormDecompose(p) {
  const e = p.tokens * p.hidden;
  return [
    { atom: "mul", args: { elements: e, bytesPerElement: p.b } },
    { atom: "reduce_sum", args: { elements: e, groups: p.tokens, bytesPerElement: p.b } },
    { atom: "rsqrt", args: { elements: p.tokens, bytesPerElement: p.b } },
    { atom: "mul", args: { elements: e, bytesPerElement: p.b } },
    { atom: "mul", args: { elements: e, bytesPerElement: p.b, weightElements: p.hidden } },
    ...(p.weightOne ? [{ atom: "add", args: { elements: e, bytesPerElement: p.b } }] : []),
    ...(p.gated ? [
      { atom: "sigmoid", args: { elements: e, bytesPerElement: p.b } },
      { atom: "mul", args: { elements: e, bytesPerElement: p.b } },
    ] : []),
  ];
}
function rmsnormResident(p) {
  const e = p.tokens * p.hidden;
  return [
    { name: "x 的平方", elements: e },
    { name: "均方和", elements: p.tokens },
    { name: "rstd", elements: p.tokens },
    { name: "x·rstd", elements: e },
    ...(p.weightOne ? [{ name: "(1+w) 缩放中间态", elements: e }] : []),
    ...(p.gated ? [{ name: "sigmoid(gate)", elements: e }] : []),
  ];
}
function rmsnormCompulsory(p) {
  return (2 * p.tokens * p.hidden + (p.gated ? p.tokens * p.hidden : 0)) * p.b + p.hidden * p.b;
}

const MODULE_LIST = [
  {
    id: "linear",
    title: "Linear Projection",
    source: { framework: "vLLM", symbol: "ColumnParallelLinear / RowParallelLinear / QKVParallelLinear", ref: "model_executor/layers/linear.py" },
    fused: (p) => linearCounts({ logicalShape: [p.out, p.inDim], tokens: p.tokens, bytesPerElement: p.b, bias: p.bias, expertFraction: p.expertFraction ?? 1 }),
    decompose: linearDecompose,
    residentIntermediates: linearResident,
    compulsoryBytes: linearCompulsory,
  },
  {
    id: "rmsnorm",
    title: "RMSNorm",
    source: { framework: "vLLM", symbol: "RMSNorm / GemmaRMSNorm", ref: "model_executor/layers/layernorm.py" },
    fused: (p) => rmsnormCounts({ tokens: p.tokens, hidden: p.hidden, bytesPerElement: p.b, weightOne: p.weightOne, gated: p.gated }),
    decompose: rmsnormDecompose,
    residentIntermediates: rmsnormResident,
    notes: ["F3 的 vector = 4·T·H 是取整口径；逐原子分解为 4·T·H − T（reduce 每组少一次加法）"],
    compulsoryBytes: rmsnormCompulsory,
  },
  {
    id: "rope",
    title: "Rotary Position Embedding",
    source: { framework: "vLLM", symbol: "RotaryEmbedding", ref: "model_executor/layers/rotary_embedding/" },
    fused: (p) => ropeCounts({ tokens: p.tokens, ropeDims: p.ropeDims, bytesPerElement: p.b }),
    decompose: (p) => [{ atom: "rope", args: { elements: p.tokens * p.ropeDims, bytesPerElement: p.b } }],
    residentIntermediates: () => [],
    notes: ["A3：sin/cos 查表，sfu 精确零", "F6 的 actOut = 2·T·D（q 与 k 各一份），逐原子为 T·D"],
    compulsoryBytes: (p) => 3 * p.tokens * p.ropeDims * p.b,
  },
  {
    id: "swiglu",
    title: "SiluAndMul",
    source: { framework: "vLLM", symbol: "SiluAndMul", ref: "model_executor/layers/activation.py" },
    fused: (p) => swigluCounts({ tokens: p.tokens, intermediate: p.intermediate, bytesPerElement: p.b }),
    decompose: (p) => {
      const e = p.tokens * p.intermediate;
      return [
        { atom: "silu", args: { elements: e, bytesPerElement: p.b } },
        { atom: "mul", args: { elements: e, bytesPerElement: p.b } },
      ];
    },
    residentIntermediates: (p) => [{ name: "silu(gate)", elements: p.tokens * p.intermediate }],
    compulsoryBytes: (p) => 3 * p.tokens * p.intermediate * p.b,
  },
  {
    id: "gate",
    title: "Sigmoid Output Gate",
    source: { framework: "vLLM", symbol: "fused sigmoid-mul epilogue", ref: "model_executor/models/qwen3_next.py Qwen3NextAttention" },
    fused: (p) => gateCounts({ tokens: p.tokens, width: p.width, bytesPerElement: p.b, gateProjection: p.gateProjection, gateProjectionInput: p.gateProjectionInput ?? 0 }),
    decompose: (p) => {
      const e = p.tokens * p.width;
      return [
        ...(p.gateProjection ? [{ atom: "matmul", args: { batch: 1, m: p.tokens, k: p.gateProjectionInput, n: p.width, bytesPerElement: p.b, rhs: "weight" } }] : []),
        { atom: "sigmoid", args: { elements: e, bytesPerElement: p.b } },
        { atom: "mul", args: { elements: e, bytesPerElement: p.b } },
      ];
    },
    residentIntermediates: (p) => [{ name: "sigmoid(G)", elements: p.tokens * p.width }],
    compulsoryBytes: (p) => 2 * p.tokens * p.width * p.b + (p.gateProjection ? p.gateProjectionInput * p.width * p.b : 0),
  },
  {
    id: "softmax",
    title: "Softmax",
    source: { framework: "aten", symbol: "aten._softmax", ref: "torch.utils.flop_counter" },
    fused: (p) => softmaxCounts({ elements: p.elements, bytesPerElement: p.b }),
    decompose: (p) => [{ atom: "softmax", args: { elements: p.elements, bytesPerElement: p.b } }],
    residentIntermediates: () => [],
    compulsoryBytes: (p) => 2 * p.elements * p.b,
  },
  {
    id: "topk_router",
    title: "MoE Top-k Routing",
    source: { framework: "vLLM", symbol: "FusedMoE.select_experts", ref: "model_executor/layers/fused_moe/" },
    fused: (p) => topkCounts({ tokens: p.tokens, experts: p.experts, topk: p.topk, bytesPerElement: p.b, normTopkProb: p.normTopkProb }),
    decompose: (p) => [
      { atom: "topk", args: { rows: p.tokens, candidates: p.experts, k: p.topk, bytesPerElement: p.b } },
      ...(p.normTopkProb ? [
        { atom: "reduce_sum", args: { elements: p.tokens * p.topk, groups: p.tokens, bytesPerElement: p.b } },
        { atom: "div", args: { elements: p.tokens * p.topk, bytesPerElement: p.b } },
      ] : []),
    ],
    residentIntermediates: (p) => (p.normTopkProb ? [{ name: "top-k 权重和", elements: p.tokens }] : []),
    notes: ["F8 的 sfu = T·k 即归一化除法；逐原子分解为 reduce_sum + div"],
    compulsoryBytes: (p) => p.tokens * p.experts * p.b + p.tokens * p.topk * 4,
  },
  {
    // 复合模块：运行时 counts 就是 F1(q_a)（formulas/index.js）。
    // **norm 与 q_b 都不在本模块内** —— q_a_layernorm 是独立的 `q_a_norm` 叶、
    // q_b_proj 是独立 linear 叶，算进来就是双计（两处双计分别由
    // 2026-09-07 参数审计与 2026-09-09 权重字节逐层归因抓出）。
    id: "mla_query_compress",
    title: "MLA Query Compression",
    source: { framework: "vLLM", symbol: "MLAModules.q_a_proj", ref: "model_executor/layers/mla.py" },
    fused: (p) => linearCounts({ logicalShape: [p.rank, p.hidden], tokens: p.tokens, bytesPerElement: p.b }),
    decompose: (p) => linearDecompose({ tokens: p.tokens, inDim: p.hidden, out: p.rank, b: p.b }),
    residentIntermediates: (p) => linearResident({ tokens: p.tokens, out: p.rank, b: p.b }),
    compulsoryBytes: (p) => linearCompulsory({ tokens: p.tokens, inDim: p.hidden, out: p.rank, b: p.b }),
    notes: ["q_a_layernorm 与 q_b_proj 都是独立叶，不在本模块内，否则双计"],
  },
  {
    // 复合模块：运行时 counts = F1(kv_a 投影) + rearrange（视图语义，全零）
    //（formulas/index.js:300）。latent cache 的写入就是这条 linear 的 actOut。
    id: "mla_kv_compress",
    title: "MLA KV Compression",
    source: { framework: "vLLM", symbol: "MLAModules.kv_a_proj_with_mqa", ref: "model_executor/layers/mla.py" },
    fused: (p) => linearCounts({ logicalShape: [p.out, p.hidden], tokens: p.tokens, bytesPerElement: p.b }),
    decompose: (p) => linearDecompose({ tokens: p.tokens, inDim: p.hidden, out: p.out, b: p.b }),
    residentIntermediates: (p) => linearResident({ tokens: p.tokens, out: p.out, b: p.b }),
    compulsoryBytes: (p) => linearCompulsory({ tokens: p.tokens, inDim: p.hidden, out: p.out, b: p.b }),
    notes: ["latent/rope 的拆分是视图（mla_kv_split 零流量），不进分解"],
  },
  {
    // DeepSeek V4 的哈希路由。2026-09-09 分类裁决：tid2eid 是 **buffer 不是
    // 参数**（Megatron-Bridge："Buffers are not parameters"；MaxText 同；
    // 出处 = Hash Layers, Roller et al. 2021）——表的常驻容量（vocab·k·4B
    // int32）由 derivedBufferBytes 计入显存，不进权重字节恒等式；本模块只计
    // gather 的真实拷贝。此前「表算权重」的口径与 gather 的物理读数冲突，
    // 是 DECOMPOSE_PENDING 里唯一一条「口径冲突」而非「工作量」的待办。
    id: "dsv4_hash_route",
    title: "DeepSeek V4 Hash MoE Routing",
    source: { framework: "vLLM", symbol: "DeepseekV4MoE hash routing (tid2eid)", ref: "models/deepseek_v4/" },
    fused: (p) => hashRouteCounts({ tokens: p.tokens, topk: p.topk, bytesPerElement: p.b }),
    decompose: (p) => [{ atom: "gather", args: { rows: p.tokens, width: p.topk, bytesPerElement: p.b } }],
    residentIntermediates: () => [],
    compulsoryBytes: (p) => 2 * p.tokens * p.topk * p.b,
    notes: [
      "tid2eid 是 buffer：容量走 derivedBufferBytes，不进权重字节恒等式",
      "gather 读 = 写 = tokens·topk（每 token 取 topk 个专家 id）",
    ],
  },
  {
    // 视觉位置编码：加法（learned 2D / rope 变体的实现差异不影响一阶口径——
    // 都是「读位置向量 + 逐元素加」）。与 addCounts 逐位同构。
    id: "vision_position",
    title: "Vision Position Embedding",
    source: { framework: "vLLM", symbol: "Qwen2_5VisionRotaryEmbedding / vision position embedding", ref: "model_executor/models/qwen2_5_vision_navigation.py" },
    fused: (p) => addCounts({ tokens: p.tokens, hidden: p.hidden, bytesPerElement: p.b }),
    decompose: (p) => [{ atom: "add", args: { elements: p.tokens * p.hidden, bytesPerElement: p.b } }],
    residentIntermediates: () => [],
    compulsoryBytes: (p) => 3 * p.tokens * p.hidden * p.b,
    notes: ["addCounts 与 add 原子逐位同构（vector 1/元素、actIn 2E·b、actOut E·b）"],
  },
  {
    // 视觉 patch merge：[V, H] → [V/merge², merge²·H] 的物化拷贝（非视图——
    // 分辨率动态、无法 strided 表达）。两端总元素数相等（V·H），
    // 与 permute_copy 原子逐位同构。运行时口径同步补了 G1 的 T_v 缺口
    //（此前 inElements/outElements 是单 token 宽度，漏乘 token 数）。
    id: "vision_merge",
    title: "Vision Patch Merge",
    source: { framework: "vLLM", symbol: "Qwen2_5_VisionPatchMerger / vision patch merge", ref: "model_executor/models/qwen2_5_vision_navigation.py" },
    fused: (p) => rearrangeCounts({
      copy: true,
      inElements: p.tokens * p.inWidth,
      outElements: Math.max(1, Math.floor(p.tokens / (p.mergeSize * p.mergeSize))) * p.inWidth * p.mergeSize * p.mergeSize,
      bytesPerElement: p.b,
    }),
    decompose: (p) => [{ atom: "permute_copy", args: { elements: p.tokens * p.inWidth, bytesPerElement: p.b } }],
    residentIntermediates: () => [],
    compulsoryBytes: (p) => 2 * p.tokens * p.inWidth * p.b,
    notes: ["tokens 必须能被 merge² 整除（内置模型均满足：2304/4、576/4）"],
  },
  {
    // 视觉激活（gelu_pytorch_tanh / gelu）：一阶口径与 silu 同构
    //（1 乘 + 2 SFU，erf/exp 差异登记在 §4 的已知近似，见 operators_reference）。
    // swigluCounts 是「读 x、y 两操作数」的融合口径（actIn 2E·b）；
    // 分解 = silu + mul（silu 读 x、mul 读中间量与 y）= actIn 3E·b，
    // 落在 bytes 夹逼内：compulsory = 2E·b ≤ fused 2E·b ≤ Σ分解 3E·b。
    id: "vision_activation",
    title: "Vision Activation",
    source: { framework: "vLLM", symbol: "Qwen2_5VisionMLP act / vision activation", ref: "model_executor/models/qwen2_5_vision_navigation.py" },
    fused: (p) => swigluCounts({ tokens: p.tokens, intermediate: p.intermediate, bytesPerElement: p.b }),
    decompose: (p) => [
      { atom: "silu", args: { elements: p.tokens * p.intermediate, bytesPerElement: p.b } },
      { atom: "mul", args: { elements: p.tokens * p.intermediate, bytesPerElement: p.b } },
    ],
    residentIntermediates: (p) => [{ name: "silu(x) 中间量", elements: p.tokens * p.intermediate }],
    compulsoryBytes: (p) => 2 * p.tokens * p.intermediate * p.b,
    notes: ["gelu 与 silu 的一阶口径差（erf vs sigmoid）登记为已知近似"],
  },
  {
    // K3 AttnResBlock 的**聚合叶**（两个 norm + 两个打分投影是独立叶，
    // 见 ops 模板与 2026-09-09 的双计修正）：对 prev_valid_blocks 个残差
    // 打分归一化后加权求和。softmax 原子与 softmaxCounts 逐位同构、
    // add 原子与 addCounts 逐位同构，分解逐位闭合。
    id: "attention_residual",
    title: "Attention Residual Aggregate",
    source: { framework: "vLLM", symbol: "KimiK3 attn_res aggregate", ref: "models/kimi_k3/amd/linear.py:562-580" },
    fused: (p) => sumCounts(
      softmaxCounts({ elements: p.tokens * p.hidden, bytesPerElement: p.b }),
      addCounts({ tokens: p.tokens, hidden: p.hidden, bytesPerElement: p.b }),
    ),
    decompose: (p) => [
      { atom: "softmax", args: { elements: p.tokens * p.hidden, bytesPerElement: p.b } },
      { atom: "add", args: { elements: p.tokens * p.hidden, bytesPerElement: p.b } },
    ],
    residentIntermediates: () => [],
    compulsoryBytes: (p) => 3 * p.tokens * p.hidden * p.b,
    notes: ["norm/proj 两对是独立叶（self_attention_res_* / mlp_res_*），不在本模块内"],
  },
  {
    // Qwen4Exp 的 delayed HyperConnection（GatedResidual，vLLM
    // qwen4_exp/common/hyperconnection.py:140-240）。七个组成部分全部
    // 用既有原子表达：grouped norm → rmsnorm 分解片段（weightOne）、
    // down/up/inject → linear 分解片段、SiLU → silu 原子（与 gateCounts
    // 逐位同构）、sigmoid 门 → sigmoid + mul、combine → add。
    id: "hyper_connection",
    title: "Hyper Connection",
    source: { framework: "vLLM", symbol: "GatedResidual", ref: "models/qwen4_exp/common/hyperconnection.py:140" },
    fused: (p) => {
      const hyperHidden = p.streams * p.hidden;
      return sumCounts(
        rmsnormCounts({ tokens: p.tokens, hidden: hyperHidden, bytesPerElement: p.b, weightOne: true }),
        linearCounts({ logicalShape: [p.lowrank, hyperHidden], tokens: p.tokens, bytesPerElement: p.b }),
        gateCounts({ tokens: p.tokens, width: p.lowrank, bytesPerElement: p.b }),
        linearCounts({ logicalShape: [hyperHidden, p.lowrank], tokens: p.tokens, bytesPerElement: p.b }),
        gateCounts({ tokens: p.tokens, width: hyperHidden, bytesPerElement: p.b }),
        linearCounts({ logicalShape: [p.streams, hyperHidden], tokens: p.tokens, bytesPerElement: p.b }),
        addCounts({ tokens: p.tokens, hidden: hyperHidden, bytesPerElement: p.b }),
      );
    },
    decompose: (p) => {
      const hyperHidden = p.streams * p.hidden;
      const e = p.tokens * hyperHidden;
      const el = p.tokens * p.lowrank;
      return [
        ...rmsnormDecompose({ tokens: p.tokens, hidden: hyperHidden, weightOne: true, b: p.b }),
        ...linearDecompose({ tokens: p.tokens, inDim: hyperHidden, out: p.lowrank, b: p.b }),
        { atom: "silu", args: { elements: el, bytesPerElement: p.b } },
        ...linearDecompose({ tokens: p.tokens, inDim: p.lowrank, out: hyperHidden, b: p.b }),
        { atom: "sigmoid", args: { elements: e, bytesPerElement: p.b } },
        { atom: "mul", args: { elements: e, bytesPerElement: p.b } },
        ...linearDecompose({ tokens: p.tokens, inDim: hyperHidden, out: p.streams, b: p.b }),
        { atom: "add", args: { elements: e, bytesPerElement: p.b } },
      ];
    },
    residentIntermediates: (p) => [
      { name: "hc_norm 的中间量组", elements: p.tokens * p.streams * p.hidden },
      { name: "SiLU 输出", elements: p.tokens * p.lowrank },
    ],
    compulsoryBytes: (p) => {
      const hyperHidden = p.streams * p.hidden;
      // 主输入（多流状态）+ 输出（混合后的 block 输入）+ 三份权重
      const weights = (2 * p.lowrank * hyperHidden + p.streams * hyperHidden) * p.b;
      return (2 * p.tokens * hyperHidden) * p.b + weights;
    },
    notes: [
      "gate 的 vector/sfu（gateCounts）= sigmoid + mul 两原子之和，逐位闭合",
      "inject 在最终 mixer（use_combine=false）由运行时置零，模块层按有 combine 的常规形态声明",
    ],
  },
  {
    // Qwen4Exp 指定层的 PLE（Position Learning Enhancement）：ngram 查表
    // （tid2eid 同类——ngram 嵌入表是 buffer，不进权重字节）→ KV 投影 →
    // grouped norm → 短卷积（silu 融合）→ 注回多流状态。
    id: "ple",
    title: "Position Learning Enhancement",
    source: { framework: "vLLM", symbol: "Qwen4Exp PLE", ref: "models/qwen4_exp/（Qwen modeling 未入库，离线取证）" },
    fused: (p) => sumCounts(
      hashRouteCounts({ tokens: p.tokens, topk: 1, bytesPerElement: p.b }),
      linearCounts({ logicalShape: [2 * p.embedDim, p.hidden], tokens: p.tokens, bytesPerElement: p.b }),
      rmsnormCounts({ tokens: p.tokens, hidden: p.embedDim, bytesPerElement: p.b }),
      causalConvCounts({ tokens: p.tokens, channels: p.embedDim, kernel: p.ngram, bytesPerElement: p.b }),
      addCounts({ tokens: p.tokens, hidden: p.hidden, bytesPerElement: p.b }),
    ),
    decompose: (p) => [
      { atom: "gather", args: { rows: p.tokens, width: 1, bytesPerElement: p.b } },
      ...linearDecompose({ tokens: p.tokens, inDim: p.hidden, out: 2 * p.embedDim, b: p.b }),
      ...rmsnormDecompose({ tokens: p.tokens, hidden: p.embedDim, b: p.b }),
      { atom: "conv1d", args: { tokens: p.tokens, channels: p.embedDim, kernel: p.ngram, bytesPerElement: p.b } },
      { atom: "silu", args: { elements: p.tokens * p.embedDim, bytesPerElement: p.b } },
      { atom: "add", args: { elements: p.tokens * p.hidden, bytesPerElement: p.b } },
    ],
    residentIntermediates: (p) => [
      { name: "卷积输出的 silu 中间量", elements: p.tokens * p.embedDim },
    ],
    compulsoryBytes: (p) => {
      const convWeights = p.embedDim * p.ngram * p.b;
      const kvWeights = 2 * p.embedDim * p.hidden * p.b;
      return (p.tokens * p.hidden * 2 + p.tokens * p.embedDim * 2) * p.b + kvWeights + convWeights;
    },
    notes: [
      "ngram 嵌入表是 buffer（与 tid2eid 同类），容量不在本模块（取证待 Qwen modeling 入库）",
      "conv 的 silu 融合段 = conv1d + silu 两原子（bytes 落夹逼）",
    ],
  },
  {
    // mHC 的 contract 段：把 hc_mult 条残差流收成单流 hidden（逐元素平均/加）。
    // 运行时 counts 就是 addCounts（formulas/index.js 的 mhc_contract），
    // 与 add 原子逐位同构。取证据：mhc kernel 的 stream collapse 无权重
    //（HCHeadOp 另计，见 agent 取证 2026-09-09）。
    id: "mhc_contract",
    title: "mHC Stream Contract",
    source: { framework: "vLLM", symbol: "mHC stream collapse", ref: "models/deepseek_v4/amd/model.py:994-1013" },
    fused: (p) => addCounts({ tokens: p.tokens, hidden: p.hidden, bytesPerElement: p.b }),
    decompose: (p) => [{ atom: "add", args: { elements: p.tokens * p.hidden, bytesPerElement: p.b } }],
    residentIntermediates: () => [],
    compulsoryBytes: (p) => 3 * p.tokens * p.hidden * p.b,
    notes: ["HCHeadOp 的 hc_head_fn/base/scale 是模型级参数，不在本模块"],
  },
];

// ---------------------- 注意力形态与稀疏选择分支 ----------------------
// 共同骨架（四种 indexer 一致，依据仓库内 modeling 取证）：
//   ReLU 打分 -> 逐头加权求和 -> top-k。**无 value 通路、无 softmax、
//   index k 单头、scores 走 fp32**。差异只在 key 池化的位置与粒度、候选粒度。
const ATTENTION_MODULES = [
  {
    id: "sdpa_attention",
    title: "Scaled Dot-Product Attention (GQA/MHA/MQA)",
    source: { framework: "vLLM", symbol: "Attention", ref: "model_executor/layers/attention/attention.py" },
    fused: (p) => attentionCounts({
      heads: p.heads, queryTokens: p.queryTokens, keyTokens: p.keyTokens,
      headDim: p.headDim, valueDim: p.valueDim, bytesPerElement: p.b, kvHeads: p.kvHeads,
    }),
    decompose: (p) => {
      const density = scoreDensity(p.phase, p.queryTokens, p.keyTokens);
      const pairs = p.heads * scorePairs(p.phase, p.queryTokens, p.keyTokens);
      return [
        { atom: "matmul", args: {
          batch: p.heads, m: p.queryTokens, k: p.headDim, n: p.keyTokens, density, bytesPerElement: p.b,
          lhsElements: p.heads * p.queryTokens * p.headDim,
          rhsElements: p.kvHeads * p.keyTokens * p.headDim,
          outElements: pairs,
        } },
        { atom: "scale", args: { elements: pairs, bytesPerElement: p.b } },
        { atom: "softmax", args: { elements: pairs, bytesPerElement: p.b } },
        { atom: "matmul", args: {
          batch: p.heads, m: p.queryTokens, k: p.keyTokens, n: p.valueDim, density, bytesPerElement: p.b,
          lhsElements: pairs,
          rhsElements: p.kvHeads * p.keyTokens * p.valueDim,
          outElements: p.heads * p.queryTokens * p.valueDim,
        } },
        { atom: "scatter", args: { rows: p.kvHeads * p.queryTokens, width: p.headDim + p.valueDim, bytesPerElement: p.b, readIn: false } },
      ];
    },
    residentIntermediates: (p) => [
      { name: "scores（flash tiling 内驻留）", elements: p.heads * scorePairs(p.phase, p.queryTokens, p.keyTokens) },
      { name: "probs（online softmax 不落 HBM）", elements: p.heads * scorePairs(p.phase, p.queryTokens, p.keyTokens) },
    ],
    boundExpectation: { prefill: "compute", decode: "memory" },
    notes: ["F2 未含 1/sqrt(d) 的 scale 段；逐原子分解显式计入"],
    compulsoryBytes: (p) => (p.heads * p.queryTokens * p.headDim
      + p.kvHeads * p.keyTokens * (p.headDim + p.valueDim)
      + p.heads * p.queryTokens * p.valueDim
      + p.kvHeads * p.queryTokens * (p.headDim + p.valueDim)) * p.b,
  },
  {
    id: "dsa_indexer",
    title: "DeepSeek Sparse Attention Indexer",
    source: { framework: "vLLM", symbol: "SparseAttnIndexer", ref: "model_executor/layers/sparse_attn_indexer.py:729" },
    // W2：fused 由分解导出（fusedFromDecomposition），与运行时
    // sparseIndexerCounts 同一实现 —— 恒等式因此结构性闭合。
    variant: { poolStage: "none", perHeadWeights: true },
    fused: (p) => sparseIndexerCounts({ ...p, poolStage: "none", perHeadWeights: true }),
    decompose: (p) => indexerDecomposition({ ...p, poolStage: "none", perHeadWeights: true }),
    residentIntermediates: (p) => indexerResident({ ...p, poolStage: "none", perHeadWeights: true }),
    compulsoryBytes: (p) => indexerCompulsory({ ...p, poolStage: "none" }),
    boundExpectation: { prefill: "compute", decode: "memory" },
    notes: [
      "无 value 通路、无 softmax（ReLU + 逐头加权求和），index k 单头（wk 输出仅 index_head_dim）",
      "scores 走 fp32（modeling_glm5_next.py:826-833 的 .float()）",
    ],
  },
  {
    id: "dsa_kpool_indexer",
    title: "DeepSeek Sparse Attention Indexer (k-pool)",
    source: { framework: "vLLM", symbol: "SparseAttnIndexerKpool", ref: "model_executor/layers/sparse_attn_indexer_kpool.py:880" },
    // W2：fused 由分解导出（fusedFromDecomposition），与运行时
    // sparseIndexerCounts 同一实现 —— 恒等式因此结构性闭合。
    variant: { poolStage: "key", perHeadWeights: true },
    fused: (p) => sparseIndexerCounts({ ...p, poolStage: "key", perHeadWeights: true }),
    decompose: (p) => indexerDecomposition({ ...p, poolStage: "key", perHeadWeights: true }),
    residentIntermediates: (p) => indexerResident({ ...p, poolStage: "key", perHeadWeights: true }),
    compulsoryBytes: (p) => indexerCompulsory({ ...p, poolStage: "key" }),
    boundExpectation: { prefill: "compute", decode: "memory" },
    notes: ["key 先按 index_kpool 池化再打分，select_k = index_topk / index_kpool（modeling_glm5_next.py:850）"],
  },
  {
    id: "qsa_indexer",
    title: "Qwen Sparse Attention Indexer",
    source: { framework: "vLLM", symbol: "QSAIndexer", ref: "models/qwen4_exp/nvidia/indexer_qsa.py:90" },
    // W2：fused 由分解导出（fusedFromDecomposition），与运行时
    // sparseIndexerCounts 同一实现 —— 恒等式因此结构性闭合。
    variant: { poolStage: "key", perHeadWeights: false },
    fused: (p) => sparseIndexerCounts({ ...p, poolStage: "key", perHeadWeights: false }),
    decompose: (p) => indexerDecomposition({ ...p, poolStage: "key", perHeadWeights: false }),
    residentIntermediates: (p) => indexerResident({ ...p, poolStage: "key", perHeadWeights: false }),
    compulsoryBytes: (p) => indexerCompulsory({ ...p, poolStage: "key" }),
    boundExpectation: { prefill: "compute", decode: "memory" },
    notes: ["key 按 indexer_compress_ratio mean 池化（modeling_qwen4_exp.py:741），block_topk = budget / ratio"],
  },
  {
    id: "minimax_block_indexer",
    title: "MiniMax M3 Block Indexer",
    source: { framework: "vLLM", symbol: "MiniMaxM3Indexer", ref: "models/minimax_m3/common/indexer.py:546" },
    // W2：fused 由分解导出（fusedFromDecomposition），与运行时
    // sparseIndexerCounts 同一实现 —— 恒等式因此结构性闭合。
    variant: { poolStage: "score", perHeadWeights: false },
    fused: (p) => sparseIndexerCounts({ ...p, poolStage: "score", perHeadWeights: false }),
    decompose: (p) => indexerDecomposition({ ...p, poolStage: "score", perHeadWeights: false }),
    residentIntermediates: (p) => indexerResident({ ...p, poolStage: "score", perHeadWeights: false }),
    compulsoryBytes: (p) => indexerCompulsory({ ...p, poolStage: "score" }),
    boundExpectation: { prefill: "compute", decode: "memory" },
    notes: ["逐 token 打分后 amax 池化成 block（modeling_minimax_m3_vl.py:574-582），选块粒度 = ⌈S/block⌉"],
  },
  {
    id: "linear_attention_state",
    title: "Gated Delta / KDA Recurrent State",
    source: { framework: "vLLM", symbol: "lightning_attn / GDN", ref: "model_executor/layers/lightning_attn.py" },
    fused: (p) => linearAttentionStateCounts({
      tokens: p.tokens, heads: p.heads, keyDim: p.keyDim, valueDim: p.valueDim,
      bytesPerElement: p.b, delta: p.delta,
      // 与 decompose 的 decay_scan steps 同源：decode 每 token 一次，
      // prefill 走 chunked（ceil(T/C)，C 默认 64）。
      stateSteps: p.phase === "decode" ? p.tokens : Math.ceil(p.tokens / (p.chunkSize ?? 64)),
    }),
    decompose: (p) => {
      const state = p.heads * p.keyDim * p.valueDim;
      const steps = p.phase === "decode" ? p.tokens : Math.ceil(p.tokens / (p.chunkSize ?? 64));
      return [
        { atom: "matmul", args: { batch: p.heads, m: p.keyDim, k: p.tokens, n: p.valueDim, bytesPerElement: p.b, outElements: state } },
        ...(p.delta ? [{ atom: "matmul", args: { batch: p.heads, m: p.tokens, k: p.keyDim, n: p.valueDim, bytesPerElement: p.b, outElements: p.tokens * p.heads * p.valueDim } }] : []),
        { atom: "matmul", args: { batch: p.heads, m: p.tokens, k: p.keyDim, n: p.valueDim, bytesPerElement: p.b, outElements: p.tokens * p.heads * p.valueDim } },
        { atom: "decay_scan", args: { steps, state, heads: p.heads, bytesPerElement: p.b, expPerStep: p.delta ? 3 : 1 } },
      ];
    },
    residentIntermediates: () => [],
    boundExpectation: { prefill: "compute", decode: "memory" },
    notes: [
      "A6：per-token 递推下界。prefill 实际为 chunked，全状态读写约 T/C 次（C 默认 64）",
      "现实现两相位都按 per-token 计，prefill 状态流量高估约 C 倍",
    ],
    compulsoryBytes: (p) => 3 * (p.phase === "decode" ? p.tokens : Math.ceil(p.tokens / (p.chunkSize ?? 64))) * p.heads * p.keyDim * p.valueDim * p.b,
  },
];

function sumFused(...list) {
  return list.reduce((acc, cur) => ({
    matrix: acc.matrix + cur.matrix,
    vector: acc.vector + cur.vector,
    sfu: acc.sfu + cur.sfu,
    bytes: {
      weights: acc.bytes.weights + cur.bytes.weights,
      actIn: acc.bytes.actIn + cur.bytes.actIn,
      actOut: acc.bytes.actOut + cur.bytes.actOut,
    },
  }), { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } });
}

/**
 * 稀疏选择分支的统一分解（DSA / DSA-kpool / QSA / MSA 共用）。
 * poolStage："none" 逐 token 打分；"key" 先池化 key 再打分（候选 = ⌈S/pool⌉）；
 * "score" 逐 token 打分后按 block 池化分数（打分仍按 S，候选 = ⌈S/pool⌉）。
 */
function indexerDecomposition(p) {
  const pool = Math.max(p.pool ?? 1, 1);
  const stage = p.poolStage ?? "none";
  const scored = stage === "key" ? Math.ceil(p.keyTokens / pool) : p.keyTokens;
  const candidates = stage === "none" ? p.keyTokens : Math.ceil(p.keyTokens / pool);
  const density = scoreDensity(p.phase, p.queryTokens, scored);
  const pairs = p.heads * scorePairs(p.phase, p.queryTokens, scored);
  const scoreBytes = p.scoreBytes ?? 4; // fp32
  return [
    // key 池化（mean / 取块）：只有 key-pool 变体有
    ...(stage === "key" ? [{ atom: "reduce_sum", args: { elements: p.keyTokens * p.dim, groups: scored * p.dim, bytesPerElement: p.b } }] : []),
    // q·k 打分：index k 单头共享
    { atom: "matmul", args: {
      batch: p.heads, m: p.queryTokens, k: p.dim, n: scored, density, bytesPerElement: p.b,
      lhsElements: p.heads * p.queryTokens * p.dim,
      rhsElements: scored * p.dim,
      outElements: pairs,
    } },
    { atom: "scale", args: { elements: pairs, bytesPerElement: scoreBytes } },
    { atom: "relu", args: { elements: pairs, bytesPerElement: scoreBytes } },
    // 逐头加权求和（DSA/kpool 有 weights_proj 的权重；QSA 为等权求和）
    ...(p.perHeadWeights ? [{ atom: "mul", args: { elements: pairs, bytesPerElement: scoreBytes } }] : []),
    { atom: "reduce_sum", args: { elements: pairs, groups: p.queryTokens * scored, bytesPerElement: scoreBytes } },
    // score 池化（MSA 的 amax over block）
    ...(stage === "score" ? [{ atom: "reduce_max", args: { elements: p.queryTokens * scored, groups: p.queryTokens * candidates, bytesPerElement: scoreBytes } }] : []),
    { atom: "topk", args: { rows: p.queryTokens, candidates, k: Math.ceil(p.budget / (stage === "none" ? 1 : pool)), bytesPerElement: scoreBytes } },
    // index k cache 写（新 token 的 index key）
    { atom: "scatter", args: { rows: p.queryTokens, width: p.dim, bytesPerElement: p.b, readIn: false } },
  ];
}

function indexerResident(p) {
  const pool = Math.max(p.pool ?? 1, 1);
  const stage = p.poolStage ?? "none";
  const scored = stage === "key" ? Math.ceil(p.keyTokens / pool) : p.keyTokens;
  const pairs = p.heads * scorePairs(p.phase, p.queryTokens, scored);
  return [
    { name: "逐头打分 scores（fp32，不落 HBM）", elements: pairs, bytesPerElement: p.scoreBytes ?? 4 },
    { name: "ReLU 后的打分", elements: pairs, bytesPerElement: p.scoreBytes ?? 4 },
    ...(stage === "key" ? [{ name: "池化后的 index key", elements: scored * p.dim }] : []),
  ];
}

/**
 * 融合公式由分解导出：fused = Σ decompose − 声明的驻留中间量（读+写各省一次）。
 * 这样「fused ≡ Σ decompose」在计算三分量上结构性成立，bytes 差额恒等于
 * residentIntermediates —— 人工复核面收敛到「分解对不对」+「融合收益列全没列」，
 * 而不是去核两份独立公式。**不做 clamp**：过度声明会让 bytes 变负，由恒等式抓出。
 */
export function fusedFromDecomposition(entry, p) {
  const dec = evaluateDecomposition(entry.decompose(p));
  let savedIn = 0;
  let savedOut = 0;
  for (const item of entry.residentIntermediates?.(p) || []) {
    const w = item.elements * (item.bytesPerElement ?? p.b);
    savedIn += w;
    savedOut += w;
  }
  return {
    matrix: dec.matrix,
    vector: dec.vector,
    sfu: dec.sfu,
    bytes: {
      weights: dec.bytes.weights,
      actIn: dec.bytes.actIn - savedIn,
      actOut: dec.bytes.actOut - savedOut,
    },
  };
}

/**
 * 稀疏选择分支的统一计费（DSA / DSA-kpool / QSA / MSA 四变体共用一份实现，
 * **但 operator_id 不共用**——算法出处不同就是不同条目，见 formulas/index.js）。
 * 参数矩阵：
 * - DSA（deepseek_v32 / glm_moe_dsa）：pool=1、poolStage="none"、perHeadWeights=true
 * - DSA-kpool（glm5_next）：pool=index_kpool、poolStage="key"、perHeadWeights=true
 * - QSA（qwen4_exp）：pool=indexer_compress_ratio、poolStage="key"、perHeadWeights=false
 * - MSA（minimax_m3_vl）：pool=sparse_block_size、poolStage="score"、perHeadWeights=false
 * 共同点（三份 modeling 取证）：无 value 通路、无 softmax（ReLU + 逐头求和）、
 * index k 单头、scores 走 fp32。
 */
export function sparseIndexerCounts(params) {
  const p = { scoreBytes: 4, ...params };
  const counts = fusedFromDecomposition(
    { decompose: indexerDecomposition, residentIntermediates: indexerResident },
    p,
  );
  // W6：`indexRead` 单列 —— indexer 从**自己那份 index-k cache** 读的字节
  //（单头、宽 index_head_dim；key 侧池化的变体按池化后的位置数读）。
  // KV 读恒等式要把它与主注意力的 kvRead 分开比：主注意力按稀疏预算读，
  // indexer 必须扫全长才能选出 top-k，两个乘子不同。indexRead ⊆ actIn，不额外累加。
  const pool = Math.max(p.pool ?? 1, 1);
  const scored = (p.poolStage ?? "none") === "key" ? Math.ceil((p.keyTokens || 0) / pool) : (p.keyTokens || 0);
  return { ...counts, bytes: { ...counts.bytes, indexRead: scored * (p.dim || 0) * (p.b || 0) } };
}



/**
 * 稀疏 indexer 的 compulsory 下界：q 读 + 单头 index k 读 + 选中索引写
 * （int32）+ index k cache 写。中间打分量按融合不落 HBM。
 */
function indexerCompulsory(p) {
  const pool = Math.max(p.pool ?? 1, 1);
  const stage = p.poolStage ?? "none";
  const scored = stage === "key" ? Math.ceil(p.keyTokens / pool) : p.keyTokens;
  const candidates = stage === "none" ? p.keyTokens : Math.ceil(p.keyTokens / pool);
  return p.queryTokens * p.heads * p.dim * p.b
    + scored * p.dim * p.b
    + p.queryTokens * p.dim * p.b
    + p.queryTokens * Math.min(Math.ceil(p.budget / (stage === "none" ? 1 : pool)), candidates) * 4;
}

/** 模块注册表：id -> 条目。 */
export const MODULES = Object.fromEntries([...MODULE_LIST, ...ATTENTION_MODULES].map((entry) => [entry.id, entry]));

/** 已登记但尚未声明分解的模块（W1 清单；报表逐条打印，W2-W4 消化）。 */
export const DECOMPOSE_PENDING = {
  mhc_pre: "Sinkhorn 段的原子词汇待定（softmax-on-streams），W4",
  mhc_post: "同上",
  mhc_fused_post_pre: "同上",
};

