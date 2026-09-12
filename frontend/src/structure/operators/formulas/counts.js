// counts.js —— 算子动作向量共享实现（docs/details/cost_counts.md F1-F9）。
//
// 单位约定（principles §3.1）：matrix 存 MACs（aten flop 公式含 2×，抄时换算）、
// vector 存 flop、sfu 存操作次数、bytes 为每次前向 compulsory traffic
// （权重读一遍 + 输入 + 输出；无 phase 分支，decode 现象由 T=1 自然涌现）。
//
// 纯函数；入参只含结构化 shape 参数，禁止 node / 显示名（principles §3.2）。
// 全局假设 A1-A4 见 cost_counts.md，逐条在注释里引用。

// M11.5 子项 3：与原子逐位同构的 counts 直接委托 atoms.js 实现（单处化，
// 防抄写漂移）。方向向下（counts → atoms），无反向依赖（layering.test.js 护栏）。
import { add, softmax, gather } from "./atoms.js";

const product = (values) => values.reduce((total, value) => total * value, 1);

// ---------------------------------------------------------------------------
// 相位口径（W3）。原先 bytes/matrix 都「无 phase 分支，decode 由 T=1 自然涌现」，
// 该假设对逐元素类与投影类成立，对下列六类不成立，必须显式分相位：
// 因果可见长度（本节）· MoE 专家权重流量 · MLA absorbed/materialized ·
// 线性注意力 chunked 状态流量 · conv/递推 state cache · KV 读语义。
// ---------------------------------------------------------------------------

/**
 * 打分式注意力实际参与打分的 (query, key) 对数（per head）。
 * - prefill：因果掩码下第 t 个 query 只看 t 个 key → Σ_{t=1..T} t = T(T+1)/2；
 *   若有前缀 cache（S > T），前缀部分全可见 → 再加 T·(S−T)。
 * - decode：T=1，可见全长 S → 1·S。
 * 手算校验见 __tests__/counts.test.js「因果对数解析检查」（对小尺寸逐 token
 * 暴力求和比对，避免期望侧与实现侧抄同一假设的同义重复）。
 */
export function scoredPairs({ phase, queryTokens, keyTokens }) {
  if (phase === "decode") return queryTokens * keyTokens;
  const t = Math.min(queryTokens, keyTokens);
  return (t * (t + 1)) / 2 + queryTokens * Math.max(keyTokens - t, 0);
}

/** 因果/稀疏密度 = 实际打分对数 / 稠密对数。喂给 atoms.matmul 的 density。 */
export function causalDensity({ phase, queryTokens, keyTokens }) {
  const dense = queryTokens * keyTokens;
  return dense > 0 ? scoredPairs({ phase, queryTokens, keyTokens }) / dense : 0;
}


/**
 * F1 线性。aten: aten.mm（torch mm_flop = m·n·2k FLOPs → 此处 MACs）。
 * logicalShape = [out, in]（weight 逻辑形状；packed 存储形状由 W5 提取层换算）。
 */
export function linearCounts({ logicalShape, tokens, bytesPerElement, weightBytesPerElement, bias = false, expertFraction = 1, weightsShared = false }) {
  const [out, inDim] = logicalShape;
  return {
    matrix: tokens * out * inDim * expertFraction,
    vector: bias ? tokens * out : 0,
    sfu: 0,
    bytes: {
      // bias 也是要从 HBM 读的权重（out 个）。此前只记权重矩阵，与本模块自己的
      // 原子分解（下方 linearAtomSteps().decompose 的 add 原子带 weightElements: out）
      // 及 compulsoryBytes 口径不一致。
      // weightsShared：这次 GEMM 复用**别处已计过**的同一份权重（如 mHC 的
      // 最终 hc_post 复用最后一层的 hc_ffn_fn），算力照计、权重字节不重复计
      //（权重字节恒等式的口径是「该相位应读一遍」）。
      // weightBytesPerElement：权重的字节宽可以不同于激活（fp32 的 mHC 混合
      // 矩阵等，paramDtypes 登记）；未传则跟随激活字节宽。
      weights: weightsShared ? 0 : (out * inDim + (bias ? out : 0)) * (weightBytesPerElement ?? bytesPerElement),
      actIn: tokens * inDim * bytesPerElement,
      actOut: tokens * out * bytesPerElement,
    },
  };
}

/**
 * F2 选择集注意力。aten: aten.bmm ×2 + aten._softmax。
 * 覆盖全部打分式变体（GQA/MQA/MLA/QSA/块稀疏/SWA），差异全在参数：
 * - **相位（W3）**：打分对数走 scoredPairs —— prefill 因果三角 T(T+1)/2、
 *   decode 全长 S。此前两相位通吃 T·S，prefill 系统性高估约 2 倍。
 * - kvHeads：K/V 头数。MHA=heads；GQA=实际 KV 头数（流量随其缩小）；
 *   MQA/MLA=1（单 KV 头或共享 latent）。matrix 不随 kvHeads 变——
 *   每个 query head 都要做完整点积，只有 K/V 流量随 KV 头数缩小。
 * - MLA：headDim = kv_lora_rank + rope_dim（latent 打分宽度），valueDim = kv_lora_rank。
 * A2：softmax 融合单遍，scores 写+读各一次。
 * bytes：Q 读（heads）、K/V 读（kvHeads；decode 时即读 KV cache）、
 * 新算 K/V 写回 cache（kvHeads·T·(D+dv)：prefill 全量、decode 1 token）、
 * scores/probs 写+读（heads）、输出写。
 */
export function attentionCounts({ heads, queryTokens, keyTokens, headDim, valueDim, bytesPerElement, kvHeads = heads, phase = "prefill" }) {
  const q = heads * queryTokens * headDim;
  const k = kvHeads * keyTokens * headDim;
  const v = kvHeads * keyTokens * valueDim;
  const scores = heads * scoredPairs({ phase, queryTokens, keyTokens });
  const context = heads * queryTokens * valueDim;
  const kvWrite = kvHeads * queryTokens * (headDim + valueDim);
  return {
    matrix: scores * (headDim + valueDim),
    // W5：4·scores = 1（QK^T/sqrt(d) 的缩放乘）+ 3（softmax 的减 max/累加/除）。
    // 此前只计 3，漏了 scale 段——融合分解恒等式实测 fused/decompose = 0.75。
    vector: 4 * scores,
    sfu: 2 * scores,
    bytes: {
      weights: 0,
      actIn: (q + k + v + 2 * scores) * bytesPerElement,
      actOut: (2 * scores + context + kvWrite) * bytesPerElement,
    },
  };
}

/**
 * F3 归一化。分解声明：mul / reduce / rsqrt / mul（无单一 aten 对应）。
 * weightOne = true 为 Gemma 风格（×(1+w)，多 TH 次加法）；
 * gated = true 为 gated RMSNorm（多一路 sigmoid 门乘）。
 */
export function rmsnormCounts({ tokens, hidden, bytesPerElement, weightOne = false, gated = false, weightWidth, affineBias = false }) {
  const gate = gated ? hidden * tokens : 0;
  // 权重宽度默认 = 归一化宽度；**逐头**归一化（q_norm/k_norm/GDN 输出门）必须显式传
  // 最后一维，否则权重被放大 heads 倍（判据见 extractor 的 normWeightWidth 注释）。
  // affineBias = LayerNorm（有 bias，权重 2×宽度），RMSNorm 只有 scale。
  const weightElements = (weightWidth ?? hidden) * (affineBias ? 2 : 1);
  return {
    matrix: 0,
    // W5：均方求和是每组 hidden 个元素做 hidden-1 次加法，故整体 4·T·H - T
    // （x²、求和、缩放乘、乘 weight 四段，减去每 token 少的那一次加法）。
    vector: 4 * hidden * tokens - tokens + (weightOne ? hidden * tokens : 0) + gate,
    sfu: tokens + (gated ? 2 * hidden * tokens : 0),
    bytes: {
      weights: weightElements * bytesPerElement,
      actIn: (hidden * tokens + (gated ? hidden * tokens : 0)) * bytesPerElement,
      actOut: hidden * tokens * bytesPerElement,
    },
  };
}

/**
 * F4 门控乘（sigmoid(G) ⊙ O）。gateProjection = true 时多一个 [W, H] 投影权重。
 */
export function gateCounts({ tokens, width, bytesPerElement, gateProjection = false, gateProjectionInput = 0 }) {
  return {
    matrix: 0,
    vector: tokens * width,
    sfu: 2 * tokens * width, // sigmoid = exp + rcp
    bytes: {
      weights: gateProjection ? gateProjectionInput * width * bytesPerElement : 0,
      actIn: (tokens * width + (gateProjection ? tokens * gateProjectionInput : 0)) * bytesPerElement,
      actOut: tokens * width * bytesPerElement,
    },
  };
}

/** F5 逐元素激活（SwiGLU：SiLU(x)·y；vision_activation 同构，φ 由配置决定）。 */
export function swigluCounts({ tokens, intermediate, bytesPerElement }) {
  return {
    matrix: 0,
    vector: 2 * tokens * intermediate,
    sfu: 2 * tokens * intermediate, // silu = sigmoid(2 SFU) + mul
    bytes: { weights: 0, actIn: 2 * tokens * intermediate * bytesPerElement, actOut: tokens * intermediate * bytesPerElement },
  };
}

/**
 * F6 旋转位置编码。A3：sin/cos 查表，sfu ≈ 0。每维对 4 乘 2 加 = 3 flop/元素。
 * `ropeDims` = **每 token 被旋转的元素总数**（跨全部 query 头与 kv 头求和，
 * 含 partial_rotary_factor），不是单头的 head_dim。
 * W5 修正两处不一致：
 *   (a) actOut 原为 2·T·ropeDims（把 q 与 k 当两个张量），而 vector 只按一个
 *       张量计 —— 现统一为「一个总量」，读 = 数据 + sin/cos 两份，写 = 数据一份；
 *   (b) 调用方原来传单头 head_dim，等于只算了一个头（见 extractor 的 rope case）。
 */
export function ropeCounts({ tokens, ropeDims, bytesPerElement }) {
  return {
    matrix: 0,
    vector: 3 * tokens * ropeDims,
    sfu: 0,
    bytes: { weights: 0, actIn: 2 * tokens * ropeDims * bytesPerElement, actOut: tokens * ropeDims * bytesPerElement },
  };
}

/** F7a 因果短卷积旧规格（**仅 PLE 复合分解在用**）：含 SiLU（sigmoid=2 SFU + mul）。
 *  Qwen4Exp PLE 的 conv 支路按离线取证含 SiLU（modeling 未入库）；与 GDN 家族的
 *  causal_conv1d 叶（下函数）是不同算子的不同口径，非双源。 */
export function causalConvCounts({ tokens, channels, kernel, bytesPerElement }) {
  return {
    matrix: tokens * channels * kernel,
    vector: tokens * channels,
    sfu: 2 * tokens * channels, // silu = sigmoid(2 SFU) + mul
    bytes: { weights: channels * kernel * bytesPerElement, actIn: tokens * channels * bytesPerElement, actOut: tokens * channels * bytesPerElement },
  };
}

/**
 * F7a 因果短卷积（GDN/KDA 家族的 q/k/v 分支卷积）——运行时权威口径。
 *
 * P0 单源化：本函数此前是含 SiLU 的旧规格（vector=TC、sfu=2TC），与
 * extractor `causal_conv1d` case 的运行时实现构成双源（G2 双轨差）。
 * 统一为 extractor 口径（恒等式与 KV 对账锁定的版本）：
 * - vector/sfu = 0：conv 的 SiLU 不在本叶（归属后续激活路径）；
 * - weights = channels·kernel（卷积核权重读，P2 覆盖判据内）；
 * - decode 相位额外读写 conv state（kernel−1 个历史 token 的通道值，
 *   W3-⑥ 补齐；prefill 的窗口在片上滑动不落 HBM）。
 */
export function causalShortConvCounts({ tokens, channels, kernel, bytesPerElement, phase = "prefill" }) {
  const convStateElements = phase === "decode" ? channels * Math.max(kernel - 1, 0) : 0;
  return {
    matrix: tokens * channels * kernel,
    vector: 0,
    sfu: 0,
    bytes: {
      weights: channels * kernel * bytesPerElement,
      actIn: (tokens * channels + convStateElements) * bytesPerElement,
      actOut: (tokens * channels + convStateElements) * bytesPerElement,
    },
  };
}

/**
 * mHC 的 comb/Sinkhorn 段（DeepSeek V4，2026-09-09 kernel 取证后落词汇）。
 * hc_{attn,ffn}_scale[2] 分支在 pre kernel 内对 hc_mult×hc_mult 的 comb tile
 * 逐 token 计算：logits-softmax + (iterations-1) 轮行/列归一化（每轮两方向
 * 各一次除法遍 + 归一化求和）。出处：vLLM deepseek_v4 tilelang_kernels.py@92-142
 * （scale 分支选择）与 torch.py@62-98（comb/softmax/Sinkhorn fp32 全程）；
 * hc_sinkhorn_iters=20 时每 token ≈16 exp + 640 div + ~750 reduce/add，
 * 全部发生在 4×4 寄存器驻留 tile 上，相对 24×hc_dim 的 GEMM 可忽略但非零。
 * exp/div/reduce 分别由 softmax/div/add 原子承载。
 */
export function sinkhornCounts({ tokens, streams, iterations, bytesPerElement }) {
  const comb = tokens * streams * streams;
  const rounds = Math.max(iterations - 1, 0);
  const soft = softmaxCounts({ elements: comb, bytesPerElement });
  const divOnce = { matrix: 0, vector: 0, sfu: comb, bytes: { weights: 0, actIn: 2 * comb * bytesPerElement, actOut: comb * bytesPerElement } };
  const addOnce = { matrix: 0, vector: comb, sfu: 0, bytes: { weights: 0, actIn: 2 * comb * bytesPerElement, actOut: comb * bytesPerElement } };
  let vector = soft.vector + rounds * 2 * addOnce.vector;
  let sfu = soft.sfu + rounds * 2 * divOnce.sfu;
  let actIn = soft.bytes.actIn + rounds * 2 * (divOnce.bytes.actIn + addOnce.bytes.actIn);
  let actOut = soft.bytes.actOut + rounds * 2 * (divOnce.bytes.actOut + addOnce.bytes.actOut);
  return { matrix: 0, vector, sfu, bytes: { weights: 0, actIn, actOut } };
}

/**
 * F7b 线性注意力递推状态（覆盖全部 linearAttentionMode 变体：
 * generic=plain；qwen3_5/qwen4_exp/kimi/kimi_k3/glm5_next=delta）。
 * keyDim/valueDim 为**每头**维度；heads 显式给出（state = heads·dk·dv，
 * 多头下 state 流量是主导项，必须显式）。
 * plain：S_t = decay⊙S + k^Tv（外积），o_t = q_t S_t
 *   → matrix = 2T·heads·dk·dv（外积 + query）。
 * delta（gated delta rule，数学修正 2026-09-07：S_{t-1}k_t 是 matvec，
 *   属矩阵 MACs 而非 vector——修正 cost_counts.md 的规格）
 *   → matrix = 3T·heads·dk·dv（外积 + delta matvec + query）。
 * 执行形态假设：按 per-token 递推计；chunked 实现总量等价（仅流量分布不同）。
 * **bytes 由递推状态主导：每 token 状态读+写 2·heads·dk·dv·b。**
 */
export function linearAttentionStateCounts({ tokens, heads = 1, keyDim, valueDim, bytesPerElement, delta = false, stateSteps = undefined }) {
  const state = heads * keyDim * valueDim;
  // W5：递推段的 vector/sfu 与原子分解对齐 —— 外积 / delta matvec / query 三段
  // 都是 matmul（已在 matrix 里），vector 只剩 decay 的逐元素乘、sfu 只剩每步
  // 每头的 exp（delta 另加 beta 的 sigmoid）。次数按 stateSteps（chunked 实现
  // 每块与状态交互一次；不传则退回 per-token 下界，A6）。
  const steps = stateSteps ?? tokens;
  return {
    matrix: (delta ? 3 : 2) * tokens * state,
    vector: steps * state,
    sfu: steps * heads * (delta ? 3 : 1),
    bytes: {
      weights: 0,
      actIn: 2 * steps * state * bytesPerElement,
      actOut: steps * state * bytesPerElement,
    },
  };
}

/**
 * F8 MoE 路由与分发。aten: aten.topk。
 * combine 的加权合并（y = Σ w_e y_e）按诚实数学计 vector = 2·TkH
 * （乘 + 累加；规格文档同步修正）。
 */
export function topkCounts({ tokens, experts, topk, bytesPerElement, normTopkProb = true }) {
  return {
    matrix: 0,
    // W5：T·E 是 top-k 扫描；normTopkProb 时另有权重求和 T·(k-1) 次加法，
    // 除法计 sfu。此前漏了求和段（分解恒等式实测 fused/decompose = 0.9734）。
    vector: tokens * experts + (normTopkProb ? tokens * Math.max(topk - 1, 0) : 0),
    sfu: normTopkProb ? tokens * topk : 0,
    // 选中的专家 id 是 int32（4B），与激活的 bytesPerElement 无关
    bytes: { weights: 0, actIn: tokens * experts * bytesPerElement, actOut: tokens * topk * 4 },
  };
}

export function moeDispatchCounts({ tokens, hidden, topk, bytesPerElement }) {
  return {
    matrix: 0, vector: 0, sfu: 0,
    bytes: { weights: 0, actIn: tokens * hidden * bytesPerElement, actOut: tokens * topk * hidden * bytesPerElement },
  };
}

export function moeCombineCounts({ tokens, hidden, topk, bytesPerElement }) {
  return {
    matrix: 0,
    vector: 2 * tokens * topk * hidden,
    sfu: 0,
    bytes: { weights: 0, actIn: (tokens * topk * hidden + tokens * topk) * bytesPerElement, actOut: tokens * hidden * bytesPerElement },
  };
}

/**
 * MoE 路由专家的融合前馈（fused_moe_mlp 叶）：gate/up/down 三段 GEMM 与 SwiGLU
 * 激活在一个叶内计数 —— 对标 vLLM FusedMoE（model_executor/layers/fused_moe/
 * 打包 w13/w2）与 SGLang fused_moe 的专家内核。N2-4 W-A 前该语义挂在纯激活的
 * swiglu id 下（身份过载；QSA/DSA/MSA 同例不共用条目）。
 * matrix = T·k·3·EH·EI；权重读被触达的专家数 = min(k·T, E)：
 *   - prefill 大 T（k·T >= E）→ 全部 E 份权重都要读一遍
 *   - decode T=1 → 只读 k 份
 * 形式上与相位无关，相位差异由 tokens 自然涌现（这是「T=1 自然涌现」
 * 真正成立的情形）。每专家 3 段 GEMM，各 EH x EI。激活段走 F5 共享实现。
 */
export function fusedMoeMlpCounts({ tokens, topk, experts, expertHidden, expertIntermediate, bytesPerElement }) {
  const activation = swigluCounts({ tokens: tokens * topk, intermediate: expertIntermediate, bytesPerElement });
  const touchedExperts = experts > 0 ? Math.min(topk * tokens, experts) : topk;
  return {
    matrix: tokens * 3 * expertHidden * expertIntermediate * topk,
    vector: activation.vector,
    sfu: activation.sfu,
    bytes: {
      weights: 3 * touchedExperts * expertHidden * expertIntermediate * bytesPerElement,
      actIn: activation.bytes.actIn,
      actOut: activation.bytes.actOut,
    },
  };
}

/**
 * 逐元素加。与 add 原子逐位同构（scratch 证明 120/120 组全字段 Object.is 相等；
 * 护栏 __tests__/countsAtomsConsistency.test.js 固化为永久法则）→ 直接委托，
 * 公式单处化，防抄写漂移（M11.5 子项 3）。
 */
export function addCounts({ tokens, hidden, bytesPerElement }) {
  return add({ elements: tokens * hidden, bytesPerElement });
}

/** dsv4 hash 路由：纯查表。tableRows = 哈希表条目数（按参数计 weights）。 */
// Hash 路由（DeepSeek V4 tid2eid 静态查表）。2026-09-09 分类裁决：
// tid2eid 是 **buffer 不是参数**（NVIDIA Megatron-Bridge 文档明文 "Buffers are
// not parameters"；MaxText 同；出处 = Hash Layers, Roller et al. 2021）——
// 所以 bytes.weights = 0（与 embedding 表同待遇：不进权重字节恒等式），
// 流量按 gather 的真实拷贝计：每 token 读 topk 个专家 id、写 topk 个。
// 表本身的常驻容量（vocab·k·4B int32）由 derivedBufferBytes 单独计入显存。
// 流量与 gather 原子逐位同构（scratch 证明 100/100 组：读=写=tokens·topk）
// → 直接委托（M11.5 子项 3）。
export function hashRouteCounts({ tokens, topk, bytesPerElement }) {
  return gather({ rows: tokens, width: topk, bytesPerElement });
}

/** 独立 softmax 算子（融合注意力条目用 F2 内含版，这个给独立节点）。A2：单遍。
 *  与 softmax 原子逐位同构（scratch 证明 28/28 组）→ 直接委托（M11.5 子项 3）。 */
export function softmaxCounts({ elements, bytesPerElement }) {
  return softmax({ elements, bytesPerElement });
}

/**
 * F9 重排。A1：split/view 类零流量（fused projection 拆分是视图）。
 * vision_merge 的 permute 是真拷贝（copy=true）。
 */
export function rearrangeCounts({ copy = false, inElements, outElements, bytesPerElement } = {}) {
  if (!copy) return { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
  return {
    matrix: 0, vector: 0, sfu: 0,
    bytes: { weights: 0, actIn: inElements * bytesPerElement, actOut: outElements * bytesPerElement },
  };
}

// ---------------------------------------------------------------------------
// 共享 bytes 助手（M11.5 子项 3）：linear / rmsnorm 的原子分解、驻留中间量与
// compulsory 下界的**单处实现**。这六个片段原先住在 modules.js（linearDecompose
// 等私有函数），与本文件的 F1/F3 闭式公式互为镜像 —— 改一边另一边就悄悄漂移。
// 现在模块层（modules.js）统一消费下方导出；函数体为纯搬运，逐字节未改。
// ---------------------------------------------------------------------------

/** linear 的原子分解（p: {tokens, inDim, out, b, bias, expertFraction, weightBytesPerElement}）。 */
function linearDecompose(p) {
  return [
    { atom: "matmul", args: { batch: 1, m: p.tokens * (p.expertFraction ?? 1), k: p.inDim, n: p.out, bytesPerElement: p.b, weightBytesPerElement: p.weightBytesPerElement, rhs: "weight", outElements: p.tokens * p.out } },
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

/**
 * linear 三件套入口：{ decompose, resident, compulsory }。
 * modules.js 的 linear 与复合模块（mla_query_compress / mla_kv_compress /
 * hyper_connection / ple / mhc_pre / mhc_fused_post_pre）统一走这里，不再各自
 * 持有片段拷贝。调用方形状注意：mhc 的 base/scale 投影传 tokens: 0、b: 4，
 * mhc_fn 传 weightBytesPerElement: 4（分解据此把权重操作数按 fp32 计）。
 */
export function linearAtomSteps(p) {
  return { decompose: linearDecompose(p), resident: linearResident(p), compulsory: linearCompulsory(p) };
}

/**
 * rmsnorm 三件套入口：{ decompose, resident, compulsory }。
 * p: {tokens, hidden, b, weightOne, gated, weightWidth, affineBias} —— 其中
 * weightWidth / affineBias 是与 rmsnormCounts 对齐的签名占位：当前分解/驻留/
 * 下界不消费（与搬运前的 modules.js 片段行为一致），现有调用方也未传。
 */
export function rmsnormAtomSteps(p) {
  return { decompose: rmsnormDecompose(p), resident: rmsnormResident(p), compulsory: rmsnormCompulsory(p) };
}

export { product };
