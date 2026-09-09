// counts.js —— 算子动作向量共享实现（docs/details/cost_counts.md F1-F9）。
//
// 单位约定（principles §3.1）：matrix 存 MACs（aten flop 公式含 2×，抄时换算）、
// vector 存 flop、sfu 存操作次数、bytes 为每次前向 compulsory traffic
// （权重读一遍 + 输入 + 输出；无 phase 分支，decode 现象由 T=1 自然涌现）。
//
// 纯函数；入参只含结构化 shape 参数，禁止 node / 显示名（principles §3.2）。
// 全局假设 A1-A4 见 cost_counts.md，逐条在注释里引用。

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
export function linearCounts({ logicalShape, tokens, bytesPerElement, bias = false, expertFraction = 1, weightsShared = false }) {
  const [out, inDim] = logicalShape;
  return {
    matrix: tokens * out * inDim * expertFraction,
    vector: bias ? tokens * out : 0,
    sfu: 0,
    bytes: {
      // bias 也是要从 HBM 读的权重（out 个）。此前只记权重矩阵，与本模块自己的
      // 原子分解（modules.js linearDecompose 的 add 原子带 weightElements: out）
      // 及 compulsoryBytes 口径不一致。
      // weightsShared：这次 GEMM 复用**别处已计过**的同一份权重（如 mHC 的
      // 最终 hc_post 复用最后一层的 hc_ffn_fn），算力照计、权重字节不重复计
      //（权重字节恒等式的口径是「该相位应读一遍」）。
      weights: weightsShared ? 0 : (out * inDim + (bias ? out : 0)) * bytesPerElement,
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

/** F7a 因果短卷积（Conv1D，kernel = w）+ SiLU。 */
export function causalConvCounts({ tokens, channels, kernel, bytesPerElement }) {
  return {
    matrix: tokens * channels * kernel,
    vector: tokens * channels,
    sfu: 2 * tokens * channels, // silu = sigmoid(2 SFU) + mul
    bytes: { weights: channels * kernel * bytesPerElement, actIn: tokens * channels * bytesPerElement, actOut: tokens * channels * bytesPerElement },
  };
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

export function addCounts({ tokens, hidden, bytesPerElement }) {
  return {
    matrix: 0,
    vector: tokens * hidden,
    sfu: 0,
    bytes: { weights: 0, actIn: 2 * tokens * hidden * bytesPerElement, actOut: tokens * hidden * bytesPerElement },
  };
}

/** dsv4 hash 路由：纯查表。tableRows = 哈希表条目数（按参数计 weights）。 */
export function hashRouteCounts({ tokens, topk, tableRows, bytesPerElement }) {
  return {
    matrix: 0, vector: 0, sfu: 0,
    bytes: { weights: tableRows * bytesPerElement, actIn: tokens * bytesPerElement, actOut: tokens * topk * bytesPerElement },
  };
}

/** 独立 softmax 算子（融合注意力条目用 F2 内含版，这个给独立节点）。A2：单遍。 */
export function softmaxCounts({ elements, bytesPerElement }) {
  return {
    matrix: 0,
    vector: 3 * elements,
    sfu: 2 * elements,
    bytes: { weights: 0, actIn: elements * bytesPerElement, actOut: elements * bytesPerElement },
  };
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

export { product };
