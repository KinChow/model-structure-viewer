// counts.js —— 算子动作向量共享实现（docs/details/cost_counts.md F1-F9）。
//
// 单位约定（principles §3.1）：matrix 存 MACs（aten flop 公式含 2×，抄时换算）、
// vector 存 flop、sfu 存操作次数、bytes 为每次前向 compulsory traffic
// （权重读一遍 + 输入 + 输出；无 phase 分支，decode 现象由 T=1 自然涌现）。
//
// 纯函数；入参只含结构化 shape 参数，禁止 node / 显示名（principles §3.2）。
// 全局假设 A1-A4 见 cost_counts.md，逐条在注释里引用。

const product = (values) => values.reduce((total, value) => total * value, 1);

/**
 * F1 线性。aten: aten.mm（torch mm_flop = m·n·2k FLOPs → 此处 MACs）。
 * logicalShape = [out, in]（weight 逻辑形状；packed 存储形状由 W5 提取层换算）。
 */
export function linearCounts({ logicalShape, tokens, bytesPerElement, bias = false }) {
  const [out, inDim] = logicalShape;
  return {
    matrix: tokens * out * inDim,
    vector: bias ? tokens * out : 0,
    sfu: 0,
    bytes: {
      weights: out * inDim * bytesPerElement,
      actIn: tokens * inDim * bytesPerElement,
      actOut: tokens * out * bytesPerElement,
    },
  };
}

/**
 * F2 选择集注意力。aten: aten.bmm ×2 + aten._softmax。
 * 覆盖全注意力与 qsa/minimax/dsv4 变体——差异只在 keyTokens(S) 的取法。
 * A2：softmax 融合单遍，scores 写+读各一次。
 * bytes 中间张量：Q/K/V 读一遍，scores 与 probs 各写+读一遍，输出写。
 */
export function attentionCounts({ heads, queryTokens, keyTokens, headDim, valueDim, bytesPerElement }) {
  const q = heads * queryTokens * headDim;
  const k = heads * keyTokens * headDim;
  const v = heads * keyTokens * valueDim;
  const scores = heads * queryTokens * keyTokens;
  const context = heads * queryTokens * valueDim;
  return {
    matrix: scores * (headDim + valueDim),
    vector: 3 * scores,
    sfu: 2 * scores,
    bytes: {
      weights: 0,
      actIn: (q + k + v + 2 * scores) * bytesPerElement,
      actOut: (2 * scores + context) * bytesPerElement,
    },
  };
}

/**
 * F3 归一化。分解声明：mul / reduce / rsqrt / mul（无单一 aten 对应）。
 * weightOne = true 为 Gemma 风格（×(1+w)，多 TH 次加法）；
 * gated = true 为 gated RMSNorm（多一路 sigmoid 门乘）。
 */
export function rmsnormCounts({ tokens, hidden, bytesPerElement, weightOne = false, gated = false }) {
  const gate = gated ? hidden * tokens : 0;
  return {
    matrix: 0,
    vector: 3 * hidden * tokens + (weightOne ? hidden * tokens : 0) + gate,
    sfu: tokens + (gated ? hidden * tokens : 0),
    bytes: {
      weights: hidden * bytesPerElement,
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
    sfu: tokens * width,
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
    sfu: tokens * intermediate,
    bytes: { weights: 0, actIn: 2 * tokens * intermediate * bytesPerElement, actOut: tokens * intermediate * bytesPerElement },
  };
}

/** F6 旋转位置编码。A3：sin/cos 查表，sfu ≈ 0。每维对 4 乘 2 加 = 3 flop/元素。 */
export function ropeCounts({ tokens, ropeDims, bytesPerElement }) {
  return {
    matrix: 0,
    vector: 3 * tokens * ropeDims,
    sfu: 0,
    bytes: { weights: 0, actIn: 2 * tokens * ropeDims * bytesPerElement, actOut: 2 * tokens * ropeDims * bytesPerElement },
  };
}

/** F7a 因果短卷积（Conv1D，kernel = w）+ SiLU。 */
export function causalConvCounts({ tokens, channels, kernel, bytesPerElement }) {
  return {
    matrix: tokens * channels * kernel,
    vector: tokens * channels,
    sfu: tokens * channels,
    bytes: { weights: channels * kernel * bytesPerElement, actIn: tokens * channels * bytesPerElement, actOut: tokens * channels * bytesPerElement },
  };
}

/**
 * F7b 线性注意力递推状态。
 * plain（gated linear attention）：S_t = decay⊙S + k^Tv（外积），o_t = q_t S_t
 *   → matrix = 2T·dk·dv（外积 + query）。
 * delta（gated delta attention，数学修正 2026-09-07：S_{t-1}k_t 是 matvec，
 *   属矩阵 MACs 而非 vector——修正 cost_counts.md 的规格）
 *   → matrix = 3T·dk·dv（外积 + delta matvec + query）。
 * **bytes 由递推状态主导：每 token 状态读+写 2·dk·dv·b。**
 */
export function linearAttentionStateCounts({ tokens, keyDim, valueDim, bytesPerElement, delta = false }) {
  const state = keyDim * valueDim;
  return {
    matrix: (delta ? 3 : 2) * tokens * state,
    vector: tokens * state * (delta ? 2 : 1),
    sfu: tokens + (delta ? tokens : 0),
    bytes: {
      weights: 0,
      actIn: 2 * tokens * state * bytesPerElement,
      actOut: tokens * state * bytesPerElement,
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
    vector: tokens * experts,
    sfu: normTopkProb ? tokens * topk : 0,
    bytes: { weights: 0, actIn: tokens * experts * bytesPerElement, actOut: tokens * topk * bytesPerElement },
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
