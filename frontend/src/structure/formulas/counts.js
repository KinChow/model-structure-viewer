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
 * 覆盖全部打分式变体（GQA/MQA/MLA/QSA/块稀疏/SWA），差异全在参数：
 * - prefill：T=S=seq（因果，O(seq²)）；decode：T=1、S=上下文全长（O(S)）；
 * - kvHeads：K/V 头数。MHA=heads；GQA=实际 KV 头数（流量随其缩小）；
 *   MQA/MLA=1（单 KV 头或共享 latent）。matrix 不随 kvHeads 变——
 *   每个 query head 都要做完整点积，只有 K/V 流量随 KV 头数缩小。
 * - MLA：headDim = kv_lora_rank + rope_dim（latent 打分宽度），valueDim = kv_lora_rank。
 * A2：softmax 融合单遍，scores 写+读各一次。
 * bytes：Q 读（heads）、K/V 读（kvHeads；decode 时即读 KV cache）、
 * 新算 K/V 写回 cache（kvHeads·T·(D+dv)：prefill 全量、decode 1 token）、
 * scores/probs 写+读（heads）、输出写。
 */
export function attentionCounts({ heads, queryTokens, keyTokens, headDim, valueDim, bytesPerElement, kvHeads = heads }) {
  const q = heads * queryTokens * headDim;
  const k = kvHeads * keyTokens * headDim;
  const v = kvHeads * keyTokens * valueDim;
  const scores = heads * queryTokens * keyTokens;
  const context = heads * queryTokens * valueDim;
  const kvWrite = kvHeads * queryTokens * (headDim + valueDim);
  return {
    matrix: scores * (headDim + valueDim),
    vector: 3 * scores,
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
export function rmsnormCounts({ tokens, hidden, bytesPerElement, weightOne = false, gated = false }) {
  const gate = gated ? hidden * tokens : 0;
  return {
    matrix: 0,
    vector: 4 * hidden * tokens + (weightOne ? hidden * tokens : 0) + gate,
    sfu: tokens + (gated ? 2 * hidden * tokens : 0),
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
export function linearAttentionStateCounts({ tokens, heads = 1, keyDim, valueDim, bytesPerElement, delta = false }) {
  const state = heads * keyDim * valueDim;
  return {
    matrix: (delta ? 3 : 2) * tokens * state,
    vector: tokens * state * (delta ? 2 : 1),
    sfu: heads * tokens * (delta ? 3 : 1), // decay exp 每 head 1 次；delta 另加 beta sigmoid 2 次
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
