// atoms.js —— 原子计算层（本体 L0）。
//
// 定位：msv 比 vLLM/SGLang 多的一层。上游最底层是算子（`Attention` 一次 kernel
// 调用，不可再分），msv 要继续把它分解为 `softmax(QK^T/sqrt(d))V`。因此本层的
// 对标出处是 **aten op 与 PyTorch `torch.utils.flop_counter.flop_registry`**，
// 不是 vLLM 的 layers（那一层对应 modules.js）。
//
// 单位约定（principles §3.1，与 counts.js 一致）：
// - matrix 存 **MACs**（aten flop 公式含 2×，抄时换算）
// - vector 存 flop（逐元素乘/加/比较各计 1）
// - sfu 存操作次数（A5：sigmoid=2、exp=1、rsqrt=1、div=1）
// - bytes 为一次前向的 compulsory traffic（读一遍 + 写一遍），不含 tiling 重读
//
// 纯函数；入参只含结构化 shape 参数，禁止 node / 显示名（principles §3.2）。
// 每个原子有一条手算 exact 单测（__tests__/atoms.test.js），T=2/d=2/heads=1。

/** 空动作向量。 */
export function zeroActions() {
  return { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
}

function actions({ matrix = 0, vector = 0, sfu = 0, weights = 0, actIn = 0, actOut = 0 }) {
  return { matrix, vector, sfu, bytes: { weights, actIn, actOut } };
}

/** 动作向量求和（分解 → 融合对账用）。 */
export function sumActions(...list) {
  return list.filter(Boolean).reduce((acc, cur) => ({
    matrix: acc.matrix + (cur.matrix || 0),
    vector: acc.vector + (cur.vector || 0),
    sfu: acc.sfu + (cur.sfu || 0),
    bytes: {
      weights: acc.bytes.weights + (cur.bytes?.weights || 0),
      actIn: acc.bytes.actIn + (cur.bytes?.actIn || 0),
      actOut: acc.bytes.actOut + (cur.bytes?.actOut || 0),
    },
  }), zeroActions());
}

// ---------------------------------------------------------------------------
// A1 matmul —— aten.mm / aten.bmm（flop_registry: mm_flop = 2·m·n·k FLOPs）。
// 逻辑形状 [batch, m, k] x [batch, k, n] -> [batch, m, n]。
//
// 计算与访存分离（Accelergy 的 action counts vs. 触达对象）：
// - matrix 由形状唯一决定，乘 density（掩码/稀疏下实际计算的输出比例：
//   因果 ≈ (S+1)/2S、块稀疏 = selected/S、稠密 = 1）
// - bytes 允许调用方用 lhsElements / rhsElements / outElements 声明真实操作数
//   足迹（GQA 的 K/V 共享、MLA 的 latent 读宽、cache 命中都属此类）；不声明时
//   按稠密形状推导。rhs="weight" 时右操作数计入 bytes.weights。
// ---------------------------------------------------------------------------
export function matmul({
  batch = 1, m, k, n, bytesPerElement, weightBytesPerElement, density = 1, rhs = "activation",
  lhsElements, rhsElements, outElements,
}) {
  // weightBytesPerElement：权重操作数的字节宽可以不同于激活（fp32 的 mHC
  // 混合矩阵等，paramDtypes 登记）；置 0 表达「复用别处已计过的同一份权重」
  //（weightsShared 语义的原子侧落点）。未传则跟随激活字节宽。
  const wb = weightBytesPerElement ?? bytesPerElement;
  const lhs = lhsElements ?? batch * m * k;
  const right = rhsElements ?? (rhs === "weight" ? k * n : batch * k * n);
  const out = outElements ?? batch * m * n * density;
  return actions({
    matrix: batch * m * k * n * density,
    weights: rhs === "weight" ? right * wb : 0,
    actIn: (lhs + (rhs === "weight" ? 0 : right)) * bytesPerElement,
    actOut: out * bytesPerElement,
  });
}


// ---------------------------------------------------------------------------
// A2 softmax —— aten._softmax。A2 假设：融合单遍（max / exp-sum / div 各一趟
// 逻辑，实现上 online softmax 单遍）。vector = 3E（减 max、累加、除），
// sfu = 2E（exp 一次 + 倒数一次）。
// ---------------------------------------------------------------------------
export function softmax({ elements, bytesPerElement }) {
  return actions({
    vector: 3 * elements,
    sfu: 2 * elements,
    actIn: elements * bytesPerElement,
    actOut: elements * bytesPerElement,
  });
}

// A3 scale —— 乘一个预计算标量（1/sqrt(d)）。每元素 1 乘，无 SFU（倒数在编译期）。
export function scale({ elements, bytesPerElement }) {
  return actions({ vector: elements, actIn: elements * bytesPerElement, actOut: elements * bytesPerElement });
}

// A4 add —— aten.add。operands 个操作数逐元素相加：(operands-1) 次加法。
// weightElements > 0 表示最后一个操作数是学习参数（norm 的 +1、bias 等），
// 该操作数计入 bytes.weights 而不是 actIn。
export function add({ elements, bytesPerElement, operands = 2, weightElements = 0 }) {
  return actions({
    vector: elements * (operands - 1),
    weights: weightElements * bytesPerElement,
    actIn: (weightElements > 0 ? (operands - 1) * elements : operands * elements) * bytesPerElement,
    actOut: elements * bytesPerElement,
  });
}

// A5 mul —— aten.mul。同 add 的计数结构，语义为乘。
export function mul({ elements, bytesPerElement, operands = 2, weightElements = 0 }) {
  return actions({
    vector: elements * (operands - 1),
    weights: weightElements * bytesPerElement,
    actIn: (weightElements > 0 ? (operands - 1) * elements : operands * elements) * bytesPerElement,
    actOut: elements * bytesPerElement,
  });
}


// A6 relu —— aten.relu。每元素一次比较选择，无 SFU。
export function relu({ elements, bytesPerElement }) {
  return actions({ vector: elements, actIn: elements * bytesPerElement, actOut: elements * bytesPerElement });
}

// A7 sigmoid —— aten.sigmoid。A5 口径：sigmoid = exp + 倒数 = 2 SFU，无 vector
// （逐元素乘由独立的 mul 原子承担）。
export function sigmoid({ elements, bytesPerElement }) {
  return actions({ sfu: 2 * elements, actIn: elements * bytesPerElement, actOut: elements * bytesPerElement });
}

// A8 silu —— aten.silu = x·sigmoid(x)：2 SFU + 1 乘。
export function silu({ elements, bytesPerElement }) {
  return actions({
    vector: elements,
    sfu: 2 * elements,
    actIn: elements * bytesPerElement,
    actOut: elements * bytesPerElement,
  });
}

// A9 reduce_max —— aten.amax。每组 n 个元素做 n-1 次比较。
export function reduceMax({ elements, groups, bytesPerElement }) {
  return actions({
    vector: Math.max(elements - groups, 0),
    actIn: elements * bytesPerElement,
    actOut: groups * bytesPerElement,
  });
}

// A10 reduce_sum —— aten.sum。每组 n 个元素做 n-1 次加法。
export function reduceSum({ elements, groups, bytesPerElement }) {
  return actions({
    vector: Math.max(elements - groups, 0),
    actIn: elements * bytesPerElement,
    actOut: groups * bytesPerElement,
  });
}

// A11 rsqrt —— aten.rsqrt。每元素 1 次 SFU。
export function rsqrt({ elements, bytesPerElement }) {
  return actions({ sfu: elements, actIn: elements * bytesPerElement, actOut: elements * bytesPerElement });
}

// A19 div —— aten.div（倒数）。A5 口径：div = 1 SFU。
// W1 发现：首版 18 原子缺此项，导致 topk 归一化（norm_topk_prob）的 sfu 无法表达。
export function div({ elements, bytesPerElement }) {
  return actions({ sfu: elements, actIn: 2 * elements * bytesPerElement, actOut: elements * bytesPerElement });
}


// A12 rope —— 旋转位置编码。A3 假设：sin/cos 查表，sfu = 0。
// 每维对（2 个元素）4 乘 2 加 => 每元素 3 flop。读 q/k 与 sin/cos，写回同宽。
export function rope({ elements, bytesPerElement }) {
  return actions({
    vector: 3 * elements,
    actIn: 2 * elements * bytesPerElement,
    actOut: elements * bytesPerElement,
  });
}

// A13 gather —— aten.index_select / embedding 查表 / paged KV cache 读。
// 无计算，按实际拷贝元素计访存（G1 口径：一阶访存按真实拷贝行数）。
// writeOut=false 表示读进寄存器/SRAM 直接消费（KV cache 读属此类），不计 actOut。
export function gather({ rows, width, bytesPerElement, writeOut = true }) {
  const elements = rows * width;
  return actions({ actIn: elements * bytesPerElement, actOut: writeOut ? elements * bytesPerElement : 0 });
}

// A14 scatter —— aten.scatter / index_put / paged KV cache 写。
// readIn=false 表示源在寄存器（刚算出的 K/V 写回 cache 属此类），不计 actIn。
export function scatter({ rows, width, bytesPerElement, readIn = true }) {
  const elements = rows * width;
  return actions({ actIn: readIn ? elements * bytesPerElement : 0, actOut: elements * bytesPerElement });
}


// A15 permute_copy —— 真实重排拷贝（非 strided view）。A1 假设：view 类零流量，
// 只有物化拷贝走本原子。
export function permuteCopy({ elements, bytesPerElement }) {
  return actions({ actIn: elements * bytesPerElement, actOut: elements * bytesPerElement });
}

// A16 conv1d —— aten.convolution（depthwise 因果短卷积）。
// MACs = tokens·channels·kernel；权重 channels·kernel 每次前向读一遍。
// decode 相位额外的 conv state（channels·(kernel-1)）读写由调用方以 stateBytes 传入。
export function conv1d({ tokens, channels, kernel, bytesPerElement, stateElements = 0 }) {
  return actions({
    matrix: tokens * channels * kernel,
    weights: channels * kernel * bytesPerElement,
    actIn: (tokens * channels + stateElements) * bytesPerElement,
    actOut: (tokens * channels + stateElements) * bytesPerElement,
  });
}

// A17 topk —— aten.topk。candidates 个候选里选 k 个：一趟扫描 candidates 次比较。
// aten.topk 返回 (values, indices)。values 按激活字节宽写出，供下游 reduce_sum /
// div 读；indices 按 int32（indexBytes，默认 4）写出。matrix 恒 0。
export function topk({ rows, candidates, k, bytesPerElement, indexBytes = 4 }) {
  const selected = rows * Math.min(k, candidates);
  return actions({
    vector: rows * candidates,
    actIn: rows * candidates * bytesPerElement,
    actOut: selected * bytesPerElement + selected * indexBytes,
  });
}

// A18 decay_scan —— 递推状态的衰减与更新扫描（线性注意力 / SSM 的 decay 段）。
// steps 步、每步 state 个元素做 1 次乘；每步每头 1 次 exp（decay）。
// 状态读写按 steps 次全状态计——prefill 的 chunked 实现由调用方把 steps 折成
// tokens/chunkSize（显式近似，记在 modules 的已知近似里）。
export function decayScan({ steps, state, heads = 1, bytesPerElement, expPerStep = 1 }) {
  return actions({
    vector: steps * state,
    sfu: steps * heads * expPerStep,
    actIn: 2 * steps * state * bytesPerElement,
    actOut: steps * state * bytesPerElement,
  });
}

/** 原子注册表：id -> 实现。护栏用它枚举，modules 的 decompose 只能引用这些 id。 */
export const ATOMS = {
  matmul,
  softmax,
  scale,
  add,
  mul,
  relu,
  sigmoid,
  silu,
  reduce_max: reduceMax,
  reduce_sum: reduceSum,
  rsqrt,
  div,
  rope,
  gather,
  scatter,
  permute_copy: permuteCopy,
  conv1d,
  topk,
  decay_scan: decayScan,
};

/** 按 [{ atom, args }] 序列求和（modules.decompose 的求值器）。 */
export function evaluateDecomposition(steps) {
  return sumActions(...steps.map(({ atom, args }) => {
    const impl = ATOMS[atom];
    if (!impl) throw new Error(`unknown atom: ${atom}`);
    return impl(args);
  }));
}
