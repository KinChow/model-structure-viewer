// roofline.js —— ERT 的 E×T 交点（W5-2）：动作向量 × chips/rates.js 费率表
// → 五路时间取 max（矩阵/向量/SFU/访存/通信，§3.7）。本模块不含模型公式，
// 不含芯片规格之外的任何数值假设。
import { resolveEfficiency } from "./efficiency.js";
import { chipRates } from "./chips/rates.js";

function positive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/** 兼容旧入参形状（cost.macs/weightBytes…）与 action 向量两种形态。
 * M11-P0-4：旧形状的 vector/sfu 未知即 null（unknown），不再伪造 0——
 * §3.3：已知零是精确陈述，未知不得冒充零。 */
function normalizeActions(cost = {}) {
  if (cost.actions) {
    const a = cost.actions;
    const computeDtypes = a.computeDtypes || {};
    return {
      matrix: a.matrix ?? null,
      // Accelergy ERT × action counts：computeDtype 桶聚成 matrixTf32，
      // 与 vLLM mHC TF32 tensor-core 分段费率对齐。扁平 matrixTf32 仍可直传。
      matrixTf32: a.matrixTf32 ?? computeDtypes.tf32 ?? 0,
      vector: a.vector ?? null,
      sfu: a.sfu ?? null,
      bytes: {
        weights: a.bytes?.weights ?? a.weights ?? null,
        actIn: a.bytes?.actIn ?? a.actIn ?? null,
        actOut: a.bytes?.actOut ?? a.actOut ?? null,
        kvRead: a.bytes?.kvRead ?? a.kvRead ?? 0,
        indexRead: a.bytes?.indexRead ?? a.indexRead ?? 0,
      },
      commBytes: a.commBytes ?? null,
    };
  }
  return {
    matrix: cost.macs ?? null,
    vector: null,
    sfu: null,
    bytes: {
      weights: cost.weightBytes ?? null,
      actIn: cost.actInBytes ?? null,
      actOut: cost.actOutBytes ?? null,
    },
    commBytes: cost.commBytes ?? null,
  };
}

/**
 * 计算单个模块的五路理论时间并给出瓶颈分类。
 * 数量已知为零 → 时间 0（精确陈述，不参与 max 主导）；
 * 数量未知或费率缺失 → 时间 null，计入 missing，bound 退化为 unknown。
 * @param {object} cost 模块成本（含 actions 或旧字段）
 * @param {object} chip 芯片规格
 * @param {{dtype?: string, efficiency?: object, interNode?: boolean}} options
 */
export function classifyRoofline(cost = {}, chip = {}, options = {}) {
  const dtype = String(options.dtype || "bf16").toLowerCase();
  const eta = resolveEfficiency(chip, options.efficiency);
  const actions = normalizeActions(cost);
  const rates = chipRates(chip, { dtype, efficiency: options.efficiency });

  // M11-P0-4：任一字节分量未知 → bytesMoved 未知（null），不得把未知当 0 计入访存。
  const b = actions.bytes || {};
  const bytesMoved = b.weights == null || b.actIn == null || b.actOut == null
    ? null
    : b.weights + b.actIn + b.actOut + (b.kvRead || 0) + (b.indexRead || 0);
  const commBytes = actions.commBytes || cost.commBytes || 0;
  const link = options.interNode ? rates.interNodeBytesPerSecond : rates.intraNodeBytesPerSecond;

  const missing = [];
  const time = (quantity, rate, keys) => {
    if (quantity == null) {
      missing.push(keys.quantity);
      return null;
    }
    if (!positive(rate)) {
      if (positive(quantity)) missing.push(keys.rate);
      return null;
    }
    return quantity / rate;
  };

  // 矩阵时间 = 非 tf32 部分按全局 dtype 费率 + tf32 桶按 peak_flops.tf32
  //（Hopper/Blackwell + DeepGEMM 的 mHC pre-GEMM 在 TF32 tensor core 运行，
  // N2-1）。芯片无 tf32 行时**整段回退全局费率**——tf32 桶只是算力档位缺失，
  // 不是数量未知，不得制造伪 missing 阻断 bound 分类。
  let matrixTime;
  if (actions.matrixTf32 > 0 && positive(chip?.peak_flops?.tf32)) {
    const restTime = time(
      (actions.matrix ?? 0) - actions.matrixTf32,
      rates.matrixPerSecond,
      { quantity: "matrix", rate: `peak_flops.${dtype}` },
    );
    const tf32Time = (actions.matrixTf32 * eta.flops) / (chip.peak_flops.tf32 / 2);
    matrixTime = restTime == null ? null : restTime + tf32Time;
  } else {
    matrixTime = time(actions.matrix, rates.matrixPerSecond, { quantity: "matrix", rate: `peak_flops.${dtype}` });
  }
  const vectorTime = time(actions.vector, rates.vectorPerSecond, { quantity: "vector", rate: "vector_flops" });
  const sfuTime = time(actions.sfu, rates.sfuPerSecond, { quantity: "sfu", rate: "sfu_ops" });
  const memoryTime = time(bytesMoved, rates.bytesPerSecond, { quantity: "bytes_moved", rate: "memory_bandwidth" });
  const commTime = time(positive(commBytes) ? commBytes : 0, link, { quantity: "comm", rate: options.interNode ? "interconnect.inter_node.bandwidth" : "interconnect.intra_node.bandwidth" });

  const candidates = [
    ["matrix", matrixTime],
    ["vector", vectorTime],
    ["sfu", sfuTime],
    ["memory", memoryTime],
    ["comm", commTime],
  ];
  // bound 可分类的条件（§3.3/§4.4）：算力侧三单元全已知（null 阻断——
  // 数量未知不得伪造分类；已知零参与 max 但不主导）；访存/通信侧仅在
  // 有量时要求时间可得。任一条件不满足 → unknown。
  const computeSideComplete = actions.matrix != null && actions.vector != null && actions.sfu != null;
  const memorySideComplete = bytesMoved === 0 || memoryTime != null;
  const commSideComplete = !positive(commBytes) || commTime != null;
  const anyTimed = candidates.some(([, value]) => value != null);
  const bound = computeSideComplete && memorySideComplete && commSideComplete && anyTimed
    ? candidates.reduce((best, current) => current[1] != null && (best[1] == null || current[1] > best[1]) ? current : best)[0]
    : "unknown";

  // P10（协议 Q7③）：五路时间取 max 即 **overlap 静态上限**——comm 与 compute
  // 取最大而非求和，是闭式不等式（任何调度下的真实耗时 ≤ 各路时间之和，且
  // ≥ 最大单路；取 max = 保守下界口径的标注）。不建模 stream/调度，与
  // "明确不做 overlap 仿真"的分界在此。
  return {
    dtype,
    overlapUpperBound: true,
    bound,
    arithmeticIntensity: matrixTime != null && positive(bytesMoved) ? (2 * actions.matrix) / bytesMoved : null,
    ridgePoint: positive(rates.matrixPerSecond) && positive(rates.bytesPerSecond)
      ? (rates.matrixPerSecond * 2) / rates.bytesPerSecond : null,
    times: { matrix: matrixTime, vector: vectorTime, sfu: sfuTime, memory: memoryTime, comm: commTime },
    bytesMoved,
    missing,
    efficiency: eta,
    rates,
  };
}
