// roofline.js —— ERT 的 E×T 交点（W5-2）：动作向量 × chips/rates.js 费率表
// → 五路时间取 max（矩阵/向量/SFU/访存/通信，§3.7）。本模块不含模型公式，
// 不含芯片规格之外的任何数值假设。
import { resolveEfficiency } from "./efficiency.js";
import { chipRates } from "./chips/rates.js";

function positive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/** 兼容旧入参形状（cost.macs/weightBytes…）与 action 向量两种形态。 */
function normalizeActions(cost = {}) {
  if (cost.actions) return cost.actions;
  return {
    matrix: cost.macs ?? null,
    vector: cost.vector ?? 0,
    sfu: cost.sfu ?? 0,
    bytes: {
      weights: cost.weightBytes ?? 0,
      actIn: cost.actInBytes ?? 0,
      actOut: cost.actOutBytes ?? 0,
    },
    commBytes: cost.commBytes ?? 0,
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

  const bytesMoved = (actions.bytes?.weights || 0) + (actions.bytes?.actIn || 0) + (actions.bytes?.actOut || 0);
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

  const matrixTime = time(actions.matrix, rates.matrixPerSecond, { quantity: "matrix", rate: `peak_flops.${dtype}` });
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

  return {
    dtype,
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
