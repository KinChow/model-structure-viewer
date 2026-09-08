// ui.js —— cost 数据 → 展示语义的视图模型（W6-1）。
// 只做数据形状映射与文案，不做计算；渲染层（components/）保持哑组件。
// 五类瓶颈单元命名与 roofline.js 的 times 键一一对应（§3.7）。

const UNIT_LABELS_ZH = { matrix: "矩阵", vector: "向量", sfu: "SFU", memory: "访存", comm: "通信" };
const UNIT_LABELS_EN = { matrix: "matrix", vector: "vector", sfu: "SFU", memory: "memory", comm: "comm" };

const BOUND_LABELS_ZH = { ...UNIT_LABELS_ZH, unknown: "未知" };
const BOUND_LABELS_EN = { ...UNIT_LABELS_EN, unknown: "unknown" };

const WEIGHT_SOURCE_LABELS_ZH = {
  checkpoint: "checkpoint 真值",
  node: "节点权重汇总",
  derived: "config 推导",
  "derived-quantized": "config 推导（量化假设）",
  "what-if": "what-if 假设",
};
const WEIGHT_SOURCE_LABELS_EN = {
  checkpoint: "checkpoint truth",
  node: "node weight sum",
  derived: "config derived",
  "derived-quantized": "config derived (quantized)",
  "what-if": "what-if assumption",
};

/**
 * @param {{totalMacs?: number|null, unknownComputePaths?: string[], weightSource?: string, actions?: object|null}} cost
 * @param {{bound?: string, times?: Record<string, number|null>}|null} roofline
 * @param {{english?: boolean}} options
 */
export function costSummaryModel(cost = {}, roofline = null, { english = false } = {}) {
  const unitLabels = english ? UNIT_LABELS_EN : UNIT_LABELS_ZH;
  const times = Object.entries(roofline?.times || {}).map(([unit, seconds]) => ({
    unit,
    label: unitLabels[unit] || unit,
    seconds: seconds ?? null,
    known: seconds != null,
  }));
  return {
    bound: roofline?.bound || "unknown",
    boundLabel: (english ? BOUND_LABELS_EN : BOUND_LABELS_ZH)[roofline?.bound || "unknown"],
    times,
    // §3.3：matrix=0 是精确陈述；unknownComputePaths 才是"成本未覆盖"
    unknownComputeCount: cost?.unknownComputePaths?.length ?? 0,
    weightSource: cost?.weightSource || null,
    weightSourceLabel: (english ? WEIGHT_SOURCE_LABELS_EN : WEIGHT_SOURCE_LABELS_ZH)[cost?.weightSource] || cost?.weightSource || null,
  };
}

/**
 * @param {{strategy?: string, graph_truth_gaps?: string[], graph_ambiguous_truth_matches?: object[],
 *          template_gaps?: string[]}|null|undefined} diagnostics
 * @param {{english?: boolean}} options
 */
export function diagnosticsModel(diagnostics, { english = false } = {}) {
  const gaps = diagnostics?.graph_truth_gaps ?? diagnostics?.template_gaps ?? [];
  const ambiguous = diagnostics?.graph_ambiguous_truth_matches ?? [];
  const strategy = diagnostics?.strategy || "no-truth";
  const adapted = strategy === "template+truth";
  return {
    strategy,
    // §4.5：skeleton-truth = 未适配结构，图来自 checkpoint 骨架，无语义绑定
    banner: adapted ? null : {
      skeleton: strategy === "skeleton-truth",
      text: strategy === "skeleton-truth"
        ? (english ? "Structure not adapted: diagram comes from the checkpoint skeleton and carries no semantic binding." : "未适配结构：图来自 checkpoint 骨架，无语义绑定。")
        : strategy === "no-truth"
          ? (english ? "No checkpoint truth loaded; weights shown are config-derived estimates." : "未加载 checkpoint 真值；权重为 config 推导估算。")
          : null,
    },
    gaps,
    gapCount: gaps.length,
    ambiguous,
    ambiguousCount: ambiguous.length,
  };
}
