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

// M11-P1-2：roofline.missing 的字段级文案。键与 roofline.js 的 missing.push
// 一一对应：数量侧（matrix/vector/sfu/bytes_moved/comm）与费率侧
// （peak_flops.<dtype>/vector_flops/sfu_ops/memory_bandwidth/interconnect.*）。
const MISSING_FIELD_LABELS_ZH = {
  matrix: "矩阵 MACs 数量",
  vector: "向量操作数量",
  sfu: "SFU 操作数量",
  bytes_moved: "访存字节数",
  comm: "通信字节数",
  vector_flops: "vector_flops 规格",
  sfu_ops: "sfu_ops 规格",
  memory_bandwidth: "memory_bandwidth 规格",
  "interconnect.intra_node.bandwidth": "节点内互联带宽",
  "interconnect.inter_node.bandwidth": "跨节点互联带宽",
};
const MISSING_FIELD_LABELS_EN = {
  matrix: "matrix MACs count",
  vector: "vector op count",
  sfu: "SFU op count",
  bytes_moved: "bytes moved",
  comm: "comm bytes",
  vector_flops: "vector_flops spec",
  sfu_ops: "sfu_ops spec",
  memory_bandwidth: "memory_bandwidth spec",
  "interconnect.intra_node.bandwidth": "intra-node link bandwidth",
  "interconnect.inter_node.bandwidth": "inter-node link bandwidth",
};

// M11-P1-4：macs_source 类目（compute.js macsSource 的值域）与节点级
// value_source（graphTruth 只写 "checkpoint"，其余为 config 推导）。
const MACS_SOURCE_ORDER = ["formula", "aggregate", "not-compute", "unknown"];
const MACS_SOURCE_LABELS_ZH = {
  formula: "公式推导",
  aggregate: "子树汇总",
  "not-compute": "非计算节点",
  unknown: "无公式",
};
const MACS_SOURCE_LABELS_EN = {
  formula: "formula",
  aggregate: "child sum",
  "not-compute": "non-compute",
  unknown: "no formula",
};
const VALUE_SOURCE_LABELS_ZH = { checkpoint: "真值", derived: "推导" };
const VALUE_SOURCE_LABELS_EN = { checkpoint: "ckpt", derived: "est" };

/**
 * roofline.missing 单键 → 双语字段名；未知键原样透传（不伪造可读名）。
 * @param {string[]} missing
 * @param {{english?: boolean}} options
 */
export function missingLabelsModel(missing = [], { english = false } = {}) {
  const labels = english ? MISSING_FIELD_LABELS_EN : MISSING_FIELD_LABELS_ZH;
  return (missing || []).map((key) => {
    if (labels[key]) return { key, label: labels[key] };
    // 动态费率键：peak_flops.<dtype>
    if (key.startsWith("peak_flops.")) {
      const dtype = key.slice("peak_flops.".length);
      return { key, label: english ? `peak_flops (${dtype}) spec` : `peak_flops（${dtype}）规格` };
    }
    return { key, label: key };
  });
}

/** macs_source 类目 → 双语短标签；未知值原样透传。 */
export function macsSourceLabel(source, { english = false } = {}) {
  if (!source) return null;
  const labels = english ? MACS_SOURCE_LABELS_EN : MACS_SOURCE_LABELS_ZH;
  return labels[source] || source;
}

/** 模型级 macsSources 计数 → 固定类目顺序的展示列表（零计数类目省略）。 */
export function macsSourcesModel(counts = {}, { english = false } = {}) {
  return MACS_SOURCE_ORDER
    .filter((source) => (counts?.[source] ?? 0) > 0)
    .map((source) => ({ source, label: macsSourceLabel(source, { english }), count: counts[source] }));
}

/**
 * 节点级 value_source 汇总：携带权重的节点里，checkpoint 真值 vs config 推导。
 * 无权重节点（容器/无参算子）不计入——它们没有"值来源"可披露。
 * @param {{nodes?: {node?: {value_source?: string|null, params?: number|null}}[]}|null|undefined} cost
 */
export function valueSourceCountsModel(cost, { english = false } = {}) {
  const nodes = cost?.nodes ?? [];
  if (!Array.isArray(nodes) || nodes.length === 0) return null;
  let checkpoint = 0;
  let derived = 0;
  for (const row of nodes) {
    const node = row?.node || row;
    if (node?.params == null) continue;
    if (node.value_source === "checkpoint") checkpoint += 1;
    else derived += 1;
  }
  if (checkpoint + derived === 0) return null;
  const labels = english ? VALUE_SOURCE_LABELS_EN : VALUE_SOURCE_LABELS_ZH;
  return {
    checkpoint,
    derived,
    text: `${labels.checkpoint} ${checkpoint} · ${labels.derived} ${derived}`,
    title: english
      ? "Weight-carrying nodes by value origin: checkpoint truth vs config-derived estimate"
      : "携带权重的节点按值来源分组：checkpoint 真值 vs config 推导",
  };
}

/**
 * M11-P1-3：η 披露。vector/SFU 两路费率在 rates.js 以 `eta.vector ?? 1` /
 * `eta.sfu ?? 1` 兜底（cost/chips/rates.js:31,34），而 resolveEfficiency
 * （cost/efficiency.js）不产出这两个键，芯片级 efficiency.vector/sfu 声明
 * 同样被丢弃——即 vector/SFU 路恒按 100% 效率计算，且 ηF/ηHBM/ηComm 滑块
 * 均不作用于它。1.0 是显式乐观上界假设而非测量值：本仓库效率默认值承袭
 * llm-analysis 的单一 flops_efficiency 折扣设计（ηF=0.7 即来源于此），
 * 该设计没有 vector/SFU 分项折扣的文献值，故计算侧暂无可引用的替代数。
 * 计算侧（rates/efficiency）归另一路修改，UI 只做披露，不伪造可调滑块。
 */
export const ETA_VECTOR_SFU_DEFAULT = 1;
export function etaDisclosureModel({ english = false } = {}) {
  return {
    value: ETA_VECTOR_SFU_DEFAULT,
    short: english
      ? "vector/SFU η=1.0 fixed (optimistic, not adjustable)"
      : "vector/SFU 固定 η=1.0（乐观上界，暂不可调）",
    detail: english
      ? "Vector/SFU rates are computed at 100% efficiency (rates.js falls back to eta.vector ?? 1 / eta.sfu ?? 1; resolveEfficiency does not emit these keys, so neither the sliders nor chip-level declarations reach them). The efficiency defaults follow llm-analysis's single flops_efficiency design (ηF=0.7), which has no literature-backed vector/SFU split, so 1.0 is an explicit optimistic upper bound, not a measurement. The ηF slider does not affect vector/SFU paths."
      : "vector/SFU 费率固定按 100% 效率计算（rates.js 兜底 eta.vector ?? 1 / eta.sfu ?? 1；resolveEfficiency 不产出这两个键，滑块与芯片级声明均无法触及）。效率默认值承袭 llm-analysis 的单一 flops_efficiency 折扣设计（ηF=0.7 即来源于此），无文献支撑的 vector/SFU 分项折扣，1.0 为显式乐观上界假设，非实测值。ηF 滑块不作用于 vector/SFU 路。",
  };
}

/**
 * M11-P1-5：checkpoint 真值获取过程的上界面模型。数据全部来自
 * structure.source（toStructureNode.js 写入），不做推断。
 * 关键场景：请求 huggingface 失败后静默切到 modelscope——
 * checkpoint_truth 仍为 "available"，但 config_endpoint ≠
 * checkpoint_truth_endpoint 且 checkpoint_truth_error 保留真实异常文本。
 * @param {{checkpoint_truth?: string, checkpoint_truth_error?: string|null,
 *           config_endpoint?: string|null, checkpoint_truth_endpoint?: string|null}|null|undefined} source
 */
export function checkpointTruthModel(source, { english = false } = {}) {
  const status = source?.checkpoint_truth || null;
  const error = source?.checkpoint_truth_error || null;
  const configEndpoint = source?.config_endpoint || null;
  const truthEndpoint = source?.checkpoint_truth_endpoint || null;
  const fallback = truthEndpoint != null && configEndpoint != null && truthEndpoint !== configEndpoint;
  const show = Boolean(error) || fallback || status === "unavailable" || status === "empty";
  if (!show) return { show: false, status, error, configEndpoint, truthEndpoint, fallback, tone: null, headline: null, meta: null };
  const tone = status === "available" ? "warn" : "error";
  const headline = fallback
    ? (english
      ? `Checkpoint truth came from ${truthEndpoint}; the requested ${configEndpoint} failed and the switch was silent.`
      : `checkpoint 真值来自 ${truthEndpoint}；请求的 ${configEndpoint} 失败后被静默切换。`)
    : status === "empty"
      ? (english ? "Checkpoint truth is empty (no tensors found)." : "checkpoint 真值为空（未找到张量）。")
      : (english ? "Checkpoint truth unavailable; weights shown are config-derived estimates." : "checkpoint 真值不可用；权重为 config 推导估算。");
  const meta = [configEndpoint, truthEndpoint].filter(Boolean).length > 0
    ? (english ? `config: ${configEndpoint ?? "unknown"} · truth: ${truthEndpoint ?? "not fetched"}` : `config：${configEndpoint ?? "未知"} · 真值：${truthEndpoint ?? "未获取"}`)
    : null;
  return { show: true, status, error, configEndpoint, truthEndpoint, fallback, tone, headline, meta };
}

/**
 * @param {{totalMacs?: number|null, unknownComputePaths?: string[], weightSource?: string, actions?: object|null,
 *          macsSources?: Record<string, number>, nodes?: object[]}} cost
 * @param {{bound?: string, times?: Record<string, number|null>, missing?: string[]}|null} roofline
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
  const missingLabels = missingLabelsModel(roofline?.missing, { english }).map((entry) => entry.label);
  return {
    bound: roofline?.bound || "unknown",
    boundLabel: (english ? BOUND_LABELS_EN : BOUND_LABELS_ZH)[roofline?.bound || "unknown"],
    times,
    // §3.3：matrix=0 是精确陈述；unknownComputePaths 才是"成本未覆盖"
    unknownComputeCount: cost?.unknownComputePaths?.length ?? 0,
    weightSource: cost?.weightSource || null,
    weightSourceLabel: (english ? WEIGHT_SOURCE_LABELS_EN : WEIGHT_SOURCE_LABELS_ZH)[cost?.weightSource] || cost?.weightSource || null,
    // M11-P1-2：bound=unknown 时"为什么 unknown"的字段级清单
    missing: roofline?.missing ?? [],
    missingCount: missingLabels.length,
    missingLabels,
    // M11-P1-4：MACs 来源类目计数 + 节点级值来源汇总
    macsSources: macsSourcesModel(cost?.macsSources, { english }),
    valueSourceCounts: valueSourceCountsModel(cost, { english }),
  };
}

/**
 * @param {{strategy?: string, graph_truth_gaps?: string[], graph_ambiguous_truth_matches?: object[],
 *          template_gaps?: string[], unsupported?: {code: string, message: string}[],
 *          warnings?: {code: string, message: string}[]}|null|undefined} diagnostics
 * @param {{english?: boolean}} options
 */
export function diagnosticsModel(diagnostics, { english = false } = {}) {
  const gaps = diagnostics?.graph_truth_gaps ?? diagnostics?.template_gaps ?? [];
  // M11-P0-6：生产出口（enrichGraphWithTruth）发 ambiguous_truth_matches，
  // 内部键 graph_ 前缀仅 bindTruthToGraph 内部使用——两个键都收，生产键优先。
  const ambiguous = diagnostics?.ambiguous_truth_matches
    ?? diagnostics?.graph_ambiguous_truth_matches
    ?? [];
  const strategy = diagnostics?.strategy || "no-truth";
  const adapted = strategy === "template+truth";
  // M11-P0-3：collectDiagnostics 产出的模板级信号此前零消费者——不支持的
  // 模型静默画出假图。§3.3：unsupported 必须显式告警。
  const unsupported = diagnostics?.unsupported ?? [];
  const warnings = diagnostics?.warnings ?? [];
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
    unsupported,
    unsupportedCount: unsupported.length,
    warnings,
    warningCount: warnings.length,
  };
}
