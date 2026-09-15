// ui.js —— cost 数据 → 展示语义的视图模型（W6-1）。
// 只做数据形状映射与文案，不做计算；渲染层（components/）保持哑组件。
// 五类瓶颈单元命名与 roofline.js 的 times 键一一对应（§3.7）。
import { t } from "../i18n/format.js";

const MACS_SOURCE_ORDER = ["formula", "aggregate", "not-compute", "unknown"];
const UNIT_KEYS = ["matrix", "vector", "sfu", "memory", "comm"];

function lang({ english } = {}) {
  return english ? "en" : "zh";
}

/**
 * roofline.missing 单键 → 双语字段名；未知键原样透传（不伪造可读名）。
 * @param {string[]} missing
 * @param {{english?: boolean}} options
 */
export function missingLabelsModel(missing = [], { english = false } = {}) {
  const language = lang({ english });
  return (missing || []).map((key) => {
    const catalogKey = `cost.missingField.${key}`;
    const label = t(language, catalogKey);
    if (label !== catalogKey) return { key, label };
    if (key.startsWith("peak_flops.")) {
      return { key, label: t(language, "cost.missingField.peakFlops", { dtype: key.slice("peak_flops.".length) }) };
    }
    return { key, label: key };
  });
}

/** macs_source 类目 → 双语短标签；未知值原样透传。 */
export function macsSourceLabel(source, { english = false } = {}) {
  if (!source) return null;
  const catalogKey = `cost.macsSource.${source}`;
  const label = t(lang({ english }), catalogKey);
  return label === catalogKey ? source : label;
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
  const language = lang({ english });
  return {
    checkpoint,
    derived,
    text: `${t(language, "cost.valueSource.checkpoint")} ${checkpoint} · ${t(language, "cost.valueSource.derived")} ${derived}`,
    title: t(language, "cost.valueSource.title"),
  };
}

/**
 * M11-P1-3：η 披露。vector/SFU 两路费率在 rates.js 以 `eta.vector ?? 1` /
 * `eta.sfu ?? 1` 兜底（cost/chips/rates.js:31,34），而 resolveEfficiency
 * （cost/efficiency.js）不产出这两个键，芯片级 efficiency.vector/sfu 声明
 * 同样被丢弃——即 vector/SFU 路恒按 100% 效率计算，且 ηF/ηHBM/ηComm 滑块
 * 均不作用于它。1.0 是显式乐观上界假设而非测量值：ηF 默认 0.7 是 UI 可调
 * 旋钮（原则 §3.6），没有独立的 vector/SFU 分项折扣，故计算侧暂不另给替代数。
 * 计算侧（rates/efficiency）归另一路修改，UI 只做披露，不伪造可调滑块。
 */
export const ETA_VECTOR_SFU_DEFAULT = 1;
export function etaDisclosureModel({ english = false } = {}) {
  const language = lang({ english });
  return {
    value: ETA_VECTOR_SFU_DEFAULT,
    short: t(language, "cost.eta.short"),
    detail: t(language, "cost.eta.detail"),
  };
}

/**
 * M11-P1-5：checkpoint 真值获取过程的上界面模型。数据全部来自
 * structure.source（modelStructure.js 写入），不做推断。
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
  const language = lang({ english });
  const tone = status === "available" ? "warn" : "error";
  const headline = fallback
    ? t(language, "cost.checkpoint.fallback", { truthEndpoint, configEndpoint })
    : status === "empty"
      ? t(language, "cost.checkpoint.empty")
      : t(language, "cost.checkpoint.unavailable");
  const meta = [configEndpoint, truthEndpoint].filter(Boolean).length > 0
    ? t(language, "cost.checkpoint.meta", {
      configEndpoint: configEndpoint ?? t(language, "cost.checkpoint.configUnknown"),
      truthEndpoint: truthEndpoint ?? t(language, "cost.checkpoint.truthMissing"),
    })
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
  const language = lang({ english });
  const times = Object.entries(roofline?.times || {}).map(([unit, seconds]) => ({
    unit,
    label: UNIT_KEYS.includes(unit) ? t(language, `cost.unit.${unit}`) : unit,
    seconds: seconds ?? null,
    known: seconds != null,
  }));
  const missingLabels = missingLabelsModel(roofline?.missing, { english }).map((entry) => entry.label);
  const bound = roofline?.bound || "unknown";
  const boundKey = UNIT_KEYS.includes(bound) ? `cost.unit.${bound}` : "cost.bound.unknown";
  const weightSource = cost?.weightSource || null;
  const weightSourceKey = weightSource ? `cost.weightSource.${weightSource}` : null;
  const weightSourceLabel = weightSourceKey ? t(language, weightSourceKey) : null;
  return {
    bound,
    boundLabel: t(language, boundKey),
    times,
    // §3.3：matrix=0 是精确陈述；unknownComputePaths 才是"成本未覆盖"
    unknownComputeCount: cost?.unknownComputePaths?.length ?? 0,
    weightSource,
    weightSourceLabel: weightSourceLabel && weightSourceLabel !== weightSourceKey ? weightSourceLabel : weightSource,
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
  const language = lang({ english });
  const gaps = diagnostics?.graph_truth_gaps ?? diagnostics?.template_gaps ?? [];
  // M11-P0-6：生产出口（enrichGraphWithTruth）发 ambiguous_truth_matches，
  // 内部键 graph_ 前缀仅 bindTruthToGraph 内部使用——两个键都收，生产键优先。
  const ambiguous = diagnostics?.ambiguous_truth_matches
    ?? diagnostics?.graph_ambiguous_truth_matches
    ?? [];
  const strategy = diagnostics?.strategy || "no-truth";
  const adapted = strategy === "template+truth" || strategy === "template+header-truth";
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
        ? t(language, "cost.diag.skeletonBanner")
        : strategy === "no-truth"
          ? t(language, "cost.diag.noTruthBanner")
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
