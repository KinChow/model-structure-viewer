// compute.js —— 成本主链（W5-1 切换）：查 FORMULAS counts 注册表（§3.1）。
// 旧 nodeMacs 分派链已删除；公式唯一来源是 structure/operators/formulas/（含 20 处
// legacy 镜像同步删除）。芯片参数不得进入本文件（§3.4，ERT 分离见 W5-2）。
import { countsForNode } from "../structure/operators/formulas/extractor.js";
import { nodeWeightBytes } from "./memory.js";
import { walkStructure } from "./traverse.js";
function countsFor(node, config, options = {}) {
  const vision = node?.attributes?.modality === "vision";
  // V3：用户可调视觉 token 数优先，config 推导值兜底；不传时与旧链路逐字节一致。
  const visionTokens = (options.visionTokens ?? config?.visionTokens) || 1;
  const executionOptions = vision
    ? { ...options, vision: true, visionTokens }
    : options;
  // extractor 公式（tokensFor/keyTokens）当前从 config 读 visionTokens，
  // 用户显式输入时需以覆盖后的 config 传入才能生效；未输入时 config 原样透传。
  const effectiveConfig = vision && options.visionTokens != null && config
    ? { ...config, visionTokens: options.visionTokens }
    : config;
  return countsForNode(node, {
    config: effectiveConfig,
    options: executionOptions,
    path: node?.id || "",
    bytesPerElement: 2,
  });
}

/** 节点矩阵 MACs；未实现算子返回 null（由 aggregate 的 unknown 链路接管）。 */
export function nodeMacs(node, config, options = {}) {
  const counts = countsFor(node, config, options);
  return counts ? counts.matrix : null;
}

function macsSource(node, counts, billingParent) {
  if (billingParent) return "formula";
  if (node?.children?.length) return "aggregate";
  if (!counts) return "unknown";
  return counts.matrix > 0 ? "formula" : "not-compute";
}

function isBillingParent(node, counts) {
  return Boolean(node?.children?.length && counts && node?.attributes?.operator_id);
}

function scaleActions(counts, multiplier) {
  const scale = (value) => (value == null ? null : value * multiplier);
  return {
    matrix: scale(counts.matrix),
    vector: scale(counts.vector),
    sfu: scale(counts.sfu),
    computeDtype: counts.computeDtype,
    bytes: {
      weights: scale(counts.bytes.weights),
      actIn: scale(counts.bytes.actIn),
      actOut: scale(counts.bytes.actOut),
      kvRead: scale(counts.bytes.kvRead),
      indexRead: scale(counts.bytes.indexRead),
    },
  };
}

// P7（步骤 7）：首参从 tree root 改为 Graph IR——walkStructure 已是图唯一遍历
// 路径，options.graph 透传随之删除（生产消费方 aggregate/lens 均已传图）。
export function computeNodeCosts(graph, config, options = {}) {
  const rows = [];
  const billedAncestors = new Set();
  walkStructure(graph, ({ node, path, multiplier }) => {
    const counts = countsFor(node, config, options);
    const billingParent = isBillingParent(node, counts);
    const coveredByAncestor = [...billedAncestors].some((prefix) => path.startsWith(`${prefix}.`));
    if (billingParent) billedAncestors.add(path);
    const useOwnCounts = billingParent || (!node?.children?.length && !coveredByAncestor);
    const computeMacs = coveredByAncestor ? 0 : useOwnCounts ? (counts ? counts.matrix : null) : 0;
    const compute = computeMacs == null ? null : computeMacs * multiplier;
    const actions = coveredByAncestor || !useOwnCounts || !counts ? null : scaleActions(counts, multiplier);
    rows.push({ path, node, multiplier, compute_macs: compute,
      macs_source: coveredByAncestor ? "aggregate" : macsSource(node, counts, billingParent),
      actions,
      weightBytes: nodeWeightBytes(node) * multiplier,
      estimate_status: compute == null ? "unknown" : "estimated" });
  });
  return rows;
}

function addNullable(left, right) {
  if (left == null || right == null) return null;
  return left + right;
}

/** 子树动作向量汇总：任一分量未知 → 整体未知（与 addNullable 同纪律）。 */
function addActions(left, right) {
  if (left == null) return right;
  if (right == null) return left;
  return {
    matrix: addNullable(left.matrix, right.matrix),
    vector: addNullable(left.vector, right.vector),
    sfu: addNullable(left.sfu, right.sfu),
    bytes: {
      weights: addNullable(left.bytes?.weights, right.bytes?.weights),
      actIn: addNullable(left.bytes?.actIn, right.bytes?.actIn),
      actOut: addNullable(left.bytes?.actOut, right.bytes?.actOut),
      kvRead: addNullable(left.bytes?.kvRead, right.bytes?.kvRead),
      indexRead: addNullable(left.bytes?.indexRead, right.bytes?.indexRead),
    },
  };
}

/**
 * 为节点 Lens 计算包含自身的子树汇总。汇总值与 compute_macs 分离，
 * 后者仍表示执行叶子的成本并用于模型总量，避免父卡展示子树成本时重复计费。
 */
export function aggregateNodeCosts(rows = []) {
  const aggregates = new Map(rows.map((row) => [row.path, {
    ...row,
    aggregate_macs: row.compute_macs,
    aggregate_weightBytes: row.weightBytes || 0,
    // M11-P0-4：子树 actions 汇总——父节点（actions=null）从子节点累加，
    // 使 lens 的父卡也能拿到 vector/sfu 计数（此前只有叶子有 actions）。
    aggregate_actions: row.actions ?? null,
  }]));
  const depth = (path) => path.split(".").length;
  for (const row of [...rows].sort((left, right) => depth(right.path) - depth(left.path))) {
    const parentPath = row.path.slice(0, row.path.lastIndexOf("."));
    if (!aggregates.has(parentPath)) continue;
    const parent = aggregates.get(parentPath);
    const current = aggregates.get(row.path);
    parent.aggregate_macs = addNullable(parent.aggregate_macs, current.aggregate_macs);
    parent.aggregate_weightBytes += current.aggregate_weightBytes;
    parent.aggregate_actions = addActions(parent.aggregate_actions, current.aggregate_actions);
  }
  return rows.map((row) => aggregates.get(row.path));
}
