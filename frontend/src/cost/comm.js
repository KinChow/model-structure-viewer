// 给定并行计划的通信量估算；不建模 overlap、调度和实际链路拥塞。
// 来源：llm-analysis 的 TP 通信公式，以及 evolution_design.md §5.3(6.3) F11-F12。

import { kvBytesPerCard, stateBytesPerCard, validatePdPlan } from "./parallel.js";

function nonNegative(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : fallback;
}

/**
 * Ring all-reduce 的每层通信字节数。
 * 来源：llm-analysis 的 get_latency_fwd_per_layer_tp_comm；o_proj 与 MLP down 各触发一次。
 */
export function ringAllReduceBytes({ batch = 1, tokens = 1, hidden, bytesPerElement = 2, tp = 1, operations = 2 } = {}) {
  if (!Number.isInteger(tp) || tp < 1 || !Number.isFinite(hidden) || hidden < 0) return 0;
  return operations * 2 * ((tp - 1) / tp) * batch * tokens * hidden * bytesPerElement;
}

/**
 * Expert all-to-all 的近似通信字节数。
 * 注意使用每 token 激活的专家数，不使用专家总数。
 */
export function expertAllToAllBytes({ batch = 1, tokens = 1, hidden, expertsPerToken, bytesPerElement = 2, operations = 2 } = {}) {
  if (!Number.isFinite(hidden) || hidden < 0 || !Number.isFinite(expertsPerToken) || expertsPerToken < 0) return 0;
  return operations * batch * tokens * expertsPerToken * hidden * bytesPerElement;
}

/** PP 相邻 stage 间一次激活传输的字节数；总量按 stage 边界数线性扩展。 */
export function pipelineP2PBytes({ batch = 1, tokens = 1, hidden, bytesPerElement = 2, pp = 1 } = {}) {
  if (!Number.isInteger(pp) || pp < 1 || !Number.isFinite(hidden) || hidden < 0) return 0;
  return Math.max(0, pp - 1) * batch * tokens * hidden * bytesPerElement;
}

/** 从节点路径和计划推导该节点的通信字节数。 */
export function nodeCommunicationBytes(node, config = {}, plan = {}, options = {}) {
  const path = String(node?.id || node?.name || "").toLowerCase();
  const role = node?.attributes?.communication_role;
  const tp = plan.tp ?? plan.TP ?? 1;
  const ep = plan.ep ?? plan.EP ?? 1;
  const attnMode = plan.attnMode ?? plan.attn_mode ?? "tp";
  const bytesPerElement = options.bytesPerElement ?? 2;
  const batch = options.batch ?? 1;
  const tokens = options.tokens ?? options.sequence ?? 1;
  if (role === "tp_attention_output") {
    if (attnMode === "dp") return 0;
    return ringAllReduceBytes({ batch, tokens, hidden: config.hiddenSize, bytesPerElement, tp, operations: 1 });
  }
  if (role === "tp_mlp_output") {
    return ringAllReduceBytes({ batch, tokens, hidden: config.hiddenSize, bytesPerElement, tp, operations: 1 });
  }
  if (role === "ep_dispatch" || role === "ep_combine") {
    if (ep <= 1) return 0;
    return expertAllToAllBytes({ batch, tokens, hidden: config.hiddenSize, expertsPerToken: config.expertsPerToken, bytesPerElement, operations: 1 });
  }
  const routedExpert = /(?:^|\.)(?:experts|expert_mlp)(?:\.|$)/.test(path);
  if (/(o_proj|output projection|down_proj)/.test(path) && !routedExpert) {
    if (attnMode === "dp" && /(self_attn|attention|o_proj)/.test(path) && !/down_proj/.test(path)) return 0;
    return ringAllReduceBytes({ batch, tokens, hidden: config.hiddenSize, bytesPerElement, tp, operations: 1 });
  }
  // 来源：evolution_design.md F12；dispatch 与 combine 各归因一次，合计正好两次 all-to-all。
  if (/(?:^|\.)(?:dispatch|combine)(?:\.|$)/.test(path)) {
    if (ep <= 1) return 0;
    return expertAllToAllBytes({ batch, tokens, hidden: config.hiddenSize, expertsPerToken: config.expertsPerToken, bytesPerElement, operations: 1 });
  }
  return 0;
}

/**
 * PD 分离的 KV 传输量，按 decode 侧 KV 布局计算。
 * 来源：evolution_design.md §5.3(7) 与 F15；只给出理论传输量，不建模 overlap。
 */
export function pdKvTransferBytes({ totalKvBytes = 0, totalStateBytes = 0, config = {}, pdPlan = {}, prefillChip, decodeChip } = {}) {
  const checked = validatePdPlan(pdPlan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, perDecodeRankBytes: null, aggregateBytes: null };
  const perRank = kvBytesPerCard(totalKvBytes, config, checked.decodePlan);
  const perStateRank = stateBytesPerCard(totalStateBytes, config, checked.decodePlan);
  const decodeRanks = checked.decodePlan.tp * checked.decodePlan.dp;
  const layoutRepackRequired = checked.prefillPlan.tp !== checked.decodePlan.tp
    || checked.prefillPlan.attnMode !== checked.decodePlan.attnMode;
  const prefillLink = prefillChip?.interconnect?.inter_node?.bandwidth || prefillChip?.interconnect?.intra_node?.bandwidth;
  const decodeLink = decodeChip?.interconnect?.inter_node?.bandwidth || decodeChip?.interconnect?.intra_node?.bandwidth;
  const linkBandwidth = prefillLink && decodeLink ? Math.min(prefillLink, decodeLink) : prefillLink || decodeLink || null;
  const linkSource = prefillChip?.interconnect?.inter_node?.bandwidth && decodeChip?.interconnect?.inter_node?.bandwidth
    ? "两侧 inter_node"
    : "可用节点内/跨节点带宽的较小值";
  return {
    ok: true,
    errors: [],
    perDecodeRankBytes: perRank.bytes,
    decodeRanks,
    perDecodeRankStateBytes: perStateRank.bytes,
    aggregateStateBytes: perStateRank.bytes * decodeRanks,
    aggregateBytes: perRank.bytes * decodeRanks + perStateRank.bytes * decodeRanks,
    linkBandwidth,
    linkSource: linkBandwidth ? linkSource : "缺少链路带宽",
    layoutRepackRequired,
    shardFactor: perRank.shardFactor,
    stateShardFactor: perStateRank.shardFactor,
    prefillPlan: checked.prefillPlan,
    decodePlan: checked.decodePlan,
  };
}

/** 汇总给定计划的节点级通信和 PP 边界通信，供摘要或对比视图使用。 */
export function planCommunicationBytes({ root, config = {}, plan = {}, batch = 1, tokens = 1, bytesPerElement = 2 } = {}) {
  let nodeBytes = 0;
  function visit(node, multiplier = 1) {
    nodeBytes += nodeCommunicationBytes(node, config, plan, { batch, tokens, bytesPerElement }) * multiplier;
    const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
    const childHasExplicitRepeat = (node?.children || []).some((child) => Number.isFinite(child?.repeat));
    for (const child of node?.children || []) visit(child, multiplier * (childHasExplicitRepeat ? 1 : repeat));
  }
  if (root) visit(root);
  const ppBytes = pipelineP2PBytes({ batch, tokens, hidden: config.hiddenSize, bytesPerElement, pp: plan.pp ?? plan.PP ?? 1 });
  return { nodeBytes, ppBytes, totalBytes: nodeBytes + ppBytes };
}

export { nonNegative };
