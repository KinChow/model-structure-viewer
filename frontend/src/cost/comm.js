// 给定并行计划的通信量估算；不建模 overlap、调度和实际链路拥塞。
// 公式见 details/modules.md §9.5 与 details/parallel_protocol.md Q6。不再参考 llm-analysis。

import { kvBytesPerCard, stateBytesPerCard, validatePdPlan } from "./parallel.js";
import { walkStructure } from "./traverse.js";


/**
 * Ring all-reduce 的每层通信字节数。
 * o_proj 与 MLP down 各触发一次；字节公式见 details/modules.md §9.5。
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
  const path = String(node?.id || "").toLowerCase();
  const role = node?.attributes?.communication_role;
  const tp = plan.tp ?? plan.TP ?? 1;
  const ep = plan.ep ?? plan.EP ?? 1;
  const attnMode = plan.attnMode ?? plan.attn_mode ?? "tp";
  const bytesPerElement = options.bytesPerElement ?? 2;
  const batch = options.batch ?? 1;
  const tokens = options.tokens ?? options.sequence ?? 1;
  // C3c（2026-09-21 H20 实证修正）：SGLang 只有在**显式** --enforce-shared-experts-fusion 且 config 合规时
  // 才把 shared expert 折成额外 routed 专家进 all-to-all；**默认关**（DeepEP/EP>1@NV 均默认 fusion off，
  // 见 evidence/parallelism/deepep_shared_expert_h20.md）。故默认 shared expert 是独立本地 MLP、不进 all-to-all
  // （dispatch 保持 topk）；仅 enforceSharedExpertsFusion=true 时 +n_shared。vLLM/neutral 恒不折叠。
  const enforceFusion = options.enforceSharedExpertsFusion
    ?? plan.enforceSharedExpertsFusion
    ?? plan.enforce_shared_experts_fusion
    ?? false;
  const sharedFused = options.frameworkProfile === "sglang" && enforceFusion === true && (config.sharedExperts ?? 0) > 0;
  const dispatchExpertsPerToken = (config.expertsPerToken ?? 0) + (sharedFused ? config.sharedExperts : 0);
  if (role === "tp_attention_output") {
    if (attnMode === "dp") return 0;
    return ringAllReduceBytes({ batch, tokens, hidden: config.hiddenSize, bytesPerElement, tp, operations: 1 });
  }
  if (role === "tp_mlp_output") {
    return ringAllReduceBytes({ batch, tokens, hidden: config.hiddenSize, bytesPerElement, tp, operations: 1 });
  }
  if (role === "ep_dispatch" || role === "ep_combine") {
    if (ep <= 1) return 0;
    return expertAllToAllBytes({ batch, tokens, hidden: config.hiddenSize, expertsPerToken: dispatchExpertsPerToken, bytesPerElement, operations: 1 });
  }
  const routedExpert = /(?:^|\.)(?:experts|expert_mlp)(?:\.|$)/.test(path);
  if (/(o_proj|output projection|down_proj)/.test(path) && !routedExpert) {
    if (attnMode === "dp" && /(self_attn|attention|o_proj)/.test(path) && !/down_proj/.test(path)) return 0;
    return ringAllReduceBytes({ batch, tokens, hidden: config.hiddenSize, bytesPerElement, tp, operations: 1 });
  }
  // 来源：evolution_design.md F12；dispatch 与 combine 各归因一次，合计正好两次 all-to-all。
  if (/(?:^|\.)(?:dispatch|combine)(?:\.|$)/.test(path)) {
    // P10（协议 Q6 + Q4）：无 EP 时专家被 DP 切（DP-shards-experts），DP 副本间
    // 仍需 all-to-all；EP 启用时走上面 ep 分支。attnMode=tp 时 token 已按
    // DP 复制、专家域含 dp——同样触发。口径标注近似（Q3：无 moe_dp 轴，用 dp 近似）。
    const dp = plan.dp ?? plan.DP ?? 1;
    if (ep <= 1 && !(dp > 1)) return 0;
    return expertAllToAllBytes({ batch, tokens, hidden: config.hiddenSize, expertsPerToken: dispatchExpertsPerToken, bytesPerElement, operations: 1 });
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
  const linkSourceCode = prefillChip?.interconnect?.inter_node?.bandwidth && decodeChip?.interconnect?.inter_node?.bandwidth
    ? "comm.bothInterNode"
    : "comm.minAvailable";
  return {
    ok: true,
    errors: [],
    perDecodeRankBytes: perRank.bytes,
    decodeRanks,
    perDecodeRankStateBytes: perStateRank.bytes,
    aggregateStateBytes: perStateRank.bytes * decodeRanks,
    aggregateBytes: perRank.bytes * decodeRanks + perStateRank.bytes * decodeRanks,
    linkBandwidth,
    linkSourceCode: linkBandwidth ? linkSourceCode : "comm.missingLink",
    // P10（协议 Q7②）：传输时间 = bytes / 链路带宽（闭式，不建模 overlap/协议开销）。
    // bytes 取 per-decode-rank 的 KV+state（时间口径与单卡接收量一致）。
    transferSeconds: linkBandwidth
      ? (perRank.bytes + perStateRank.bytes) / linkBandwidth
      : null,
    layoutRepackRequired,
    shardFactor: perRank.shardFactor,
    stateShardFactor: perStateRank.shardFactor,
    prefillPlan: checked.prefillPlan,
    decodePlan: checked.decodePlan,
  };
}

/** 汇总给定计划的节点级通信和 PP 边界通信，供摘要或对比视图使用。
 *  P7（步骤 7）：tree root 入参退役——Graph IR 是唯一遍历路径。 */
export function planCommunicationBytes({ graph, config = {}, plan = {}, batch = 1, tokens = 1, bytesPerElement = 2, frameworkProfile, enforceSharedExpertsFusion = false } = {}) {
  let nodeBytes = 0;
  walkStructure(graph, ({ node, multiplier }) => {
    nodeBytes += nodeCommunicationBytes(node, config, plan, { batch, tokens, bytesPerElement, frameworkProfile, enforceSharedExpertsFusion }) * multiplier;
  });
  const ppBytes = pipelineP2PBytes({ batch, tokens, hidden: config.hiddenSize, bytesPerElement, pp: plan.pp ?? plan.PP ?? 1 });
  return { nodeBytes, ppBytes, totalBytes: nodeBytes + ppBytes };
}
