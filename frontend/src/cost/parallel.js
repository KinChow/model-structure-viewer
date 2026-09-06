// 给定 TP/PP/EP/DP 计划的资源投影；不搜索计划，也不预测吞吐或延迟。
// 来源：llm-analysis 的并行内存分解方法，以及 evolution_design.md §5.3(6)。

import { linearStateElementsPerLayer, linearStateElementsPerSequence, nodeWeightBytes } from "./memory.js";
import { childRepeatMultiplier, walkStructure } from "./traverse.js";

function positiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function positiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/** 校验用户给定的并行计划，避免静默接受不可能的卡数配置。 */
export function validatePlan(plan = {}, config = {}) {
  const normalized = {
    tp: plan.tp ?? plan.TP ?? 1,
    pp: plan.pp ?? plan.PP ?? 1,
    ep: plan.ep ?? plan.EP ?? 1,
    dp: plan.dp ?? plan.DP ?? 1,
    worldSize: plan.worldSize ?? plan.world_size,
    attnMode: plan.attnMode ?? plan.attn_mode ?? "tp",
    vocabParallel: plan.vocabParallel ?? plan.vocab_parallel ?? true,
  };
  const errors = [];
  for (const key of ["tp", "pp", "ep", "dp"]) if (!positiveInteger(normalized[key])) errors.push(`${key} 必须是正整数`);
  const expectedWorld = normalized.tp * normalized.pp * normalized.dp;
  if (normalized.worldSize != null && normalized.worldSize !== expectedWorld) {
    errors.push(`world_size 应为 TP×PP×DP=${expectedWorld}`);
  }
  if (!["tp", "dp"].includes(normalized.attnMode)) errors.push("attn_mode 只能是 tp 或 dp");
  if (config?.experts && normalized.ep > config.experts) errors.push("EP 不能大于专家总数");
  return { ok: errors.length === 0, errors, plan: { ...normalized, worldSize: normalized.worldSize ?? expectedWorld } };
}

/**
 * 计算单卡 KV cache 字节数。
 * 来源：llm-analysis 的 get_memory_kv_cache_per_gpu；GQA/MLA/DP-attention 分支依据设计文档 F5-F7。
 */
export function kvBytesPerCard(totalKvBytes, config = {}, plan = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { bytes: null, shardFactor: null, errors: checked.errors };
  const { tp, attnMode } = checked.plan;
  const isMla = config.kvLoraRank != null && config.qkRopeHeadDim != null;
  const shardFactor = isMla || attnMode === "dp"
    ? 1
    : Math.min(tp, config.kvHeads || config.attentionHeads || 1);
  return { bytes: totalKvBytes / shardFactor, shardFactor, errors: [] };
}

/** KDA request state is sharded with attention TP, but replicated under DP-attention. */
export function stateBytesPerCard(totalStateBytes, config = {}, plan = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { bytes: null, shardFactor: null, errors: checked.errors };
  const shardFactor = checked.plan.attnMode === "dp" ? 1 : checked.plan.tp;
  return { bytes: totalStateBytes / shardFactor, shardFactor, errors: [] };
}

function stateBytesForLayerRange(totalStateBytes, config = {}, start = 0, end = -1) {
  const layers = config.layers || config.attentionSchedule?.length || 0;
  const totalElements = linearStateElementsPerSequence(config);
  if (!layers || !totalElements || end < start) return 0;
  let selected = 0;
  for (let index = Math.max(0, start); index <= Math.min(end, layers - 1); index += 1) {
    selected += linearStateElementsPerLayer(config, index);
  }
  return totalStateBytes * selected / totalElements;
}

function modulePath(node) {
  return String(node?.id || node?.name || "").toLowerCase();
}

function isRoutedExpertPath(path) {
  return /(?:^|\.)(?:experts|expert_mlp)(?:\.|$)/.test(path);
}

/** 按模块类别计算权重在单卡上的 TP/EP 投影；PP 只负责 stage 归属。 */
export function weightBytesPerCard(totalBytes, node, plan = {}) {
  const path = modulePath(node);
  const tp = plan.tp ?? plan.TP ?? 1;
  const ep = plan.ep ?? plan.EP ?? 1;
  const vocabParallel = plan.vocabParallel ?? plan.vocab_parallel ?? true;
  if (isRoutedExpertPath(path) && ep > 1) return { bytes: totalBytes / ep, divisor: ep, axis: "ep" };
  if (/(^|\.)[^.]*norm[^.]*($|\.)/.test(path)) return { bytes: totalBytes, divisor: 1, axis: "replicated" };
  if (/(embed|lm_head|output)/.test(path) && !vocabParallel) return { bytes: totalBytes, divisor: 1, axis: "replicated" };
  if (tp > 1) return { bytes: totalBytes / tp, divisor: tp, axis: "tp" };
  return { bytes: totalBytes, divisor: 1, axis: "replicated" };
}

/**
 * 把单个图节点的理论成本投影到一个 rank。
 * 来源：llm-analysis@d841e40aec8c 的 get_latency_fwd_per_layer_attn/mlp 与
 * get_activation_memory_per_layer_attn/mlp：逐卡计算量、权重和激活按并行轴切分；
 * MoE 专家沿用本文件的 EP 归属规则。这里仅做解析式除法，不模拟 kernel、通信重叠或负载不均衡。
 */
export function nodeCostPerCard(cost = {}, node, plan = {}) {
  const projection = weightBytesPerCard(cost.weightBytes || 0, node, plan);
  const divide = (value) => value == null ? value : value / projection.divisor;
  return {
    ...cost,
    macs: divide(cost.macs),
    weightBytes: projection.bytes,
    actInBytes: divide(cost.actInBytes),
    actOutBytes: divide(cost.actOutBytes),
    projection: { axis: projection.axis, divisor: projection.divisor },
  };
}

/** 专家权重在 EP rank 上的平均/最坏区间；来源：vLLM MoE 负载不均衡建模讨论。 */
export function expertWeightRange(totalBytes, experts, ep = 1) {
  if (!Number.isFinite(totalBytes) || totalBytes < 0 || !positiveInteger(experts) || !positiveInteger(ep)) {
    return { averageBytes: null, worstBytes: null, expertsPerRank: null };
  }
  const expertsPerRank = Math.ceil(experts / ep);
  return {
    averageBytes: totalBytes / ep,
    worstBytes: (totalBytes / experts) * expertsPerRank,
    expertsPerRank,
  };
}

/** 将层索引映射到 PP stage；首尾 stage 可额外承载 embedding/lm_head。 */
export function stageForLayer(layerIndex, layers, pp = 1) {
  if (!positiveInteger(pp) || !positiveInteger(layers) || layerIndex < 0 || layerIndex >= layers) return null;
  return Math.min(pp - 1, Math.floor(layerIndex * pp / layers));
}

function stageLayerBounds(stage, layers, pp) {
  return { start: Math.floor(stage * layers / pp), end: Math.floor((stage + 1) * layers / pp) - 1 };
}

function layerSpanForNode(node) {
  const range = node?.attributes?.range;
  if (typeof range === "string" && /^\d+\.\.\d+$/.test(range)) {
    const [start, end] = range.split("..").map(Number);
    return { start, end };
  }
  const path = String(node?.id || "");
  const match = path.match(/(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/);
  if (match && !/(^|\.)experts(\.|$)/.test(path) && Number.isFinite(node?.repeat) && node.repeat > 1) {
    const start = Number(match[1]);
    return { start, end: start + node.repeat - 1 };
  }
  if (!/(^|\.)experts(\.|$)/.test(path) && Number.isFinite(node?.repeat) && node.repeat > 1 && node?.children?.length === 1) {
    const childMatch = String(node.children[0]?.id || "").match(/(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/);
    if (childMatch) {
      const start = Number(childMatch[1]);
      return { start, end: start + node.repeat - 1 };
    }
  }
  return null;
}

/** 逐 stage 返回已投影的权重与 KV，供后续 fit UI 使用。 */
export function projectPlan({ weightBytes = 0, kvBytes = 0, stateBytes = 0, config = {}, plan = {} } = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, stages: [] };
  const { pp, dp } = checked.plan;
  if (arguments[0]?.root) {
    const projected = projectNodePlan({ root: arguments[0].root, targetWeightBytes: weightBytes, kvBytes, stateBytes, config, plan: checked.plan });
    if (projected.stages.some((stage) => stage.weightBytes > 0) || weightBytes <= 0) return projected;
  }
  const kv = kvBytesPerCard(kvBytes, config, checked.plan);
  const state = stateBytesPerCard(stateBytes, config, checked.plan);
  const perStageWeight = weightBytes / pp;
  return {
    ok: true,
    errors: [],
    plan: checked.plan,
    stages: Array.from({ length: pp }, (_, stage) => ({
      stage,
      ranks: checked.plan.tp * dp,
      weightBytes: perStageWeight,
      kvBytes: kv.bytes,
      stateBytes: state.bytes / pp,
      dpRanks: dp,
    })),
  };
}

/**
 * 根据 IR 节点路径把权重归属到 PP stage，避免 embedding/lm_head 被平均摊薄。
 * 来源：llm-analysis 的 get_memory_weight_per_stage；具体模块切分复用本文件的 TP/EP 规则。
 */
function treeWeightBytes(root) {
  let total = 0;
  walkStructure(root, ({ node, multiplier }) => {
    total += nodeWeightBytes(node) * multiplier;
  });
  return total;
}

export function projectNodePlan({ root, targetWeightBytes, kvBytes = 0, stateBytes = 0, config = {}, plan = {} } = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, stages: [] };
  const { pp, dp } = checked.plan;
  const naturalWeightBytes = treeWeightBytes(root);
  const weightScale = positiveNumber(targetWeightBytes) && naturalWeightBytes > 0 ? targetWeightBytes / naturalWeightBytes : 1;
  const stages = Array.from({ length: pp }, (_, stage) => ({ stage, ranks: checked.plan.tp * dp, weightBytes: 0, kvBytes: 0, stateBytes: 0, dpRanks: dp, expertWeightBytes: 0 }));
  function visit(node, inheritedRepeat = 1, inheritedLayerSpan = null) {
    const path = String(node?.id || node?.name || "").toLowerCase();
    const ownLayerSpan = layerSpanForNode(node);
    const layerSpan = ownLayerSpan || inheritedLayerSpan;
    const rawWeight = nodeWeightBytes(node) * inheritedRepeat * weightScale;
    const projected = weightBytesPerCard(rawWeight, node, checked.plan).bytes;
    const isExpert = isRoutedExpertPath(path);
    if (layerSpan && config.layers) {
      for (const stage of stages) {
        const bounds = stageLayerBounds(stage.stage, config.layers, pp);
        const overlap = Math.max(0, Math.min(layerSpan.end, bounds.end) - Math.max(layerSpan.start, bounds.start) + 1);
        if (overlap > 0) {
          stage.weightBytes += projected * overlap;
          if (isExpert) stage.expertWeightBytes += rawWeight * overlap;
        }
      }
    } else {
      let stage = 0;
      if (/(lm_head|output_head|language_model_head)/.test(path)) stage = pp - 1;
      else if (/(final_norm|norm$)/.test(path) && pp > 1) stage = pp - 1;
      stages[stage].weightBytes += projected;
      if (isExpert) stages[stage].expertWeightBytes += rawWeight;
    }
    const layerRepeatHandled = Boolean(ownLayerSpan);
    const childMultiplier = childRepeatMultiplier(node, inheritedRepeat, { repeatHandled: layerRepeatHandled });
    for (const child of node?.children || []) visit(child, childMultiplier, layerSpan);
  }
  if (root) visit(root);
  const kv = kvBytesPerCard(kvBytes, config, checked.plan);
  const state = stateBytesPerCard(stateBytes, config, checked.plan);
  for (const stage of stages) {
    const bounds = config.layers ? stageLayerBounds(stage.stage, config.layers, pp) : null;
    const stageLayers = bounds ? Math.max(0, bounds.end - bounds.start + 1) : 0;
    stage.kvBytes = config.layers ? kv.bytes * stageLayers / config.layers : kv.bytes / pp;
    stage.stateBytes = config.layers
      ? state.bytes == null ? null : state.bytes * (stateBytesForLayerRange(stateBytes, config, bounds.start, bounds.end) / Math.max(stateBytes, 1))
      : (state.bytes || 0) / pp;
    const expertRange = expertWeightRange(stage.expertWeightBytes, config.experts, checked.plan.ep);
    stage.expertWeightAverageBytes = expertRange.averageBytes;
    stage.expertWeightWorstBytes = expertRange.worstBytes;
    stage.weightAverageBytes = stage.weightBytes;
    stage.weightWorstBytes = expertRange.averageBytes != null
      ? stage.weightBytes - expertRange.averageBytes + expertRange.worstBytes
      : stage.weightBytes;
    delete stage.expertWeightBytes;
  }
  return { ok: true, errors: [], plan: checked.plan, stages, kvShardFactor: kv.shardFactor };
}

/** PD 两侧逐 stage fit；只计算显存容纳性，不预测吞吐或服务延迟。 */
export function projectPdFit({ root, weightBytes = 0, kvBytes = 0, prefillKvBytes, decodeKvBytes, stateBytes = 0, prefillStateBytes, decodeStateBytes, config = {}, pdPlan = {}, prefillChip, decodeChip, activationBytes = 0, runtimeBytes = 0, commBufferBytes = 0 } = {}) {
  const checked = validatePdPlan(pdPlan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, prefill: null, decode: null };
  function side(plan, chip, sideKvBytes, sideStateBytes) {
    const projection = projectPlan({ root, weightBytes, kvBytes: sideKvBytes, stateBytes: sideStateBytes, config, plan });
    const capacity = chip?.memory_bytes;
    const stages = projection.stages.map((stage) => {
      const totalBytes = stage.weightBytes + stage.kvBytes + (stage.stateBytes || 0) + activationBytes + runtimeBytes + commBufferBytes;
      const worstTotalBytes = (stage.weightWorstBytes ?? stage.weightBytes) + stage.kvBytes + (stage.stateBytes || 0) + activationBytes + runtimeBytes + commBufferBytes;
      return { ...stage, totalBytes, worstTotalBytes,
        fit: positiveNumber(capacity) ? totalBytes <= capacity : null,
        worstFit: positiveNumber(capacity) ? worstTotalBytes <= capacity : null };
    });
    const fit = positiveNumber(capacity) ? stages.every((stage) => stage.fit === true) : null;
    return { chipId: chip?.id || null, capacityBytes: capacity ?? null, stages, fit };
  }
  return {
    ok: true,
    errors: [],
    prefill: side(checked.prefillPlan, prefillChip, prefillKvBytes ?? kvBytes, prefillStateBytes ?? stateBytes),
    decode: side(checked.decodePlan, decodeChip, decodeKvBytes ?? kvBytes, decodeStateBytes ?? stateBytes),
  };
}

/** 给定 stage 投影下，由最紧张 stage 决定最大上下文。 */
export function maxContextForStages(stages = [], { capacityBytes, activationBytes = 0, runtimeBytes = 0, sequence = 1 } = {}) {
  if (!positiveNumber(capacityBytes) || !positiveNumber(sequence) || stages.length === 0) return null;
  const limits = stages.map((stage) => {
    const kvPerContextToken = stage.kvBytes / sequence;
    if (!positiveNumber(kvPerContextToken)) return null;
    return Math.max(0, Math.floor((capacityBytes - stage.weightBytes - (stage.stateBytes || 0) - activationBytes - runtimeBytes) / kvPerContextToken));
  }).filter((value) => value != null);
  return limits.length ? Math.min(...limits) : null;
}

/** 校验 PD 分离的 prefill/decode 两侧计划；两侧可以使用不同 TP/PP/DP。 */
export function validatePdPlan(pdPlan = {}, config = {}) {
  const prefill = validatePlan(pdPlan.prefill_plan || pdPlan.prefillPlan || {}, config);
  const decode = validatePlan(pdPlan.decode_plan || pdPlan.decodePlan || {}, config);
  return {
    ok: prefill.ok && decode.ok,
    errors: [
      ...prefill.errors.map((error) => `prefill_plan：${error}`),
      ...decode.errors.map((error) => `decode_plan：${error}`),
    ],
    prefillPlan: prefill.plan,
    decodePlan: decode.plan,
  };
}
