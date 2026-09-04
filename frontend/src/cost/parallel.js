// 给定 TP/PP/EP/DP 计划的资源投影；不搜索计划，也不预测吞吐或延迟。
// 来源：llm-analysis 的并行内存分解方法，以及 evolution_design.md §5.3(6)。

import { nodeWeightBytes } from "./memory.js";

function positiveInteger(value) {
  return Number.isInteger(value) && value > 0;
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

function modulePath(node) {
  return String(node?.id || node?.name || "").toLowerCase();
}

/** 按模块类别计算权重在单卡上的 TP/EP 投影；PP 只负责 stage 归属。 */
export function weightBytesPerCard(totalBytes, node, plan = {}) {
  const path = modulePath(node);
  const tp = plan.tp ?? plan.TP ?? 1;
  const ep = plan.ep ?? plan.EP ?? 1;
  const vocabParallel = plan.vocabParallel ?? plan.vocab_parallel ?? true;
  if (/(^|\.)(experts)(\.|$)/.test(path) && ep > 1) return { bytes: totalBytes / ep, divisor: ep, axis: "ep" };
  if (/(^|\.)[^.]*norm[^.]*($|\.)/.test(path)) return { bytes: totalBytes, divisor: 1, axis: "replicated" };
  if (/(embed|lm_head|output)/.test(path) && !vocabParallel) return { bytes: totalBytes, divisor: 1, axis: "replicated" };
  if (tp > 1) return { bytes: totalBytes / tp, divisor: tp, axis: "tp" };
  return { bytes: totalBytes, divisor: 1, axis: "replicated" };
}

/** 将层索引映射到 PP stage；首尾 stage 可额外承载 embedding/lm_head。 */
export function stageForLayer(layerIndex, layers, pp = 1) {
  if (!positiveInteger(pp) || !positiveInteger(layers) || layerIndex < 0 || layerIndex >= layers) return null;
  return Math.min(pp - 1, Math.floor(layerIndex * pp / layers));
}

/** 逐 stage 返回已投影的权重与 KV，供后续 fit UI 使用。 */
export function projectPlan({ weightBytes = 0, kvBytes = 0, config = {}, plan = {} } = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, stages: [] };
  const { pp, dp } = checked.plan;
  if (arguments[0]?.root) return projectNodePlan({ root: arguments[0].root, kvBytes, config, plan: checked.plan });
  const kv = kvBytesPerCard(kvBytes, config, checked.plan);
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
      dpRanks: dp,
    })),
  };
}

/**
 * 根据 IR 节点路径把权重归属到 PP stage，避免 embedding/lm_head 被平均摊薄。
 * 来源：llm-analysis 的 get_memory_weight_per_stage；具体模块切分复用本文件的 TP/EP 规则。
 */
export function projectNodePlan({ root, kvBytes = 0, config = {}, plan = {} } = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, stages: [] };
  const { pp, dp } = checked.plan;
  const stages = Array.from({ length: pp }, (_, stage) => ({ stage, ranks: checked.plan.tp * dp, weightBytes: 0, kvBytes: 0, dpRanks: dp }));
  function visit(node, inheritedRepeat = 1) {
    const path = String(node?.id || node?.name || "").toLowerCase();
    const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
    const rawWeight = nodeWeightBytes(node) * inheritedRepeat;
    const projected = weightBytesPerCard(rawWeight, node, checked.plan).bytes;
    let stage = 0;
    const layerMatch = path.match(/(?:^|\.)(?:layers|decoder)\.(\d+)(?:\.|$)/);
    if (layerMatch && config.layers) stage = stageForLayer(Number(layerMatch[1]), config.layers, pp) ?? 0;
    else if (/(lm_head|output_head|language_model_head)/.test(path)) stage = pp - 1;
    else if (/(final_norm|norm$)/.test(path) && pp > 1) stage = pp - 1;
    stages[stage].weightBytes += projected;
    for (const child of node?.children || []) visit(child, inheritedRepeat * repeat);
  }
  if (root) visit(root);
  const kv = kvBytesPerCard(kvBytes, config, checked.plan);
  for (const stage of stages) stage.kvBytes = kv.bytes;
  return { ok: true, errors: [], plan: checked.plan, stages, kvShardFactor: kv.shardFactor };
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
