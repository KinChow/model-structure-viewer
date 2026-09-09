// 给定 TP/PP/EP/DP 计划的资源投影；不搜索计划，也不预测吞吐或延迟。
// 来源：llm-analysis 的并行内存分解方法，以及 evolution_design.md §5.3(6)。

import { linearStateElementsPerLayer, linearStateElementsPerSequence, nodeWeightBytes } from "./memory.js";
import { childRepeatMultiplier, graphNodeToNode, walkStructure } from "./traverse.js";
import { deriveBuildPlan } from "../structure/config/plan.js";
import { LAYER_INDEX_RE } from "../structure/formulas/extractor.js";
import { declaredWeightBytesPerCard, declaredWeightElements, expertShardDivisor } from "./sharding.js";
const planOf = (config) => deriveBuildPlan(config?.raw ?? config);

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
    // N2-4 W-B：混合 ETP 轴（TRT-LLM 语义）。缺省 undefined——组合语义由
    // sharding.js expertShardDivisor 按 EP 状态取缺省（EP 启用 moe_tp=1 完整
    // 专家、未启用 moe_tp=tp 矩阵切分），不在此处伪造数值。
    moeTp: plan.moeTp ?? plan.moe_tp,
    moeEp: plan.moeEp ?? plan.moe_ep,
    worldSize: plan.worldSize ?? plan.world_size,
    attnMode: plan.attnMode ?? plan.attn_mode ?? "tp",
    vocabParallel: plan.vocabParallel ?? plan.vocab_parallel ?? true,
  };
  const errors = [];
  for (const key of ["tp", "pp", "ep", "dp"]) if (!positiveInteger(normalized[key])) errors.push(`${key} 必须是正整数`);
  for (const key of ["moeTp", "moeEp"]) {
    if (normalized[key] != null && !positiveInteger(normalized[key])) errors.push(`${key} 必须是正整数`);
  }
  const expectedWorld = normalized.tp * normalized.pp * normalized.dp;
  if (normalized.worldSize != null && normalized.worldSize !== expectedWorld) {
    errors.push(`world_size 应为 TP×PP×DP=${expectedWorld}`);
  }
  if (!["tp", "dp"].includes(normalized.attnMode)) errors.push("attn_mode 只能是 tp 或 dp");
  if (config?.experts && normalized.ep > config.experts) errors.push("EP 不能大于专家总数");
  if (config?.experts && normalized.moeEp > config.experts) errors.push("moe_ep 不能大于专家总数");
  // vLLM 组合语义（AMD playbook / DP 文档核实）：EP 启用时 ep_size = tp × dp
  // （DP attention + EP 是 DeepSeek 系标准部署）。仅在依赖 ep 缺省（未显式声明
  // moe_ep 的混合 ETP 不受此约束）且 DP attention 时硬校验。
  if (normalized.ep > 1 && normalized.attnMode === "dp" && normalized.moeEp == null
    && normalized.ep !== normalized.tp * normalized.dp) {
    errors.push(`EP 启用时 ep 应为 TP×DP=${normalized.tp * normalized.dp}（vLLM：EP_SIZE = TP_SIZE × DP_SIZE）`);
  }
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
  const layers = config.layers || planOf(config).attentionSchedule?.length || 0;
  const totalElements = linearStateElementsPerSequence(config);
  if (!layers || !totalElements || end < start) return 0;
  let selected = 0;
  for (let index = Math.max(0, start); index <= Math.min(end, layers - 1); index += 1) {
    selected += linearStateElementsPerLayer(config, index);
  }
  return totalStateBytes * selected / totalElements;
}

function modulePath(node) {
  return String(node?.id || "").toLowerCase();
}

function isRoutedExpertPath(path) {
  return /(?:^|\.)(?:experts|expert_mlp)(?:\.|$)/.test(path);
}

/** 按模块类别计算权重在单卡上的 TP/EP 投影；PP 只负责 stage 归属。 */
// 单卡投影规则表（顺序敏感，首条命中生效）。来源：llm-analysis TP/EP 投影
// 语义——① 路由专家按 sharding.js 的组合语义切（EP 启用 ÷moe_ep、未启用
// ÷moe_tp×dp——vLLM DP-shards-experts，N2-4 W-B）；② norm 复制；③ 词表并行
// 关闭时 embed/lm_head 复制；④ 其余 TP>1 按 TP 切；⑤ 默认复制。新增规则加表项。
// 有 weightMatrices 声明的叶子不走本表（weightBytesPerCard 声明优先，sharding.js
// declaredClassDivisor 与本表逐条同义——锚 3）。
const WEIGHT_PROJECTION_RULES = [
  {
    axis: "ep",
    when: (path, ctx) => isRoutedExpertPath(path) && (ctx.ep > 1 || ctx.dp > 1 || (ctx.moeEp ?? 1) > 1 || (ctx.moeTp ?? 1) > 1),
    divisor: (ctx) => expertShardDivisor(ctx).divisor,
  },
  { axis: "replicated", when: (path) => /(^|\.)[^.]*norm[^.]*($|\.)/.test(path), divisor: () => 1 },
  { axis: "replicated", when: (path, ctx) => /(embed|lm_head|output)/.test(path) && !ctx.vocabParallel, divisor: () => 1 },
  { axis: "tp", when: (path, ctx) => ctx.tp > 1, divisor: (ctx) => ctx.tp },
  { axis: "replicated", when: () => true, divisor: () => 1 },
];

export function weightBytesPerCard(totalBytes, node, plan = {}) {
  // N2-4 W-B：声明优先（feature flag = weightMatrices 存在，无声明叶逐位走
  // 规则表回退）。组级 class 投影见 sharding.js；axis/divisor 取主导组，
  // 供 nodeCostPerCard 分摊 compute。
  const declaration = node?.attributes?.weightMatrices;
  if (Array.isArray(declaration) && declaration.length > 0) {
    return declaredWeightBytesPerCard(totalBytes, declaration, plan);
  }
  const path = modulePath(node);
  const ctx = {
    tp: plan.tp ?? plan.TP ?? 1,
    ep: plan.ep ?? plan.EP ?? 1,
    dp: plan.dp ?? plan.DP ?? 1,
    moeTp: plan.moeTp ?? plan.moe_tp,
    moeEp: plan.moeEp ?? plan.moe_ep,
    vocabParallel: plan.vocabParallel ?? plan.vocab_parallel ?? true,
  };
  const rule = WEIGHT_PROJECTION_RULES.find((candidate) => candidate.when(path, ctx));
  return { bytes: totalBytes / rule.divisor(ctx), divisor: rule.divisor(ctx), axis: rule.axis };
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
  // M11-P0-4：动作向量与标量同轴投影——vector/sfu/bytes 与 macs 一样按切分
  // 维度除到每卡；分量未知保持 null（不伪造零）。
  const actions = cost.actions ? {
    matrix: divide(cost.actions.matrix),
    vector: divide(cost.actions.vector),
    sfu: divide(cost.actions.sfu),
    bytes: {
      weights: divide(cost.actions.bytes?.weights),
      actIn: divide(cost.actions.bytes?.actIn),
      actOut: divide(cost.actions.bytes?.actOut),
    },
    commBytes: cost.actions.commBytes ?? null,
  } : undefined;
  return {
    ...cost,
    ...(actions ? { actions } : {}),
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

// 层 span 提取复用 extractor 的共享正则（M11-P2-7 收敛）；
// stageForLayer 旧原语已被 stageLayerBounds 内联取代（生产零调用，M11-P2 删除）。
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
  const match = path.match(LAYER_INDEX_RE);
  if (match && !/(^|\.)experts(\.|$)/.test(path) && Number.isFinite(node?.repeat) && node.repeat > 1) {
    const start = Number(match[1]);
    return { start, end: start + node.repeat - 1 };
  }
  if (!/(^|\.)experts(\.|$)/.test(path) && Number.isFinite(node?.repeat) && node.repeat > 1 && node?.children?.length === 1) {
    const childMatch = String(node.children[0]?.id || "").match(LAYER_INDEX_RE);
    if (childMatch) {
      const start = Number(childMatch[1]);
      return { start, end: start + node.repeat - 1 };
    }
  }
  return null;
}

/** 逐 stage 返回已投影的权重与 KV，供后续 fit UI 使用。 */
export function projectPlan({ root, graph, weightBytes = 0, kvBytes = 0, stateBytes = 0, config = {}, plan = {} } = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, stages: [] };
  const { pp, dp } = checked.plan;
  if (root || graph) {
    const projected = projectNodePlan({ root, graph, targetWeightBytes: weightBytes, kvBytes, stateBytes, config, plan: checked.plan });
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
 * N2-4 W-B：无 weight_shapes 的声明叶（内置模型默认路径）按 weightMatrices
 * 声明计驻留字节（bf16）——此前派生路径全零，专家/非专家的 EP/TP 切分塌缩，
 * 树投影整体死路（stages 恒 0 → 退平摊）。
 */
function nodeResidentWeightBytes(node) {
  return nodeWeightBytes(node) || declaredWeightElements(node?.attributes?.weightMatrices) * 2;
}

function treeWeightBytes(root, graph) {
  let total = 0;
  walkStructure(root, ({ node, multiplier }) => {
    total += nodeResidentWeightBytes(node) * multiplier;
  }, graph);
  return total;
}

export function projectNodePlan({ root, graph, targetWeightBytes, kvBytes = 0, stateBytes = 0, config = {}, plan = {} } = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, stages: [] };
  const { pp, dp } = checked.plan;
  const naturalWeightBytes = treeWeightBytes(root, graph);
  const weightScale = positiveNumber(targetWeightBytes) && naturalWeightBytes > 0 ? targetWeightBytes / naturalWeightBytes : 1;
  const stages = Array.from({ length: pp }, (_, stage) => ({ stage, ranks: checked.plan.tp * dp, weightBytes: 0, kvBytes: 0, stateBytes: 0, dpRanks: dp, expertWeightBytes: 0, expertCount: null }));
  function accountNode(node, inheritedRepeat, inheritedLayerSpan, children, visitChild) {
    const nodeForScope = children.length ? { ...node, children } : node;
    const path = String(node?.id || "").toLowerCase();
    const ownLayerSpan = layerSpanForNode(nodeForScope);
    const layerSpan = ownLayerSpan || inheritedLayerSpan;
    const rawWeight = nodeResidentWeightBytes(node) * inheritedRepeat * weightScale;
    const projected = weightBytesPerCard(rawWeight, node, checked.plan).bytes;
    const isExpert = isRoutedExpertPath(path);
    // N2-4 W-B：expertWeightRange 的专家数接声明的 count（声明即语义——不再
    // 从 config 反推）；无声明的专家叶保持 config.experts。
    const declaredCount = Array.isArray(node?.attributes?.weightMatrices)
      ? node.attributes.weightMatrices.filter((group) => group.class === "ep").reduce((sum, group) => sum + (group.count ?? 1), 0)
      : null;
    if (layerSpan && config.layers) {
      for (const stage of stages) {
        const bounds = stageLayerBounds(stage.stage, config.layers, pp);
        const overlap = Math.max(0, Math.min(layerSpan.end, bounds.end) - Math.max(layerSpan.start, bounds.start) + 1);
        if (overlap > 0) {
          stage.weightBytes += projected * overlap;
          if (isExpert) {
            stage.expertWeightBytes += rawWeight * overlap;
            if (declaredCount) stage.expertCount = declaredCount;
          }
        }
      }
    } else {
      let stage = 0;
      if (/(lm_head|output_head|language_model_head)/.test(path)) stage = pp - 1;
      else if (/(final_norm|norm$)/.test(path) && pp > 1) stage = pp - 1;
      stages[stage].weightBytes += projected;
      if (isExpert) {
        stages[stage].expertWeightBytes += rawWeight;
        if (declaredCount) stages[stage].expertCount = declaredCount;
      }
    }
    const layerRepeatHandled = Boolean(ownLayerSpan);
    const childMultiplier = childRepeatMultiplier(nodeForScope, inheritedRepeat, { repeatHandled: layerRepeatHandled });
    for (const child of children) visitChild(child, childMultiplier, layerSpan);
  }
  function visitTree(node, inheritedRepeat = 1, inheritedLayerSpan = null) {
    accountNode(node, inheritedRepeat, inheritedLayerSpan, node?.children || [], visitTree);
  }
  function visitGraph(graphValue) {
    const byId = new Map((graphValue.nodes || []).map((node) => [node.id, node]));
    const childrenByParent = new Map();
    for (const node of graphValue.nodes || []) {
      if (node.parent_id == null) continue;
      const children = childrenByParent.get(node.parent_id) || [];
      children.push(node);
      childrenByParent.set(node.parent_id, children);
    }
    for (const children of childrenByParent.values()) children.sort((left, right) => (left.order || 0) - (right.order || 0));
    function visitNode(nodeId, inheritedRepeat = 1, inheritedLayerSpan = null) {
      const graphNode = byId.get(nodeId);
      if (!graphNode) return;
      const node = graphNodeToNode(graphNode);
      const children = childrenByParent.get(nodeId) || [];
      const childNodes = children.map(graphNodeToNode);
      accountNode(node, inheritedRepeat, inheritedLayerSpan, childNodes, (child, multiplier, layerSpan) => {
        visitNode(children.find((candidate) => (candidate.module_id || candidate.id) === child.id)?.id, multiplier, layerSpan);
      });
    }
    visitNode(graphValue.root_id || graphValue.nodes.find((node) => node.parent_id == null)?.id || "root");
  }
  if (graph?.nodes?.length) visitGraph(graph);
  else if (root) visitTree(root);
  const kv = kvBytesPerCard(kvBytes, config, checked.plan);
  const state = stateBytesPerCard(stateBytes, config, checked.plan);
  for (const stage of stages) {
    const bounds = config.layers ? stageLayerBounds(stage.stage, config.layers, pp) : null;
    const stageLayers = bounds ? Math.max(0, bounds.end - bounds.start + 1) : 0;
    stage.kvBytes = config.layers ? kv.bytes * stageLayers / config.layers : kv.bytes / pp;
    stage.stateBytes = config.layers
      ? state.bytes == null ? null : state.bytes * (stateBytesForLayerRange(stateBytes, config, bounds.start, bounds.end) / Math.max(stateBytes, 1))
      : (state.bytes || 0) / pp;
    // N2-4 W-B：不均衡区间的集合切分度来自组合语义（EP 启用 = epSize，未启用 =
    // dp——DP 也切专家集合）；专家数优先用声明的 count，缺省回 config.experts。
    const sharding = expertShardDivisor(checked.plan);
    const expertRange = expertWeightRange(stage.expertWeightBytes, stage.expertCount ?? config.experts, sharding.setDegree);
    stage.expertWeightAverageBytes = expertRange.averageBytes;
    stage.expertWeightWorstBytes = expertRange.worstBytes;
    stage.weightAverageBytes = stage.weightBytes;
    stage.weightWorstBytes = expertRange.averageBytes != null
      ? stage.weightBytes - expertRange.averageBytes + expertRange.worstBytes
      : stage.weightBytes;
    delete stage.expertWeightBytes;
    delete stage.expertCount;
  }
  return { ok: true, errors: [], plan: checked.plan, stages, kvShardFactor: kv.shardFactor };
}

/** PD 两侧逐 stage fit；只计算显存容纳性，不预测吞吐或服务延迟。 */
export function projectPdFit({ root, graph, weightBytes = 0, kvBytes = 0, prefillKvBytes, decodeKvBytes, stateBytes = 0, prefillStateBytes, decodeStateBytes, config = {}, pdPlan = {}, prefillChip, decodeChip, activationBytes = 0, runtimeBytes = 0, commBufferBytes = 0 } = {}) {
  const checked = validatePdPlan(pdPlan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, prefill: null, decode: null };
  function side(plan, chip, sideKvBytes, sideStateBytes) {
    const projection = projectPlan({ root, graph, weightBytes, kvBytes: sideKvBytes, stateBytes: sideStateBytes, config, plan });
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
