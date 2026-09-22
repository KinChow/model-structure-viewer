// 给定 TP/PP/EP/DP 计划的资源投影；不搜索计划，也不预测吞吐或延迟。
// 容量 walk 图声明；切分规则见 details/parallel_protocol.md。不再参考 llm-analysis。

import { nodeWeightCapacityBytes } from "./memory.js";
import { childResidentRepeat, graphNodeToNode, walkStructure } from "./traverse.js";
import { LAYER_INDEX_RE } from "../structure/operators/formulas/extractor.js";
import { declaredWeightBytesPerCard, expertShardDivisor, resolveFrameworkPlan } from "./sharding.js";
import { normalizeParallelPlan } from "./parallelPlan.js";

function positiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function positiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/** 校验用户给定的并行计划，避免静默接受不可能的卡数配置。
 *  P6：归一化与校验单源迁至 parallelPlan.js（协议 Q8：并行 plan 独立 schema），
 *  本函数保留原签名委托——validatePlan 的全部消费者（kvBytesPerCard/state/
 *  projectPlan/PD/lens/comm）行为不变。 */
export function validatePlan(plan = {}, config = {}) {
  return normalizeParallelPlan(plan, config);
}

/**
 * 计算单卡 KV cache 字节数。
 * 除法规则见 details/parallel_protocol.md 与原则 §3.6：GQA = min(TP, kv_heads)，MLA 不切，DP-attention 复制。
 */
export function kvBytesPerCard(totalKvBytes, config = {}, plan = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { bytes: null, shardFactor: null, errors: checked.errors };
  const { tp, attnMode } = checked.plan;
  const isMla = config.kvLoraRank != null && config.qkRopeHeadDim != null;
  const shardFactor = isMla || attnMode === "dp"
    ? 1
    : Math.min(tp, config.kvHeads || config.attentionHeads || 1);
  // P10：KV keep-ratio（streaming/滑窗估算口径，非运行时行为承诺）。
  const keepRatio = checked.plan.kvKeepRatio ?? 1;
  return { bytes: (totalKvBytes / shardFactor) * keepRatio, shardFactor, keepRatio, errors: [] };
}

/** KDA request state is sharded with attention TP, but replicated under DP-attention. */
export function stateBytesPerCard(totalStateBytes, config = {}, plan = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { bytes: null, shardFactor: null, errors: checked.errors };
  const shardFactor = checked.plan.attnMode === "dp" ? 1 : checked.plan.tp;
  return { bytes: totalStateBytes / shardFactor, shardFactor, errors: [] };
}

function stateBytesForLayerRange(graph, totalStateBytes, start = 0, end = -1) {
  if (!graph?.nodes?.length || end < start) return 0;
  let selected = 0;
  let total = 0;
  walkStructure(graph, ({ node, multiplier }) => {
    const elements = (node?.attributes?.state_elements || 0) * multiplier;
    if (!elements) return;
    total += elements;
    const match = String(node?.id || "").match(LAYER_INDEX_RE);
    const layer = match ? Number(match[1]) : null;
    if (layer != null && layer >= start && layer <= end) selected += elements;
  });
  if (!total) return 0;
  return totalStateBytes * selected / total;
}

/** 按模块类别计算权重在单卡上的 TP/EP 投影；PP 只负责 stage 归属。
 *
 * P5（执行路线步骤 3 收口）：路径正则规则表已删除——weightMatrices 是权重
 * 归属**唯一**入口（`details/parallel_protocol.md` §一分层纪律）。P2 覆盖率
 * 棘轮归零后（18399/18399 带权叶全声明），回退路径失去存在理由；保留它只会
 * 让"路径猜归属"的知识复活（router ÷tp 的分片轴错误正是规则表时代的产物）。
 * 无声明的带权叶（新接入模型未声明时）返回 axis: "unknown"——诚实缺项，
 * 不再猜；覆盖率护栏测试会立即红并指认该叶。
 */
export function weightBytesPerCard(totalBytes, node, plan = {}) {
  const declaration = node?.attributes?.weightMatrices;
  if (Array.isArray(declaration) && declaration.length > 0) {
    return declaredWeightBytesPerCard(totalBytes, declaration, plan, node);
  }
  if (totalBytes > 0) {
    // 无声明但确有驻留权重：unknown（不伪造归属，不静默按复制处理）。
    return { bytes: totalBytes, divisor: 1, axis: "unknown" };
  }
  return { bytes: 0, divisor: 1, axis: "replicated" };
}

/**
 * 把单个图节点的理论成本投影到一个 rank。
 * 权重 / MAC 按 class 切；激活按 weightMatrices.split（Megatron Column/Row）。
 * 仅做解析式除法，不模拟 kernel、通信重叠或负载不均衡。
 */
export function nodeCostPerCard(cost = {}, node, plan = {}) {
  const projection = weightBytesPerCard(cost.weightBytes || 0, node, plan);
  const tp = plan.tp ?? plan.TP ?? 1;
  const split = activationSplitOf(node);
  const actInDivisor = split === "input" ? tp : 1;
  const actOutDivisor = split === "output" ? tp : 1;
  const divide = (value, divisor) => value == null ? value : value / divisor;
  // MAC / 权重跟 class；actIn/actOut 跟 split。未知保持 null。
  const actions = cost.actions ? {
    matrix: divide(cost.actions.matrix, projection.divisor),
    vector: divide(cost.actions.vector, projection.divisor),
    sfu: divide(cost.actions.sfu, projection.divisor),
    bytes: {
      weights: divide(cost.actions.bytes?.weights, projection.divisor),
      actIn: divide(cost.actions.bytes?.actIn, actInDivisor),
      actOut: divide(cost.actions.bytes?.actOut, actOutDivisor),
    },
    commBytes: cost.actions.commBytes ?? null,
  } : undefined;
  return {
    ...cost,
    ...(actions ? { actions } : {}),
    macs: divide(cost.macs, projection.divisor),
    weightBytes: projection.bytes,
    actInBytes: divide(cost.actInBytes, actInDivisor),
    actOutBytes: divide(cost.actOutBytes, actOutDivisor),
    projection: { axis: projection.axis, divisor: projection.divisor, split },
  };
}

/** ColumnParallel = output（输入完整、输出 /TP）；RowParallel = input（输入 /TP、输出完整）。 */
function activationSplitOf(node) {
  const groups = node?.attributes?.weightMatrices;
  if (!Array.isArray(groups) || groups.length === 0) return null;
  const dominant = groups.reduce((best, group) => {
    const elements = (group.count ?? 1) * (group.matrices ?? 1) * (group.out || 0) * (group.in || 0);
    if (!best || elements > best.elements) return { split: group.split || null, elements };
    return best;
  }, null);
  return dominant?.split || null;
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
  if (/(^|\.)mtp(\.|$)/.test(path)) return null;
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
// P7（步骤 7）：root 兜底取点退役——图缺位时才走平坦摊薄的兜底投影。
export function projectPlan({ graph, accounting, weightBytes = 0, kvBytes = 0, stateBytes = 0, config = {}, plan = {} } = {}) {
  if (accounting) {
    const effective = resolveFrameworkPlan(plan, accounting.framework, config);
    const projection = projectPlan({ graph, weightBytes: accounting.total.weightBytes, config, plan: effective });
    if (!projection.ok) return projection;
    const pp = projection.plan.pp;
    const fraction = (scope, stage) => {
      if (scope?.placement === "last") return stage === pp - 1 ? 1 : 0;
      if (!scope || scope.placement === "first") return stage === 0 ? 1 : 0;
      const bounds = stageLayerBounds(stage, config.layers || scope.end + 1, pp);
      return Math.max(0, Math.min(scope.end, bounds.end) - Math.max(scope.start, bounds.start) + 1)
        / (scope.end - scope.start + 1);
    };
    for (const stage of projection.stages) {
      stage.kvBytes = 0;
      stage.boundedKvBytes = 0;
      stage.stateBytes = 0;
      stage.speculativeStateBytes = 0;
      stage.bufferBytes = 0;
      for (const pool of accounting.pools) {
        const share = fraction(pool.scope, stage.stage);
        stage.kvBytes += kvBytesPerCard(pool.kvBytes * share, config, projection.plan).bytes;
        stage.boundedKvBytes += kvBytesPerCard(pool.boundedKvBytes * share, config, projection.plan).bytes;
        stage.stateBytes += stateBytesPerCard(pool.stateBytes * share, config, projection.plan).bytes;
        stage.speculativeStateBytes += stateBytesPerCard(pool.speculativeStateBytes * share, config, projection.plan).bytes;
      }
      for (const buffer of accounting.buffers) stage.bufferBytes += buffer.bytes * fraction(buffer.scope, stage.stage);
      stage.totalBytes = stage.weightBytes + stage.kvBytes + stage.stateBytes
        + stage.speculativeStateBytes + stage.bufferBytes;
    }
    return { ...projection, accounting };
  }
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, stages: [] };
  const { pp, dp } = checked.plan;
  if (graph?.nodes?.length) {
    const projected = projectNodePlan({ graph, targetWeightBytes: weightBytes, kvBytes, stateBytes, config, plan: checked.plan });
    if (projected.stages.some((stage) => stage.weightBytes > 0) || weightBytes <= 0) return projected;
  }
  const kv = kvBytesPerCard(kvBytes, config, checked.plan);
  const state = stateBytesPerCard(stateBytes, config, checked.plan);
  const perStageWeight = weightBytes / pp;
  return {
    ok: true,
    errors: [],
    plan: checked.plan,
    stages: Array.from({ length: pp }, (_, stage) => {
      const kvStage = kv.bytes;
      const stateStage = state.bytes / pp;
      return {
        stage,
        ranks: checked.plan.tp * dp,
        weightBytes: perStageWeight,
        kvBytes: kvStage,
        stateBytes: stateStage,
        dpRanks: dp,
        totalBytes: perStageWeight + kvStage + (stateStage || 0),
      };
    }),
  };
}

/**
 * 根据 IR 节点路径把权重归属到 PP stage，避免 embedding/lm_head 被平均摊薄。
 * 模块切分走本文件的 TP/EP 规则（weightMatrices）。
 * N2-4 W-B：无 weight_shapes 的声明叶（内置模型默认路径）按 weightMatrices
 * 声明计驻留字节（bf16）——此前派生路径全零，专家/非专家的 EP/TP 切分塌缩，
 * 树投影整体死路（stages 恒 0 → 退平摊）。
 */
function graphWeightBytes(graph) {
  let total = 0;
  walkStructure(graph, ({ node, resident }) => {
    total += nodeWeightCapacityBytes(node) * resident;
  });
  return total;
}

// P7（步骤 7）：tree root 入参与 visitTree 回退分支退役——Graph IR 是唯一归属路径。
export function projectNodePlan({ graph, targetWeightBytes, kvBytes = 0, stateBytes = 0, config = {}, plan = {} } = {}) {
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, stages: [] };
  const { pp, dp } = checked.plan;
  const naturalWeightBytes = graphWeightBytes(graph);
  const weightScale = Number.isFinite(targetWeightBytes) && targetWeightBytes >= 0 && naturalWeightBytes > 0
    ? targetWeightBytes / naturalWeightBytes : 1;
  const stages = Array.from({ length: pp }, (_, stage) => ({ stage, ranks: checked.plan.tp * dp, weightBytes: 0, kvBytes: 0, stateBytes: 0, dpRanks: dp, expertWeightBytes: 0, expertCount: null }));
  function accountNode(node, inheritedRepeat, inheritedLayerSpan, children, visitChild) {
    const nodeForScope = children.length ? { ...node, children } : node;
    const path = String(node?.id || "").toLowerCase();
    const ownLayerSpan = layerSpanForNode(nodeForScope);
    const layerSpan = ownLayerSpan || inheritedLayerSpan;
    const rawWeight = nodeWeightCapacityBytes(node) * inheritedRepeat * weightScale;
    const projected = weightBytesPerCard(rawWeight, node, checked.plan).bytes;
    // P5：专家块识别改走声明（ep 组在场）——与权重归属同一事实源，路径正则
    // 随规则表一起退役。
    const isExpert = Array.isArray(node?.attributes?.weightMatrices)
      && node.attributes.weightMatrices.some((group) => group.class === "ep");
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
      if (/(lm_head|output_head|language_model_head|(?:^|\.)mtp(?:\.|$))/.test(path)) stage = pp - 1;
      else if (/(final_norm|norm$)/.test(path) && pp > 1) stage = pp - 1;
      stages[stage].weightBytes += projected;
      if (isExpert) {
        stages[stage].expertWeightBytes += rawWeight;
        if (declaredCount) stages[stage].expertCount = declaredCount;
      }
    }
    const layerRepeatHandled = Boolean(ownLayerSpan);
    const childMultiplier = childResidentRepeat(nodeForScope, inheritedRepeat, { repeatHandled: layerRepeatHandled });
    for (const child of children) visitChild(child, childMultiplier, layerSpan);
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
  const kv = kvBytesPerCard(kvBytes, config, checked.plan);
  const state = stateBytesPerCard(stateBytes, config, checked.plan);
  for (const stage of stages) {
    const bounds = config.layers ? stageLayerBounds(stage.stage, config.layers, pp) : null;
    const stageLayers = bounds ? Math.max(0, bounds.end - bounds.start + 1) : 0;
    stage.kvBytes = config.layers ? kv.bytes * stageLayers / config.layers : kv.bytes / pp;
    stage.stateBytes = config.layers
      ? state.bytes == null ? null : state.bytes * (stateBytesForLayerRange(graph, stateBytes, bounds.start, bounds.end) / Math.max(stateBytes, 1))
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
    stage.totalBytes = stage.weightBytes + stage.kvBytes + (stage.stateBytes || 0);
    delete stage.expertWeightBytes;
    delete stage.expertCount;
  }
  return { ok: true, errors: [], plan: checked.plan, stages, kvShardFactor: kv.shardFactor };
}

/** PD 两侧逐 stage fit；只计算显存容纳性，不预测吞吐或服务延迟。
 *  P7（步骤 7）：root 入参退役，projectPlan 与本函数一致只收 Graph IR。 */
export function projectPdFit({ graph, prefillAccounting, decodeAccounting, weightBytes = 0, kvBytes = 0, prefillKvBytes, decodeKvBytes, stateBytes = 0, prefillStateBytes, decodeStateBytes, config = {}, pdPlan = {}, prefillChip, decodeChip } = {}) {
  if (prefillAccounting || decodeAccounting) pdPlan = {
    prefill_plan: resolveFrameworkPlan(pdPlan.prefill_plan || pdPlan.prefillPlan, prefillAccounting?.framework, config),
    decode_plan: resolveFrameworkPlan(pdPlan.decode_plan || pdPlan.decodePlan, decodeAccounting?.framework, config),
  };
  const checked = validatePdPlan(pdPlan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, prefill: null, decode: null };
  function side(plan, chip, sideKvBytes, sideStateBytes, accounting) {
    const projection = projectPlan({ graph, accounting, weightBytes, kvBytes: sideKvBytes, stateBytes: sideStateBytes, config, plan });
    const capacity = chip?.memory_bytes;
    const stages = projection.stages.map((stage) => {
      const totalBytes = stage.totalBytes;
      const worstTotalBytes = (stage.weightWorstBytes ?? stage.weightBytes) + stage.kvBytes
        + (stage.stateBytes || 0) + (stage.speculativeStateBytes || 0) + (stage.bufferBytes || 0);
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
    prefill: side(checked.prefillPlan, prefillChip, prefillKvBytes ?? kvBytes, prefillStateBytes ?? stateBytes, prefillAccounting),
    decode: side(checked.decodePlan, decodeChip, decodeKvBytes ?? kvBytes, decodeStateBytes ?? stateBytes, decodeAccounting),
  };
}

/** 集中式 Fit / card：只比投影后每卡驻留和芯片容量。plan 无效 → 未知。 */
export function planFitsCard(projection, capacityBytes) {
  if (projection == null) return undefined;
  if (!projection.ok) return null;
  if (!positiveNumber(capacityBytes) || !projection.stages?.length) return null;
  return projection.stages.every((stage) => stage.totalBytes <= capacityBytes);
}

/** 给定 stage 投影下，由最紧张 stage 决定最大上下文。 */
export function maxContextForStages(stages = [], { capacityBytes, sequence = 1 } = {}) {
  if (!positiveNumber(capacityBytes) || !positiveNumber(sequence) || stages.length === 0) return null;
  const limits = stages.map((stage) => {
    const fixedBytes = stage.weightBytes + (stage.stateBytes || 0)
      + (stage.speculativeStateBytes || 0) + (stage.bufferBytes || 0)
      + (stage.boundedKvBytes || 0);
    if (fixedBytes > capacityBytes) return 0;
    const kvPerContextToken = (stage.kvBytes - (stage.boundedKvBytes || 0)) / sequence;
    if (!positiveNumber(kvPerContextToken)) return null;
    return Math.max(0, Math.floor((capacityBytes - fixedBytes) / kvPerContextToken));
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
      ...prefill.errors.map((error) => ({ code: "plan.pdWrap", params: { phase: "prefill" }, inner: error })),
      ...decode.errors.map((error) => ({ code: "plan.pdWrap", params: { phase: "decode" }, inner: error })),
    ],
    prefillPlan: prefill.plan,
    decodePlan: decode.plan,
  };
}
