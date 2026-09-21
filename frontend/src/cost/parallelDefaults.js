// 固定的单机部署档位，不是 TP/PP/EP/DP 优化器。与 Fit 复用同一份 framework
// accounting 和 rank 投影；不能把总显存直接除以卡数（MLA cache 与复制权重不一定切分）。
import { DEFAULT_NODES, DEFAULT_PLAN } from "./defaults.js";
import { planFitsCard, projectPlan } from "./parallel.js";

export const SINGLE_NODE_CARD_TIERS = Object.freeze([1, 2, 4, 8]);

export function recommendSingleNodePlan({ graph, accounting, config = {}, chip } = {}) {
  const candidates = SINGLE_NODE_CARD_TIERS.map((cards) => {
    const plan = { ...DEFAULT_PLAN, tp: cards };
    const capacity = chip?.memory_bytes;
    const weightLowerBound = accounting?.total?.weightBytes > 0
      ? accounting.total.weightBytes / cards + (accounting.total.bufferBytes || 0)
      : null;
    // 仅用权重下界，避免对即使 TP8 也放不下的超大模型重复 walk 四次 Graph IR。
    // 这只能提前否决；仍有可能适配的档位继续走权威 projectPlan + Fit。
    if (graph?.nodes?.length && Number.isFinite(capacity) && capacity > 0
      && weightLowerBound != null && weightLowerBound > capacity) {
      return { cards, plan, fit: false };
    }
    const projection = graph?.nodes?.length && accounting?.total?.weightBytes > 0
      ? projectPlan({ graph, accounting, config, plan }) : null;
    return { cards, plan, fit: planFitsCard(projection, capacity) ?? null };
  });
  // 即使 TP8 也放不下时保留单机边界并展示 no-fit；未知容量/账本不能冒充单卡适配。
  const selected = candidates.find(({ fit }) => fit === true) || candidates.at(-1);
  return {
    ...selected,
    candidates,
    plans: { prefill: selected.plan, decode: selected.plan },
    nodes: DEFAULT_NODES,
    // 物理拓扑保持标准单机 8 卡；小模型在该主机内使用 TP1/2/4，其余卡仍可扩展。
    gpusPerNode: 8,
  };
}
