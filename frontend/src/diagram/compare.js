// 来源：Google Model Explorer 的 split panes、同节点同步和 diff highlight 交互模式。
// https://github.com/google-ai-edge/model-explorer/tree/f7378e60f6fdca4f611cc48d6603eacd46243557/src/ui/src/components/visualizer

export const COMPARISON_MODE = Object.freeze({ OFF: "off", CHIP: "chip", PLAN: "plan" });

/**
 * 构造单变量对比场景：芯片对比锁定计划，方案对比锁定芯片。
 * 这样 bound 翻转才可归因于用户选择的那一类变量。
 */
export function resolveComparisonScenario(mode, primary = {}, candidate = {}) {
  if (mode === COMPARISON_MODE.CHIP) return { chip: candidate.chip, plan: primary.plan };
  if (mode === COMPARISON_MODE.PLAN) return { chip: primary.chip, plan: candidate.plan };
  return null;
}

// 比较两个 lens 的节点 bound，输出发生翻转的路径。

export function boundFlips(primary = {}, secondary = {}) {
  const paths = new Set([...Object.keys(primary), ...Object.keys(secondary)]);
  return [...paths]
    .filter((path) => (primary[path]?.bound || "unknown") !== (secondary[path]?.bound || "unknown"))
    .map((path) => ({ path, primary: primary[path]?.bound || "unknown", secondary: secondary[path]?.bound || "unknown" }));
}
