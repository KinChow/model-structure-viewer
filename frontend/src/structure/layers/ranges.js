import { deriveBuildPlan } from "../config/plan.js";
export function compactRanges(kinds) {
  if (!Array.isArray(kinds) || kinds.length === 0) return [];
  const ranges = [];
  let start = 0;
  for (let index = 1; index <= kinds.length; index += 1) {
    if (index === kinds.length || kinds[index] !== kinds[start]) {
      ranges.push({ start, end: index - 1, kind: kinds[start] });
      start = index;
    }
  }
  return ranges;
}

export function layerKinds(normalized, defaultKind) {
  const plan = deriveBuildPlan(normalized.raw ?? normalized);
  if (plan.layerSchedule?.length) return plan.layerSchedule;
  if (!normalized.layers) return [];
  return Array.from({ length: normalized.layers }, () => defaultKind);
}
