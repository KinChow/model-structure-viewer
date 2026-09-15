import { t } from "../i18n/format.js";

/** Display badges for a structure node. `repeat === 0` is a billing flag, not a count. */
export function nodeBadges(node, language = "zh") {
  const repeat = node?.repeat;
  const numericRepeat = Number(repeat);
  const childCount = node?.children?.length ?? 0;
  const badges = [];
  if (Number.isFinite(numericRepeat) && numericRepeat > 1) {
    badges.push({ kind: "repeat", text: `×${numericRepeat}` });
  }
  if (numericRepeat === 0) {
    badges.push({ kind: "draft", text: t(language, "diagram.draftBadge") });
  }
  if (childCount > 0) {
    badges.push({ kind: "children", text: t(language, "diagram.childCount", { n: childCount }) });
  }
  return badges;
}

export function badgeText(node, language = "zh") {
  return nodeBadges(node, language).map((badge) => badge.text).join(" ");
}
