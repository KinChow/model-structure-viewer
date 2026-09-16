// 受控数字输入的"草稿 + 失焦提交"提交口径（业界常见做法：编辑时允许空/中间态，
// 失焦或回车时再夹取，避免每次击键就把清空态强制回填成最小值）。
// ref: 受控 number input 的通用模式——编辑期保留原始字符串，提交期规范化。

/**
 * 计算受控数字输入在提交（失焦/回车）时的最终值。
 * @param {string} raw 输入框当前原始字符串
 * @param {object} opts { min, max, fallback, allowEmpty }
 * @returns {{ value: number|undefined }}
 *   allowEmpty 且为空 → value: undefined（表示"继承/未设置"）
 *   非空非法 → fallback；否则夹到 [min, max]
 */
export function resolveNumberCommit(raw, { min = null, max = null, fallback = 1, allowEmpty = false } = {}) {
  const trimmed = typeof raw === "string" ? raw.trim() : raw;
  if (trimmed === "" || trimmed == null) {
    return { value: allowEmpty ? undefined : fallback };
  }
  let next = Number(trimmed);
  if (!Number.isFinite(next)) return { value: allowEmpty ? undefined : fallback };
  if (min != null) next = Math.max(min, next);
  if (max != null) next = Math.min(max, next);
  return { value: next };
}
