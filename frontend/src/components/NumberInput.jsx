import { useEffect, useRef, useState } from "react";
import { resolveNumberCommit } from "../cost/numberField.js";

// 受控数字输入：聚焦编辑期允许清空/中间态（不每次击键强制回填最小值），
// 失焦或回车时再夹取到 [min, max] 并提交。合法中间值仍实时提交，保证成本
// 估算随输入即时更新；仅"空/非法"这类中间态推迟到提交时才规范化。
// ref: 受控 number input 的 draft + commit-on-blur 通用模式。
export default function NumberInput({ value, onCommit, min = null, max = null, fallback = 1, allowEmpty = false, ...rest }) {
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(value ?? (allowEmpty ? "" : String(fallback)));
  const focusedRef = useRef(false);

  useEffect(() => {
    if (!focusedRef.current) setDraft(value ?? (allowEmpty ? "" : String(fallback)));
  }, [value, allowEmpty, fallback]);

  const handleChange = (event) => {
    const raw = event.target.value;
    setDraft(raw);
    // 编辑期：仅当是合法数字时才实时提交，空/中间态不回填。
    const trimmed = raw.trim();
    if (trimmed === "") return;
    const num = Number(trimmed);
    if (!Number.isFinite(num)) return;
    let next = num;
    if (min != null) next = Math.max(min, next);
    if (max != null) next = Math.min(max, next);
    if (String(next) === trimmed) onCommit(next);
  };

  const commit = () => {
    const { value: resolved } = resolveNumberCommit(draft, { min, max, fallback: fallback ?? min ?? 1, allowEmpty });
    onCommit(resolved);
    setDraft(resolved ?? (allowEmpty ? "" : String(fallback)));
  };

  return (
    <input
      type="number"
      {...(min != null ? { min } : {})}
      {...(max != null ? { max } : {})}
      value={focused ? draft : (value ?? (allowEmpty ? "" : fallback))}
      onFocus={(event) => { focusedRef.current = true; setFocused(true); setDraft(event.target.value); }}
      onChange={handleChange}
      onBlur={() => { focusedRef.current = false; setFocused(false); commit(); }}
      onKeyDown={(event) => {
        if (event.key === "Enter") { event.currentTarget.blur(); }
        rest.onKeyDown?.(event);
      }}
      {...rest}
    />
  );
}
