import { t } from "../i18n/format.js";

function formatValue(value) {
  if (Array.isArray(value)) {
    // 含对象的数组用 JSON，避免 [object Object]；纯标量数组仍用逗号拼接。
    return value.some((item) => item && typeof item === "object")
      ? JSON.stringify(value)
      : value.join(", ");
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// dataflow_edge_relations 是内部边标注载体（CED 全局 KV 等），与 dataflow_edges 冗余，
// 不在属性面板展示。
function AttributeGrid({ attributes, sourceFields, limit = 12, excludeKeys = ["class", "dataflow_edge_relations"], language = "zh" }) {
  const excluded = new Set(excludeKeys);
  const entries = Object.entries(attributes || {}).filter(([key]) => !excluded.has(key));
  const visibleEntries = limit ? entries.slice(0, limit) : entries;
  return (
    <div className="attribute-grid">
      {visibleEntries.map(([key, value]) => (
        <span key={key}>
          <em>{key === "note_code" ? "note" : key}</em>
          <strong>{key === "note_code" ? t(language, value) : formatValue(value)}</strong>
        </span>
      ))}
      {sourceFields?.length > 0 && (
        <span className="wide">
          <em>fields</em>
          <strong>{sourceFields.join(", ")}</strong>
        </span>
      )}
    </div>
  );
}

export default AttributeGrid;
