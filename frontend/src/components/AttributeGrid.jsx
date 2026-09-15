import { t } from "../i18n/format.js";

function formatValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function AttributeGrid({ attributes, sourceFields, limit = 12, excludeKeys = ["class"], language = "zh" }) {
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
