function formatValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function AttributeGrid({ attributes, sourceFields, limit = 12, excludeKeys = ["class"] }) {
  const excluded = new Set(excludeKeys);
  const entries = Object.entries(attributes || {}).filter(([key]) => !excluded.has(key));
  const visibleEntries = limit ? entries.slice(0, limit) : entries;
  return (
    <div className="attribute-grid">
      {visibleEntries.map(([key, value]) => (
        <span key={key}>
          <em>{key}</em>
          <strong>{formatValue(value)}</strong>
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
