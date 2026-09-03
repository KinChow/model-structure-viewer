import AttributeGrid from "./AttributeGrid";
import ShapeFlow from "./ShapeFlow";

function formatCount(n) {
  if (n == null || !Number.isFinite(n)) return null;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function TruthSection({ node }) {
  const hasTruth =
    node.params != null ||
    node.dtype ||
    (node.weight_shapes && Object.keys(node.weight_shapes).length > 0);
  if (!hasTruth) return null;
  const sourceLabel = node.value_source === "checkpoint" ? "checkpoint 真值" : node.value_source || "未知";
  return (
    <section className="truth-section">
      <h4>
        参数真值 <span className={`badge ${node.value_source === "checkpoint" ? "truth" : ""}`}>{sourceLabel}</span>
      </h4>
      {node.params != null && <div className="truth-row"><b>参数量</b>{formatCount(node.params)}</div>}
      {node.dtype && <div className="truth-row"><b>dtype</b>{node.dtype}</div>}
      {node.weight_shapes && Object.keys(node.weight_shapes).length > 0 && (
        <div className="truth-row">
          <b>weight shapes</b>
          <code>
            {Object.entries(node.weight_shapes)
              .map(([name, shape]) => `${name} ${JSON.stringify(shape)}`)
              .join(", ")}
          </code>
        </div>
      )}
    </section>
  );
}

function NodeDetailPanel({ node, onClose }) {
  if (!node) return null;
  const confidence = typeof node.confidence === "number" ? node.confidence.toFixed(2) : null;
  const className = node.attributes?.class;
  return (
    <aside className="detail-panel">
      <header>
        <div>
          <h3 title={node.name}>{node.name}</h3>
          <div className="detail-badges">
            <span className="badge type">{node.type}</span>
            {className && <span className="badge class">{className}</span>}
            {node.repeat && <span className="badge repeat">×{node.repeat}</span>}
            {confidence && <span className="badge confidence">conf {confidence}</span>}
            {node.children?.length > 0 && (
              <span className="badge children">{node.children.length} children</span>
            )}
          </div>
        </div>
        <button className="close" onClick={onClose} aria-label="Close detail panel">
          ×
        </button>
      </header>
      <TruthSection node={node} />
      <ShapeFlow attributes={node.attributes} />
      <AttributeGrid attributes={node.attributes} sourceFields={node.source_fields} limit={null} />
    </aside>
  );
}

export default NodeDetailPanel;
