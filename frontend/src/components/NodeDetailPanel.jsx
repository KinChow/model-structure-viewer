import AttributeGrid from "./AttributeGrid";

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
      <AttributeGrid attributes={node.attributes} sourceFields={node.source_fields} limit={null} />
    </aside>
  );
}

export default NodeDetailPanel;
