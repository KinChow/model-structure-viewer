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
  if (!hasTruth) {
    // 自解释：无真值 ≠ 漏绑，按节点性质说明原因
    let reason = "无独立权重";
    if (node.type === "operator") reason = "无参数算子（不占用权重）";
    else if (node.children?.length > 0) reason = "容器节点（参数归集在子节点）";
    return (
      <section className="truth-section muted">
        <h4>参数真值</h4>
        <div className="truth-row">{reason}</div>
      </section>
    );
  }
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

function FormulaSection({ node }) {
  const formula = node.attributes?.formula;
  const formulaId = node.attributes?.formula_id;
  if (!formulaId && !formula) return null;
  return <section className="formula-section"><h4>公式 <span className="badge class">{formulaId || "operator"}</span></h4>{formula && <code>{formula}</code>}{node.attributes?.explanation && <p>{node.attributes.explanation}</p>}</section>;
}

function NodeDetailPanel({ node, path, breadcrumbs = [], totalParameters, onSelectPath, onClose }) {
  if (!node) return null;
  const confidence = typeof node.confidence === "number" ? node.confidence.toFixed(2) : null;
  const className = node.attributes?.class;
  const parameterShare = Number.isFinite(node.params) && Number.isFinite(totalParameters) && totalParameters > 0
    ? Math.min(100, Math.max(0, (node.params / totalParameters) * 100))
    : null;
  return (
    <aside className="detail-panel">
      {path && <div className="detail-breadcrumb" aria-label="Structure path">{(breadcrumbs.length > 0 ? breadcrumbs : path.split(".").map((part, index, parts) => ({ path: parts.slice(0, index + 1).join("."), name: part === "root" ? "model" : `#${part}` }))).map((item, index, items) => <span key={item.path}><button type="button" className={index === items.length - 1 ? "current" : ""} onClick={() => index < items.length - 1 && onSelectPath?.(item.path)}>{item.name}</button>{index < items.length - 1 && <i>/</i>}</span>)}</div>}
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
      {parameterShare != null && <div className="inspector-parameter-share" title={`${parameterShare.toFixed(2)}% of model parameters`}><div className="inspector-parameter-track"><span style={{ width: `${Math.max(parameterShare, 0.5)}%` }} /></div><small>{parameterShare.toFixed(2)}% of model parameters</small></div>}
      <FormulaSection node={node} />
      <ShapeFlow attributes={node.attributes} />
      <AttributeGrid attributes={node.attributes} sourceFields={node.source_fields} limit={null} />
    </aside>
  );
}

export default NodeDetailPanel;
