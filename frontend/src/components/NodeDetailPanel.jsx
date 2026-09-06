import { useState } from "react";
import AttributeGrid from "./AttributeGrid";
import ShapeFlow from "./ShapeFlow";
import { formatBytes, formatCount, formatQuantity, formatSeconds } from "../formatters.js";

function TruthSection({ node, language = "zh" }) {
  const english = language === "en";
  const hasTruth =
    node.params != null ||
    node.dtype ||
    (node.weight_shapes && Object.keys(node.weight_shapes).length > 0);
  if (!hasTruth) {
    // 自解释：无真值 ≠ 漏绑，按节点性质说明原因
    let reason = english ? "No independent weights" : "无独立权重";
    if (node.type === "operator") reason = english ? "Parameter-free operator" : "无参数算子（不占用权重）";
    else if (node.children?.length > 0) reason = english ? "Container node; parameters are in children" : "容器节点（参数归集在子节点）";
    return (
      <section className="truth-section muted">
        <h4>{english ? "Parameter truth" : "参数真值"}</h4>
        <div className="truth-row">{reason}</div>
      </section>
    );
  }
  const sourceLabel = node.value_source === "checkpoint" ? (english ? "checkpoint truth" : "checkpoint 真值") : node.value_source || (english ? "unknown" : "未知");
  return (
    <section className="truth-section">
      <h4>
        {english ? "Parameter truth" : "参数真值"} <span className={`badge ${node.value_source === "checkpoint" ? "truth" : ""}`}>{sourceLabel}</span>
      </h4>
      {node.params != null && <div className="truth-row"><b>{english ? "Parameters" : "参数量"}</b>{formatCount(node.params, { largeDigits: 2 })}</div>}
      {node.dtype && <div className="truth-row"><b>dtype</b>{node.dtype}</div>}
      {node.weight_shapes && Object.keys(node.weight_shapes).length > 0 && (
        <div className="truth-row">
          <b>{english ? "weight shapes" : "权重 Shape"}</b>
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

function FormulaSection({ node, language = "zh" }) {
  const formula = node.attributes?.formula;
  const formulaId = node.attributes?.formula_id;
  if (!formulaId && !formula) return null;
  return <section className="formula-section"><h4>{language === "en" ? "Formula" : "公式"} <span className="badge class">{formulaId || "operator"}</span></h4>{formula && <code>{formula}</code>}{node.attributes?.explanation && <p>{node.attributes.explanation}</p>}</section>;
}

function LensSection({ lens, activeLenses = new Set(), language = "zh" }) {
  if (!lens || activeLenses.size === 0) return null;
  const aggregateOnly = activeLenses.has("vram") || activeLenses.has("kv");
  return <section className="node-lens-section"><h4>Cost Lens <span className={`badge ${lens.bound === "unknown" ? "" : "truth"}`}>{lens.bound}</span></h4>{activeLenses.has("vram") && <div className="truth-row"><b>VRAM</b>{formatBytes(lens.metrics?.vramBytes)}</div>}{activeLenses.has("compute") && <><div className="truth-row"><b>{language === "en" ? "MACs / forward" : "MACs / forward"}</b>{formatQuantity(lens.metrics?.macs)}</div><div className="truth-row"><b>FLOPs / forward</b>{formatQuantity(lens.metrics?.flops)}</div><div className="truth-row muted"><b>{language === "en" ? "Roofline compute" : "Roofline 计算"}</b>{formatSeconds(lens.metrics?.computeSeconds)}</div></>}{activeLenses.has("memory") && <div className="truth-row"><b>Memory</b>{formatBytes(lens.metrics?.memoryBytes)}</div>}{activeLenses.has("compute") && <div className="truth-row"><b>{language === "en" ? "Communication" : "通信"}</b>{formatSeconds(lens.metrics?.communicationSeconds)}</div>}{activeLenses.has("kv") && <div className="truth-row muted"><b>KV Cache</b>{language === "en" ? "See cost panel aggregate" : "见成本面板汇总"}</div>}{aggregateOnly && !activeLenses.has("vram") && !activeLenses.has("kv") && <div className="truth-row muted">{language === "en" ? "No node-level aggregate" : "无节点级汇总"}</div>}</section>;
}

function ChildModulesSection({ node, path, language = "zh", onSelectPath }) {
  if (!node.children?.length) return null;
  return <section className="child-modules-section"><h4>{language === "en" ? "Child modules" : "子模块"}</h4><div className="child-module-list">{node.children.map((child, index) => <button type="button" key={`${child.id || child.name}-${index}`} onClick={() => onSelectPath?.(`${path}.${index}`)}><span className="child-module-kind">{child.type}</span><strong>{child.name}</strong>{child.repeat > 1 && <b>×{child.repeat}</b>}<span className="child-module-arrow">→</span></button>)}</div></section>;
}

function NodeDetailPanel({ node, path, breadcrumbs = [], totalParameters, costLens, activeLenses = new Set(), language = "zh", collapsed = false, onToggleCollapsed, onSelectPath, onClose }) {
  const [copyState, setCopyState] = useState("idle");
  if (!node) return null;
  const confidence = typeof node.confidence === "number" ? node.confidence.toFixed(2) : null;
  const className = node.attributes?.class;
  const parameterShare = Number.isFinite(node.params) && Number.isFinite(totalParameters) && totalParameters > 0
    ? Math.min(100, Math.max(0, (node.params / totalParameters) * 100))
    : null;
  const english = language === "en";
  const hasShape = Boolean(node.attributes?.input_shape || node.attributes?.output_shape);
  const hasAttributes = Object.entries(node.attributes || {}).some(([key]) => key !== "class") || (node.source_fields?.length > 0);
  async function copyPath() {
    if (!path) return;
    try {
      if (!navigator.clipboard?.writeText) throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(path);
      setCopyState("success");
    } catch {
      setCopyState("unavailable");
    }
  }
  return (
    <aside className={`detail-panel${collapsed ? " mobile-inspector-collapsed" : ""}`}>
      <button type="button" className="detail-sheet-handle" aria-label={collapsed ? (english ? "Expand inspector" : "展开详情") : (english ? "Collapse inspector" : "收起详情")} aria-expanded={!collapsed} onClick={onToggleCollapsed} />
      {path && <div className="detail-breadcrumb" aria-label="Structure path"><div className="detail-breadcrumb-path">{(breadcrumbs.length > 0 ? breadcrumbs : path.split(".").map((part, index, parts) => ({ path: parts.slice(0, index + 1).join("."), name: part === "root" ? "model" : `#${part}` }))).map((item, index, items) => <span key={item.path}><button type="button" className={index === items.length - 1 ? "current" : ""} onClick={() => index < items.length - 1 && onSelectPath?.(item.path)}>{item.name}</button>{index < items.length - 1 && <i>/</i>}</span>)}</div><button type="button" className={`copy-path ${copyState}`} onClick={copyPath}>{copyState === "success" ? (language === "en" ? "Copied" : "已复制") : copyState === "unavailable" ? (language === "en" ? "Copy unavailable" : "无法复制") : language === "en" ? "Copy path" : "复制路径"}</button></div>}
      <header>
        <div>
          <h3 title={node.name}>{node.name}</h3>
          <div className="detail-badges">
            <span className="badge type">{node.type}</span>
            {className && <span className="badge class">{className}</span>}
            {node.repeat && <span className="badge repeat">×{node.repeat}</span>}
            {confidence && <span className="badge confidence">conf {confidence}</span>}
            {node.children?.length > 0 && (
              <span className="badge children">{node.children.length} {english ? "children" : "个子节点"}</span>
            )}
          </div>
        </div>
        <button className="close" onClick={onClose} aria-label={english ? "Close detail panel" : "关闭详情面板"}>
          ×
        </button>
      </header>
      <TruthSection node={node} language={language} />
      {parameterShare != null && <div className="inspector-parameter-share" title={`${parameterShare.toFixed(2)}% ${english ? "of model parameters" : "模型参数占比"}`}><div className="inspector-parameter-track"><span style={{ width: `${Math.max(parameterShare, 0.5)}%` }} /></div><small>{parameterShare.toFixed(2)}% {english ? "of model parameters" : "模型参数占比"}</small></div>}
      <LensSection lens={costLens} activeLenses={activeLenses} language={language} />
      <FormulaSection node={node} language={language} />
      <ChildModulesSection node={node} path={path} language={language} onSelectPath={onSelectPath} />
      {hasShape && <details className="inspector-disclosure" open>
        <summary>{english ? "Shape / tensor flow" : "Shape / Tensor 流"}</summary>
        <ShapeFlow attributes={node.attributes} />
      </details>}
      {hasAttributes && <details className="inspector-disclosure" open>
        <summary>{english ? "Attributes" : "属性"}</summary>
        <AttributeGrid attributes={node.attributes} sourceFields={node.source_fields} limit={null} />
      </details>}
    </aside>
  );
}

export default NodeDetailPanel;
