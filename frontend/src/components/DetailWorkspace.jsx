import { useState } from "react";
import SummaryChips from "./SummaryChips";
import StructureSearchBox from "./StructureSearchBox";
import ArchitectureTab from "./ArchitectureTab";
import CostSummary from "./CostSummary";
import ExportTab from "./ExportTab";
import RawConfigTab from "./RawConfigTab";
import NodeDetailPanel from "./NodeDetailPanel";

function breadcrumbForPath(root, path) {
  if (!root || !path) return [];
  const parts = path.split(".");
  const items = [{ path: "root", name: root.name }];
  let current = root;
  for (let index = 1; index < parts.length; index += 1) {
    const child = current.children?.[Number(parts[index])];
    if (!child) break;
    current = child;
    items.push({ path: parts.slice(0, index + 1).join("."), name: child.name });
  }
  return items;
}

function ModelSummaryPanel({ structure, sourceLabel, language, onSelectPath }) {
  const summary = structure?.summary || {};
  const english = language === "en";
  const rows = [
    [english ? "Architecture" : "架构", summary.architecture],
    [english ? "Parameters" : "参数量", summary.parameters_total ? `${(summary.parameters_total / 1e9).toFixed(2)}B` : "-"],
    [english ? "Layers" : "层数", summary.text_layers],
    ["Hidden Size", summary.hidden_size],
    ["Experts", summary.num_local_experts ?? summary.n_routed_experts],
    ["Context", summary.max_position_embeddings],
    [english ? "Source" : "来源", sourceLabel],
    [english ? "Status" : "状态", summary.strategy || "-"],
  ];
  return <div className="model-inspector-summary"><span className="inspector-kicker">MODEL SUMMARY</span><h2>{summary.model_family || summary.model_type || "Model"}</h2><dl className="model-summary-grid">{rows.map(([label, value]) => <span key={label}><dt>{label}</dt><dd>{value ?? "-"}</dd></span>)}</dl><section className="summary-module-section"><h3>{english ? "Top-level modules" : "顶层模块"}</h3><div className="summary-module-list">{(structure?.root?.children || []).map((node, index) => <button type="button" key={`${node.id}-${index}`} onClick={() => onSelectPath?.(`root.${index}`)}><span className="summary-module-kind">{node.type}</span><strong>{node.name}</strong>{node.repeat > 1 && <b>×{node.repeat}</b>}<span className="summary-module-arrow">→</span></button>)}</div></section><p>{english ? "Select a module or structure node to inspect details." : "选择模块或结构节点查看详情。"}</p><div className="inspector-rule" /></div>;
}

function DetailHeader({ structure, sourceLabel, language, onLanguageChange, onThemeChange, theme, onBack, onSettings }) {
  const id = structure?.source?.model_id || structure?.summary?.model_family || structure?.summary?.model_type || "model";
  const english = language === "en";
  return (
    <header className="detail-header">
      <button className="detail-brand" type="button" onClick={onBack}>Model Structure Viewer<span>.</span></button>
      <div className="detail-model-id" title={id}>{id}</div>
      <div className="detail-header-actions">
        <button type="button" onClick={() => onLanguageChange(english ? "zh" : "en")}>{english ? "EN / 中" : "中 / EN"}</button>
        <button type="button" onClick={onThemeChange}>{theme}</button>
        <button type="button" onClick={onSettings}>设置 / Settings</button>
      </div>
    </header>
  );
}

export default function DetailWorkspace({
  structure,
  sourceLabel,
  language,
  theme,
  onLanguageChange,
  onThemeChange,
  onBack,
  onSettings,
  selectedNode,
  selectedNodePath,
  onSelectNode,
  onCloseNode,
  searchTerm,
  onSearchChange,
  matchResults,
  matchedPaths,
  expandedGroups,
  zoom,
  onZoomChange,
  fitNonce,
  onFit,
  chips,
  onAddChip,
  allCollapsiblePaths,
  layersExpandedPaths,
  onToggleLayerPath,
  onExpandAllLayers,
  onCollapseAllLayers,
  exporter,
}) {
  const [auxView, setAuxView] = useState(null);
  const [costOpen, setCostOpen] = useState(false);
  const [activeLenses, setActiveLenses] = useState(() => new Set(["vram"]));
  const [activePhase, setActivePhase] = useState("prefill");
  const [activeMode, setActiveMode] = useState("centralized");
  const [activePlans, setActivePlans] = useState({ prefill: { tp: 1, pp: 1, ep: 1, dp: 1, attnMode: "tp" }, decode: { tp: 1, pp: 1, ep: 1, dp: 1, attnMode: "tp" } });
  const [activeNodes, setActiveNodes] = useState({ centralized: 1, prefill: 1, decode: 2 });
  const [activeGpusPerNode, setActiveGpusPerNode] = useState(8);
  const [activeMachineId, setActiveMachineId] = useState(chips?.[0]?.id || "");
  const [nodeLens, setNodeLens] = useState({});
  const t = language === "en" ? { export: "Export", raw: "Raw config", cost: "Cost & placement" } : { export: "导出", raw: "原始配置", cost: "成本与部署" };
  const rawJson = structure?.extra_config ? JSON.stringify(structure.extra_config, null, 2) : "";
  const selectedData = selectedNode?.node || selectedNode;
  const selectedPath = selectedNodePath || selectedNode?.path || null;
  const breadcrumbs = breadcrumbForPath(structure?.root, selectedPath);
  const activeMachineName = chips?.find((chip) => chip.id === activeMachineId)?.name || "GPU";
  const activeNodeCount = activeNodes?.[activeMode === "pd" ? activePhase : "centralized"] || 1;
  return (
    <main className={`detail-page theme-${theme}`}>
      <DetailHeader structure={structure} sourceLabel={sourceLabel} language={language} onLanguageChange={onLanguageChange} onThemeChange={onThemeChange} theme={theme} onBack={onBack} onSettings={onSettings} />
      <section className="detail-summary"><SummaryChips structure={structure} sourceLabel={sourceLabel} language={language} /></section>
      <section className="detail-layout">
        <div className="detail-main">
          <div className="detail-search-row"><StructureSearchBox value={searchTerm} onChange={onSearchChange} hitCount={matchedPaths.size} results={matchResults} onSelect={onSelectNode} /><div className="detail-aux-actions"><button type="button" className={auxView === "export" ? "active" : ""} onClick={() => setAuxView(auxView === "export" ? null : "export")}>{t.export}</button><button type="button" className={auxView === "raw" ? "active" : ""} onClick={() => setAuxView(auxView === "raw" ? null : "raw")}>{t.raw}</button></div></div>
          <div className="detail-cost-toggle"><button type="button" onClick={() => setCostOpen((value) => !value)} aria-expanded={costOpen}><span>{t.cost}</span><span className="detail-cost-summary">{activeMode === "pd" ? "PD" : "Centralized"} · {activeMachineName} · {activePhase} · {activeNodeCount}×{activeGpusPerNode} GPU</span><span>{costOpen ? "−" : "+"}</span></button></div>
          {costOpen && <div className="detail-cost-panel"><CostSummary structure={structure} chips={chips} onAddChip={onAddChip} language={language} lenses={activeLenses} onLensesChange={setActiveLenses} phase={activePhase} onPhaseChange={setActivePhase} mode={activeMode} onModeChange={setActiveMode} plans={activePlans} onPlansChange={setActivePlans} nodes={activeNodes} onNodesChange={setActiveNodes} gpusPerNode={activeGpusPerNode} onGpusPerNodeChange={setActiveGpusPerNode} machineId={activeMachineId} onMachineIdChange={setActiveMachineId} /></div>}
          <ArchitectureTab structure={structure} zoom={zoom} onZoomChange={onZoomChange} fitNonce={fitNonce} onFit={onFit} selectedPath={selectedPath} matchedPaths={matchedPaths} expandedGroups={expandedGroups} searchActive={Boolean(searchTerm.trim())} hitCount={matchedPaths.size} onSelectNode={onSelectNode} onToggleGroup={onToggleLayerPath} onExpandAllGroups={onExpandAllLayers} onCollapseAllGroups={onCollapseAllLayers} chips={chips} onAddChip={onAddChip} language={language} activeLenses={activeLenses} activePhase={activePhase} activeMode={activeMode} activePlans={activePlans} onPlanChange={setActivePlans} activeNodes={activeNodes} gpusPerNode={activeGpusPerNode} activeMachineId={activeMachineId} onMachineChange={setActiveMachineId} onNodeLensChange={setNodeLens} compactControls />
          {auxView === "export" && <div className="detail-aux-panel"><ExportTab format={exporter.format} onFormatChange={exporter.setFormat} text={exporter.text} onRun={() => exporter.run(structure)} /></div>}
          {auxView === "raw" && <div className="detail-aux-panel"><RawConfigTab rawJson={rawJson} /></div>}
        </div>
        <div className="detail-inspector-slot">{selectedData ? <NodeDetailPanel node={selectedData} path={selectedPath} breadcrumbs={breadcrumbs} totalParameters={structure?.summary?.parameters_total} costLens={nodeLens?.[selectedPath]} activeLenses={activeLenses} language={language} onSelectPath={onSelectNode} onClose={onCloseNode} /> : <ModelSummaryPanel structure={structure} sourceLabel={sourceLabel} language={language} onSelectPath={onSelectNode} />}</div>
      </section>
    </main>
  );
}
