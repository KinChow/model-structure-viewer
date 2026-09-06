import { useState } from "react";
import SummaryChips from "./SummaryChips";
import StructureSearchBox from "./StructureSearchBox";
import ArchitectureTab from "./ArchitectureTab";
import CostSummary from "./CostSummary";
import ExportTab from "./ExportTab";
import RawConfigTab from "./RawConfigTab";
import NodeDetailPanel from "./NodeDetailPanel";
import { structureStatus } from "../diagnostics";
import { normalizeConfig } from "../structure/config/normalize.js";
import { derivedWeightParameters } from "../cost/derivedWeights.js";
import { COMPARISON_MODE } from "../diagram/compare.js";
import { DEFAULT_COMPARE_PLAN, DEFAULT_LOADS, DEFAULT_NODES, DEFAULT_PLAN } from "../cost/defaults.js";
import { DEFAULT_EFFICIENCY } from "../cost/efficiency.js";
import { graphChildren, graphNodeAt, graphViewNode } from "../structure/graph/selectors.js";

function breadcrumbForPath(graph, path, legacyRoot = null) {
  if (!path) return [];
  if (graph?.nodes) {
    const items = [];
    let current = graphNodeAt(graph, path);
    while (current) {
      items.unshift({ path: current.id, name: current.name });
      current = current.parent_id ? graphNodeAt(graph, current.parent_id) : null;
    }
    return items;
  }
  const root = legacyRoot;
  if (!root) return [];
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

function parameterTotalForStructure(structure) {
  const reported = structure?.summary?.parameters_total;
  if (reported != null) return { value: reported, derived: false };
  if (!structure?.extra_config) return { value: null, derived: false };
  const derived = derivedWeightParameters(normalizeConfig(structure.extra_config));
  return { value: derived > 0 ? derived : null, derived: derived > 0 };
}

function ModelSummaryPanel({ structure, sourceLabel, language, onSelectPath, parameterTotal }) {
  const summary = structure?.summary || {};
  const status = structureStatus(structure);
  const english = language === "en";
  const rows = [
    [english ? "Architecture" : "架构", summary.architecture],
    [english ? "Parameters" : "参数量", parameterTotal.value != null ? `${(parameterTotal.value / 1e9).toFixed(2)}B${parameterTotal.derived ? " · derived" : ""}` : "-"],
    [english ? "Layers" : "层数", summary.text_layers],
    ["Hidden Size", summary.hidden_size],
    ["Experts", summary.num_local_experts ?? summary.n_routed_experts],
    ["Context", summary.max_position_embeddings],
    [english ? "Source" : "来源", sourceLabel],
    [english ? "Status" : "状态", status.label],
  ];
  const topLevel = structure?.graph?.nodes
    ? graphChildren(structure.graph, structure.graph.root_id || "root").map((node) => graphViewNode(structure.graph, node.id))
    : (structure?.root?.children || []);
  return <div className="model-inspector-summary"><span className="inspector-kicker">{english ? "MODEL SUMMARY" : "模型摘要"}</span><h2>{summary.model_family || summary.model_type || "Model"}</h2><dl className="model-summary-grid">{rows.map(([label, value]) => <span key={label}><dt>{label}</dt><dd>{value ?? "-"}</dd></span>)}</dl><p className="model-summary-status" title={status.detail}>{status.detail}</p><section className="summary-module-section"><h3>{english ? "Top-level modules" : "顶层模块"}</h3><div className="summary-module-list">{topLevel.map((node, index) => <button type="button" key={node.path || node.id} onClick={() => onSelectPath?.(structure?.graph?.nodes ? node.path : `root.${index}`)}><span className="summary-module-kind">{node.type}</span><strong>{node.name}</strong>{node.repeat > 1 && <b>×{node.repeat}</b>}<span className="summary-module-arrow">→</span></button>)}</div></section><p>{english ? "Select a module or structure node to inspect details." : "选择模块或结构节点查看详情。"}</p><div className="inspector-rule" /></div>;
}

function DetailHeader({ structure, sourceLabel, language, onLanguageChange, onThemeChange, theme, onBack, onSettings }) {
  const id = structure?.source?.model_id || structure?.summary?.model_family || structure?.summary?.model_type || "model";
  const english = language === "en";
  const themeAction = theme === "dark"
    ? (english ? "Switch to light theme" : "切换到浅色主题")
    : (english ? "Switch to dark theme" : "切换到深色主题");
  return (
    <header className="detail-header">
      <button className="detail-brand" type="button" onClick={onBack}>Model Structure Viewer<span>.</span></button>
      <div className="detail-model-id" title={id}>{id}</div>
      <div className="detail-header-actions">
        <button type="button" onClick={() => onLanguageChange(english ? "zh" : "en")}>{english ? "EN / 中" : "中 / EN"}</button>
        <button type="button" title={themeAction} aria-label={themeAction} onClick={onThemeChange}>{theme === "dark" ? (english ? "Dark" : "深色") : (english ? "Light" : "浅色")}</button>
        <button type="button" onClick={onSettings}>{english ? "Settings" : "设置"}</button>
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
  loading = false,
  loadingPhase = "reading",
}) {
  const [auxView, setAuxView] = useState(null);
  const [costOpen, setCostOpen] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [activeLenses, setActiveLenses] = useState(() => new Set(["vram"]));
  const [activePhase, setActivePhase] = useState("prefill");
  const [activeMode, setActiveMode] = useState("centralized");
  const [activePlans, setActivePlans] = useState(() => ({ prefill: DEFAULT_PLAN, decode: DEFAULT_PLAN }));
  const [activeNodes, setActiveNodes] = useState(DEFAULT_NODES);
  const [activeGpusPerNode, setActiveGpusPerNode] = useState(8);
  const [activeMachineId, setActiveMachineId] = useState(chips?.[0]?.id || "");
  const [activeLoads, setActiveLoads] = useState(DEFAULT_LOADS);
  const [comparisonMode, setComparisonMode] = useState(COMPARISON_MODE.OFF);
  const [compareChipId, setCompareChipId] = useState(chips?.[1]?.id || chips?.[0]?.id || "");
  const [comparePlan, setComparePlan] = useState(DEFAULT_COMPARE_PLAN);
  const [efficiency, setEfficiency] = useState(DEFAULT_EFFICIENCY);
  const [nodeLens, setNodeLens] = useState({});
  const [costFitStatus, setCostFitStatus] = useState(null);
  const t = language === "en" ? { export: "Export", raw: "Raw config", cost: "Cost & placement", fit: "fit", notFit: "not fit", unknown: "unknown" } : { export: "导出", raw: "原始配置", cost: "成本与部署", fit: "已适配", notFit: "不适配", unknown: "未知" };
  const rawJson = structure?.extra_config ? JSON.stringify(structure.extra_config, null, 2) : "";
  const selectedData = selectedNode?.node || selectedNode;
  const selectedPath = selectedNodePath || selectedNode?.path || null;
  const parameterTotal = parameterTotalForStructure(structure);
  const breadcrumbs = breadcrumbForPath(structure?.graph, selectedPath, structure?.root);
  const activeMachineName = chips?.find((chip) => chip.id === activeMachineId)?.name || "GPU";
  const activeNodeCount = activeNodes?.[activeMode === "pd" ? activePhase : "centralized"] || 1;
  const fitLabel = costFitStatus == null ? null : activeMode === "pd" && costFitStatus.phaseFits
    ? `P ${costFitStatus.phaseFits.prefill == null ? t.unknown : costFitStatus.phaseFits.prefill ? t.fit : t.notFit} · D ${costFitStatus.phaseFits.decode == null ? t.unknown : costFitStatus.phaseFits.decode ? t.fit : t.notFit}`
    : costFitStatus.known ? (costFitStatus.fit ? t.fit : t.notFit) : t.unknown;
  const fitClass = costFitStatus?.phaseFits
    ? costFitStatus.phaseFits.prefill === false || costFitStatus.phaseFits.decode === false ? "no-fit" : costFitStatus.phaseFits.prefill === true && costFitStatus.phaseFits.decode === true ? "fit" : "unknown"
    : costFitStatus?.fit ? "fit" : costFitStatus?.known ? "no-fit" : "unknown";
  const deploymentSummary = activeMode === "pd"
    ? `PD · ${activeMachineName} · P ${activeNodes.prefill || 1}×${activeGpusPerNode} GPU · D ${activeNodes.decode || 1}×${activeGpusPerNode} GPU · ${activePhase}`
    : `${activeMachineName} · ${activeNodeCount}×${activeGpusPerNode} GPU`;
  const changeActivePhase = (next) => { setCostFitStatus(null); setActivePhase(next); };
  const changeActiveMode = (next) => { setCostFitStatus(null); setActiveMode(next); };
  const selectSearchResult = (path) => {
    setInspectorCollapsed(false);
    onSelectNode(path);
    onSearchChange("");
  };
  return (
    <main className={`detail-page theme-${theme}`}>
      <DetailHeader structure={structure} sourceLabel={sourceLabel} language={language} onLanguageChange={onLanguageChange} onThemeChange={onThemeChange} theme={theme} onBack={onBack} onSettings={onSettings} />
      <section className="detail-summary"><SummaryChips structure={structure} sourceLabel={sourceLabel} language={language} /></section>
      <section className="detail-layout">
        <div className="detail-main">
          <div className="detail-search-row"><StructureSearchBox value={searchTerm} onChange={onSearchChange} hitCount={matchedPaths.size} results={matchResults} onSelect={selectSearchResult} language={language} /><div className="detail-aux-actions"><button type="button" className={auxView === "export" ? "active" : ""} onClick={() => setAuxView(auxView === "export" ? null : "export")}>{t.export}</button><button type="button" className={auxView === "raw" ? "active" : ""} onClick={() => setAuxView(auxView === "raw" ? null : "raw")}>{t.raw}</button></div></div>
          <ArchitectureTab structure={structure} zoom={zoom} onZoomChange={onZoomChange} fitNonce={fitNonce} onFit={onFit} selectedPath={selectedPath} matchedPaths={matchedPaths} expandedGroups={expandedGroups} searchActive={Boolean(searchTerm.trim())} hitCount={matchedPaths.size} onSelectNode={onSelectNode} onToggleGroup={onToggleLayerPath} onExpandAllGroups={onExpandAllLayers} onCollapseAllGroups={onCollapseAllLayers} chips={chips} onAddChip={onAddChip} language={language} activeLenses={activeLenses} activePhase={activePhase} onPhaseChange={changeActivePhase} activeMode={activeMode} activePlans={activePlans} onPlanChange={setActivePlans} activeNodes={activeNodes} gpusPerNode={activeGpusPerNode} activeMachineId={activeMachineId} onMachineChange={setActiveMachineId} activeLoads={activeLoads} onNodeLensChange={setNodeLens} comparisonMode={comparisonMode} onComparisonModeChange={setComparisonMode} compareChipId={compareChipId} onCompareChipIdChange={setCompareChipId} comparePlan={comparePlan} onComparePlanChange={setComparePlan} efficiency={efficiency} onEfficiencyChange={setEfficiency} compactControls />
          <div className="detail-cost-toggle"><button type="button" onClick={() => setCostOpen((value) => !value)} aria-expanded={costOpen} aria-controls="detail-cost-panel"><span>{t.cost}</span><span className="detail-cost-summary">{activeMode === "pd" ? deploymentSummary : `Centralized · ${deploymentSummary}`}</span>{fitLabel && <span className={`detail-fit-status ${fitClass}`}>{fitLabel}</span>}<span>{costOpen ? "−" : "+"}</span></button></div>
          <div id="detail-cost-panel" className={`detail-cost-panel${costOpen ? "" : " is-collapsed"}`} aria-hidden={!costOpen}><CostSummary structure={structure} chips={chips} onAddChip={onAddChip} language={language} onFitStatusChange={setCostFitStatus} lenses={activeLenses} onLensesChange={setActiveLenses} phase={activePhase} onPhaseChange={changeActivePhase} mode={activeMode} onModeChange={changeActiveMode} plans={activePlans} onPlansChange={setActivePlans} nodes={activeNodes} onNodesChange={setActiveNodes} gpusPerNode={activeGpusPerNode} onGpusPerNodeChange={setActiveGpusPerNode} machineId={activeMachineId} onMachineIdChange={setActiveMachineId} loads={activeLoads} onLoadsChange={setActiveLoads} comparisonMode={comparisonMode} onComparisonModeChange={setComparisonMode} compareChipId={compareChipId} onCompareChipIdChange={setCompareChipId} comparePlan={comparePlan} onComparePlanChange={setComparePlan} efficiency={efficiency} onEfficiencyChange={setEfficiency} /></div>
          {auxView === "export" && <div className="detail-aux-panel"><ExportTab format={exporter.format} onFormatChange={exporter.setFormat} text={exporter.text} onRun={() => exporter.run(structure)} /></div>}
          {auxView === "raw" && <div className="detail-aux-panel"><RawConfigTab rawJson={rawJson} /></div>}
        </div>
        <div className="detail-inspector-slot">{selectedData ? <NodeDetailPanel node={selectedData} path={selectedPath} breadcrumbs={breadcrumbs} totalParameters={parameterTotal.value} costLens={nodeLens?.[selectedPath]} activeLenses={activeLenses} language={language} collapsed={inspectorCollapsed} onToggleCollapsed={() => setInspectorCollapsed((value) => !value)} onSelectPath={(path) => { setInspectorCollapsed(false); onSelectNode(path); }} onClose={() => { setInspectorCollapsed(false); onCloseNode(); }} /> : <ModelSummaryPanel structure={structure} sourceLabel={sourceLabel} language={language} parameterTotal={parameterTotal} onSelectPath={(path) => { setInspectorCollapsed(false); onSelectNode(path); }} />}</div>
      </section>
      {loading && <div className="detail-loading-overlay" role="status" aria-live="polite"><span className="detail-loading-dot" /><span>{({ reading: language === "en" ? "Reading model files" : "读取模型配置", building: language === "en" ? "Building model structure" : "构建模型结构", metadata: language === "en" ? "Checking weight metadata" : "检查权重元数据" })[loadingPhase] || (language === "en" ? "Opening model..." : "正在打开模型...")}</span><i aria-hidden="true">...</i></div>}
    </main>
  );
}
