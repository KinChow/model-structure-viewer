import { useMemo, useState } from "react";
import frontendPackage from "../../package.json";
import SummaryChips from "./SummaryChips";
import StructureSearchBox from "./StructureSearchBox";
import ArchitectureTab from "./ArchitectureTab";
import CostSummary from "./CostSummary";
import DiagnosticsPanel from "./DiagnosticsPanel";
import ExportTab from "./ExportTab";
import RawConfigTab from "./RawConfigTab";
import NodeDetailPanel from "./NodeDetailPanel";
import { structureStatus } from "../diagnostics";
import { graphWeightCapacity } from "../cost/memory.js";
import { COMPARISON_MODE } from "../diagram/compare.js";
import { buildNodeLens } from "../diagram/lens.js";
import { DEFAULT_COMPARE_PLAN, DEFAULT_LOADS, DEFAULT_NODES, DEFAULT_PLAN } from "../cost/defaults.js";
import { DEFAULT_EFFICIENCY } from "../cost/efficiency.js";
import { graphChildren, graphNodeAt, graphViewNode } from "../structure/graph/selectors.js";

function breadcrumbForPath(graph, path) {
  if (!graph?.nodes || !path) return [];
  const items = [];
  let current = graphNodeAt(graph, path);
  while (current) {
    items.unshift({ path: current.id, name: current.name });
    current = current.parent_id ? graphNodeAt(graph, current.parent_id) : null;
  }
  return items;
}

function parameterTotalForStructure(structure) {
  const reported = structure?.summary?.parameters_total;
  if (reported != null) return { value: reported, derived: false };
  if (!structure?.graph) return { value: null, derived: false };
  const derived = graphWeightCapacity(structure.graph).elements;
  return { value: derived > 0 ? derived : null, derived: derived > 0 };
}

function ModelSummaryPanel({ structure, sourceLabel, language, onSelectPath, parameterTotal }) {
  const summary = structure?.summary || {};
  const status = structureStatus(structure, language);
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
    : [];
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
      <button className="detail-brand" type="button" onClick={onBack}>Model Structure Viewer <span className="detail-version">v{frontendPackage.version}</span></button>
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
  selectedNodePath,
  onSelectNode,
  onCloseNode,
  searchTerm,
  onSearchChange,
  matchResults,
  matchedPaths,
  expandedGroups,
  zoom,
  fitNonce,
  onFit,
  chips,
  onAddChip,
  onToggleLayerPath,
  onExpandAllLayers,
  onCollapseAllLayers,
  exporter,
  loading = false,
  loadingPhase = "reading",
  onVerify,
  verifyResult = null,
  verifyLoading = false,
  verifyError = "",
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
  const [costFitStatus, setCostFitStatus] = useState(null);
  // M10-E：nodeLens 派生上移——原先是 ArchitectureTab useMemo 计算后经 onNodeLensChange effect 回写本组件 state 的双份状态，
  // 现由输入 state 的归属方（本组件）直接派生，NodeDetailPanel 与 ArchitectureTab 共享同一份结果。
  const lensPhase = activePhase ?? "prefill";
  const lensChip = chips.find((entry) => entry.id === (activeMachineId ?? "")) || chips[0];
  const lensPlan = activePlans?.[lensPhase] || DEFAULT_PLAN;
  const lensLoad = activeLoads?.[lensPhase] || { batch: 1, sequence: 2048 };
  const nodeLensResult = useMemo(
    () => buildNodeLens(structure, lensChip, { phase: lensPhase, batch: lensLoad.batch, sequence: lensLoad.sequence, plan: lensPlan, efficiency: efficiency || DEFAULT_EFFICIENCY }),
    [structure, lensChip, lensPhase, lensLoad, lensPlan, efficiency],
  );
  const nodeLens = nodeLensResult.nodes;
  const t = language === "en" ? { export: "Export", raw: "Raw config", cost: "Cost & placement", fit: "fit", notFit: "not fit", unknown: "unknown" } : { export: "导出", raw: "原始配置", cost: "成本与部署", fit: "已适配", notFit: "不适配", unknown: "未知" };
  const rawJson = structure?.extra_config ? JSON.stringify(structure.extra_config, null, 2) : "";
  // M10-E：selectedNode 改为本地派生（原先 App 派生后与 selectedNodePath 成对透传，语义重复）。
  const selectedNode = selectedNodePath && structure?.graph
    ? { node: structure.graph.nodes ? graphViewNode(structure.graph, selectedNodePath) : null, path: selectedNodePath }
    : null;
  const selectedData = selectedNode?.node || selectedNode;
  const selectedPath = selectedNodePath || selectedNode?.path || null;
  const parameterTotal = parameterTotalForStructure(structure);
  const breadcrumbs = breadcrumbForPath(structure?.graph, selectedPath);
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
  };
  return (
    <main className={`detail-page theme-${theme}`}>
      <DetailHeader structure={structure} sourceLabel={sourceLabel} language={language} onLanguageChange={onLanguageChange} onThemeChange={onThemeChange} theme={theme} onBack={onBack} onSettings={onSettings} />
      <section className="detail-summary"><SummaryChips structure={structure} sourceLabel={sourceLabel} language={language} /></section>
      <DiagnosticsPanel structure={structure} language={language} onVerify={onVerify} verifyResult={verifyResult} verifyLoading={verifyLoading} verifyError={verifyError} />
      <section className="detail-layout">
        <div className="detail-main">
          <div className="detail-search-row"><StructureSearchBox value={searchTerm} onChange={onSearchChange} hitCount={matchedPaths.size} results={matchResults} onSelect={selectSearchResult} language={language} /><div className="detail-aux-actions"><button type="button" className={auxView === "export" ? "active" : ""} onClick={() => setAuxView(auxView === "export" ? null : "export")}>{t.export}</button><button type="button" className={auxView === "raw" ? "active" : ""} onClick={() => setAuxView(auxView === "raw" ? null : "raw")}>{t.raw}</button></div></div>
          <ArchitectureTab
            structure={structure}
            language={language}
            chips={chips}
            onAddChip={onAddChip}
            compactControls
            diagram={{ zoom, fitNonce, onFit, selectedPath, matchedPaths, expandedGroups, searchActive: Boolean(searchTerm.trim()), onSelectNode, onToggleGroup: onToggleLayerPath, onExpandAllGroups: onExpandAllLayers, onCollapseAllGroups: onCollapseAllLayers }}
            cost={{ activeLenses, activePhase, onPhaseChange: changeActivePhase, activeMode, activePlans, onPlanChange: setActivePlans, activeNodes, gpusPerNode: activeGpusPerNode, activeMachineId, onMachineChange: setActiveMachineId, activeLoads, nodeLensResult, comparisonMode, onComparisonModeChange: setComparisonMode, compareChipId, onCompareChipIdChange: setCompareChipId, comparePlan, onComparePlanChange: setComparePlan, efficiency, onEfficiencyChange: setEfficiency }}
          />
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
