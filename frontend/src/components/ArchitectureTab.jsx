import { useEffect, useMemo, useRef, useState } from "react";
import StructureDiagram from "../diagram/ReactFlowStructureDiagram";
import EmptyState from "./EmptyState";
import { PUBLIC_CHIPS } from "../cost/chips/public.js";
import { collectFormulaLinks } from "../diagram/formulaLinks.js";
import { boundFlips, COMPARISON_MODE, resolveComparisonScenario } from "../diagram/compare.js";
import { buildNodeLens } from "../diagram/lens.js";
import ManualChipForm from "./ManualChipForm.jsx";
import { getChipCoverage } from "../cost/chips/coverage.js";

function downloadSvg(structure) {
  const legacySvg = document.querySelector(".diagram-svg");
  const flow = document.querySelector(".react-flow");
  const viewport = flow?.querySelector(".react-flow__viewport");
  if (!legacySvg && !viewport) return;
  const source = legacySvg
    ? legacySvg.outerHTML
    : (() => {
      const rect = flow.getBoundingClientRect();
      const styles = [...document.styleSheets].flatMap((sheet) => {
        try {
          return [...sheet.cssRules].map((rule) => rule.cssText);
        } catch {
          return [];
        }
      }).join("\n");
      return `<svg xmlns="http://www.w3.org/2000/svg" width="${rect.width}" height="${rect.height}" viewBox="0 0 ${rect.width} ${rect.height}"><style>${styles}</style><foreignObject width="100%" height="100%"><div xmlns="http://www.w3.org/1999/xhtml" style="width:${rect.width}px;height:${rect.height}px;overflow:hidden">${viewport.innerHTML}</div></foreignObject></svg>`;
    })();
  const blob = new Blob([source], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${structure?.summary?.model_family || "model"}-structure.svg`;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  requestAnimationFrame(() => {
    link.remove();
    URL.revokeObjectURL(url);
  });
}

function chipLinkText(chip, language = "zh") {
  const bandwidth = chip?.interconnect?.intra_node?.bandwidth;
  return Number.isFinite(bandwidth) ? `${bandwidth / 1e9} GB/s` : language === "en" ? "link unknown" : "链路未知";
}

function chipOptionText(chip) {
  return `${chip.name}${chip.confidence === "local" ? " (local)" : ""}`;
}

function scenarioLabel(prefix, scenario, language = "zh") {
  if (!scenario) return prefix;
  const { chip, plan } = scenario;
  return `${prefix} · ${chip?.name || (language === "en" ? "unknown GPU" : "未知芯片")} · TP ${plan.tp} / EP ${plan.ep} / Attention ${String(plan.attnMode).toUpperCase()} · ${chipLinkText(chip, language)}`;
}

function coverageWarningText(warning, english) {
  if (!english) return warning;
  if (warning.startsWith("缺少 interconnect.inter_node.bandwidth")) return "Missing interconnect.inter_node.bandwidth; cross-node uses intra-node bandwidth and may be optimistic";
  if (warning.startsWith("缺少规格来源")) return "Missing hardware specification source";
  if (warning.startsWith("未知 confidence")) return warning.replace("未知 confidence", "Unknown confidence");
  return warning;
}

function DiagramPane({ label, syncId, ...diagramProps }) {
  return (
    <div>
      <div className="diagram-compare-label">{label}</div>
      <StructureDiagram {...diagramProps} scrollSyncId={syncId} showGroupToggle={false} />
    </div>
  );
}

function ArchitectureTab({
  structure,
  zoom,
  onZoomChange,
  fitNonce,
  onFit,
  selectedPath,
  matchedPaths,
  expandedGroups,
  searchActive,
  hitCount,
  onSelectNode,
  onToggleGroup,
  onExpandAllGroups,
  onCollapseAllGroups,
  activeLenses = new Set(["vram"]),
  activePhase,
  onPhaseChange,
  activeMode = "centralized",
  activePlans,
  onPlanChange,
  activeNodes,
  gpusPerNode = 1,
  activeMachineId,
  onMachineChange,
  activeLoads,
  onNodeLensChange,
  chips = PUBLIC_CHIPS,
  onAddChip,
  language = "zh",
  compactControls = false,
}) {
  const english = language === "en";
  const [internalPhase, setInternalPhase] = useState("prefill");
  const phase = activePhase || internalPhase;
  const [internalChipId, setInternalChipId] = useState(chips[0]?.id || "");
  const chipId = activeMachineId || internalChipId;
  const [tp, setTp] = useState(1);
  const [ep, setEp] = useState(1);
  const [attnMode, setAttnMode] = useState("tp");
  const [comparisonMode, setComparisonMode] = useState(COMPARISON_MODE.OFF);
  const [compareChipId, setCompareChipId] = useState(chips[1]?.id || chips[0]?.id || "");
  const [compareTp, setCompareTp] = useState(2);
  const [compareEp, setCompareEp] = useState(1);
  const [compareAttnMode, setCompareAttnMode] = useState("tp");
  const [etaFlops, setEtaFlops] = useState(0.7);
  const [etaHbm, setEtaHbm] = useState(0.9);
  const [etaComm, setEtaComm] = useState(0.8);
  const [formulaHoveredPath, setFormulaHoveredPath] = useState(null);
  const [diagramHoveredPath, setDiagramHoveredPath] = useState(null);
  const [formulaOpen, setFormulaOpen] = useState(false);
  const [edgeMode, setEdgeMode] = useState("dataflow");
  const compareScrollGroup = useRef(new Map());
  const [advancedOpen, setAdvancedOpen] = useState(!compactControls);
  const [canvasFocus, setCanvasFocus] = useState(false);
  useEffect(() => {
    if (!canvasFocus) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previousOverflow; };
  }, [canvasFocus]);
  const changePhase = (next) => activePhase ? onPhaseChange?.(next) : setInternalPhase(next);
  const internalPlan = useMemo(() => ({ tp, ep, attnMode }), [tp, ep, attnMode]);
  const plan = activePlans?.[phase] || internalPlan;
  const updatePlan = (next) => {
    if (activePlans) onPlanChange?.({ ...activePlans, [phase]: next });
    else {
      setTp(next.tp);
      setEp(next.ep);
      setAttnMode(next.attnMode);
    }
  };
  const changeChip = (next) => activeMachineId ? onMachineChange?.(next) : setInternalChipId(next);
  const formulaLinks = useMemo(() => collectFormulaLinks(structure?.root), [structure]);
  const chip = chips.find((entry) => entry.id === chipId) || chips[0];
  const candidateChip = chips.find((entry) => entry.id === compareChipId) || chips[1] || chips[0];
  const primaryScenario = useMemo(
    () => ({ chip, plan }),
    [chip, plan],
  );
  const load = activeLoads?.[phase] || { batch: 1, sequence: 2048 };
  const candidateScenario = useMemo(
    () => ({ chip: candidateChip, plan: { tp: compareTp, ep: compareEp, attnMode: compareAttnMode } }),
    [candidateChip, compareTp, compareEp, compareAttnMode],
  );
  const compareScenario = useMemo(
    () => resolveComparisonScenario(comparisonMode, primaryScenario, candidateScenario),
    [comparisonMode, primaryScenario, candidateScenario],
  );
  const coverage = useMemo(() => getChipCoverage(chip, "bf16"), [chip]);
  const compareCoverage = useMemo(
    () => compareScenario?.chip ? getChipCoverage(compareScenario.chip, "bf16") : null,
    [compareScenario],
  );
  const efficiency = useMemo(
    () => ({ flops: etaFlops, hbm: etaHbm, intra_node_comm: etaComm }),
    [etaFlops, etaHbm, etaComm],
  );
  const nodeLensResult = useMemo(
    () => buildNodeLens(structure, chip, { phase, batch: load.batch, sequence: load.sequence, plan: primaryScenario.plan, efficiency }),
    [structure, chip, phase, load, primaryScenario, efficiency],
  );
  const compareLensResult = useMemo(
    () => compareScenario
      ? buildNodeLens(structure, compareScenario.chip, { phase, plan: compareScenario.plan, efficiency })
      : null,
    [structure, compareScenario, phase, efficiency],
  );
  const nodeLens = nodeLensResult.nodes;
  const compareNodeLens = compareLensResult?.nodes || {};
  const flips = useMemo(
    () => compareLensResult?.ok ? boundFlips(nodeLens, compareNodeLens) : [],
    [nodeLens, compareNodeLens, compareLensResult?.ok],
  );
  const flipPaths = useMemo(() => new Set(flips.map((flip) => flip.path)), [flips]);
  const sharedHoveredPath = formulaHoveredPath ?? diagramHoveredPath;
  const activeFormulaPath = sharedHoveredPath ?? selectedPath;
  const ui = language === "en" ? {
    architecture: "Architecture",
    analysis: "Analysis",
    hideAnalysis: "Hide analysis",
    focus: "Focus canvas",
    exitFocus: "Exit focus",
    phase: "Phase",
    attention: "Attention",
    compare: "Compare",
    off: "Off",
    chip: "Chip",
    plan: "Plan",
    compareChip: "Compare GPU",
    compareTp: "Compare TP",
    compareEp: "Compare EP",
    compareAttention: "Compare Attention",
    flops: "eta FLOPs",
    hbm: "eta HBM",
    comm: "eta Comm",
    zoomOut: "Zoom out",
    fit: "Fit",
    zoomIn: "Zoom in",
    exportSvg: "SVG",
    expandAll: "Expand all",
    collapseAll: "Collapse all",
    base: "Base",
    chipComparison: "Chip comparison",
    planComparison: "Plan comparison",
    edges: "Edges",
    allEdges: "All",
    structureEdges: "Structure",
    dataflowEdges: "Data flow",
  } : {
    architecture: "架构",
    analysis: "分析配置",
    hideAnalysis: "收起分析",
    focus: "专注画布",
    exitFocus: "退出专注",
    phase: "阶段",
    attention: "Attention",
    compare: "对比",
    off: "关闭",
    chip: "芯片",
    plan: "方案",
    compareChip: "对比芯片",
    compareTp: "对比 TP",
    compareEp: "对比 EP",
    compareAttention: "对比 Attention",
    flops: "ηF",
    hbm: "ηHBM",
    comm: "ηComm",
    zoomOut: "缩小",
    fit: "适应画布",
    zoomIn: "放大",
    exportSvg: "SVG",
    expandAll: "展开全部",
    collapseAll: "收起全部",
    base: "基准",
    chipComparison: "芯片对比",
    planComparison: "方案对比",
    edges: "连线",
    allEdges: "全部",
    structureEdges: "结构",
    dataflowEdges: "数据流",
  };
  const diagramProps = {
    structure,
    zoom,
    fitNonce,
    selectedPath,
    matchedPaths,
    expandedGroups,
    externalHoveredPath: sharedHoveredPath,
    searchActive,
    onSelectNode,
    onFit,
    activeLenses,
    focusMode: canvasFocus,
    onExitFocus: () => setCanvasFocus(false),
    scrollSync: { group: compareScrollGroup.current },
    language,
    edgeMode,
    onEdgeModeChange: setEdgeMode,
  };
  useEffect(() => {
    onNodeLensChange?.(nodeLens);
  }, [nodeLens, onNodeLensChange]);
  return (
    <section className={`diagram-panel${canvasFocus ? " canvas-focus" : ""}`}>
      <div className="panel-toolbar">
        <h2>
          {ui.architecture}
          {searchActive && (
            <span className="hit-count inline"> · {english ? `${hitCount} match${hitCount === 1 ? "" : "es"}` : `${hitCount} 个匹配`}</span>
          )}
        </h2>
        <div className="toolbar-actions">
          {compactControls && <button type="button" onClick={() => setAdvancedOpen((value) => !value)}>{advancedOpen ? ui.hideAnalysis : ui.analysis}</button>}
          <button type="button" onClick={() => setCanvasFocus((value) => !value)}>{canvasFocus ? ui.exitFocus : ui.focus}</button>
          {advancedOpen && <div className="toolbar-analysis" aria-label={ui.analysis}>
          {!compactControls && <label className="lens-control">GPU<select value={chip?.id || ""} onChange={(event) => changeChip(event.target.value)}>{chips.map((entry) => <option key={entry.id} value={entry.id}>{chipOptionText(entry)}</option>)}</select></label>}
          {!compactControls && <>
            <label className="lens-control">{ui.phase}<select value={phase} onChange={(event) => changePhase(event.target.value)}><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label>
            <label className="lens-control">TP<input type="number" min="1" value={plan.tp} onChange={(event) => updatePlan({ ...plan, tp: Math.max(1, Number(event.target.value) || 1) })} /></label>
            <label className="lens-control">EP<input type="number" min="1" value={plan.ep} onChange={(event) => updatePlan({ ...plan, ep: Math.max(1, Number(event.target.value) || 1) })} /></label>
            <label className="lens-control">{ui.attention}<select value={plan.attnMode} onChange={(event) => updatePlan({ ...plan, attnMode: event.target.value })}><option value="tp">TP</option><option value="dp">DP</option></select></label>
          </>}
          <div className="lens-mode-switch" role="group" aria-label={ui.compare}>
            <span>{ui.compare}</span>
            {[
              [COMPARISON_MODE.OFF, ui.off],
              [COMPARISON_MODE.CHIP, ui.chip],
              [COMPARISON_MODE.PLAN, ui.plan],
            ].map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                className={comparisonMode === mode ? "active" : ""}
                aria-pressed={comparisonMode === mode}
                onClick={() => {
                  setComparisonMode(mode);
                  setDiagramHoveredPath(null);
                }}
              >
                {label}
              </button>
            ))}
          </div>
          {comparisonMode === COMPARISON_MODE.CHIP && <label className="lens-control">{ui.compareChip}<select value={candidateChip?.id || ""} onChange={(event) => setCompareChipId(event.target.value)}>{chips.map((entry) => <option key={entry.id} value={entry.id}>{chipOptionText(entry)}</option>)}</select></label>}
          {comparisonMode === COMPARISON_MODE.PLAN && <label className="lens-control">{ui.compareTp}<input type="number" min="1" value={compareTp} onChange={(event) => setCompareTp(Math.max(1, Number(event.target.value) || 1))} /></label>}
          {comparisonMode === COMPARISON_MODE.PLAN && <label className="lens-control">{ui.compareEp}<input type="number" min="1" value={compareEp} onChange={(event) => setCompareEp(Math.max(1, Number(event.target.value) || 1))} /></label>}
          {comparisonMode === COMPARISON_MODE.PLAN && <label className="lens-control">{ui.compareAttention}<select value={compareAttnMode} onChange={(event) => setCompareAttnMode(event.target.value)}><option value="tp">TP</option><option value="dp">DP</option></select></label>}
          <label className="lens-control">{ui.flops}<input type="number" min="0.1" max="1" step="0.05" value={etaFlops} onChange={(event) => setEtaFlops(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.7)))} /></label>
          <label className="lens-control">{ui.hbm}<input type="number" min="0.1" max="1" step="0.05" value={etaHbm} onChange={(event) => setEtaHbm(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.9)))} /></label>
          <label className="lens-control">{ui.comm}<input type="number" min="0.1" max="1" step="0.05" value={etaComm} onChange={(event) => setEtaComm(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.8)))} /></label>
          {!compactControls && <ManualChipForm language={language} onAdd={(entry) => { onAddChip?.(entry); changeChip(entry.id); }} />}
          </div>}
          {compactControls && <>
            <button type="button" onClick={onExpandAllGroups}>{ui.expandAll}</button>
            <button type="button" onClick={onCollapseAllGroups}>{ui.collapseAll}</button>
          </>}
          <button type="button" title={ui.zoomOut} aria-label={ui.zoomOut} onClick={() => onZoomChange(Math.max(0.25, zoom - 0.1))}>−</button>
          <button type="button" title={ui.fit} aria-label={ui.fit} onClick={onFit}>{ui.fit}</button>
          <span className="zoom-level" aria-label="Zoom level">{Math.round(zoom * 100)}%</span>
          <button type="button" title={ui.zoomIn} aria-label={ui.zoomIn} onClick={() => onZoomChange(Math.min(2.5, zoom + 0.1))}>+</button>
          <button type="button" title={ui.exportSvg} aria-label={ui.exportSvg} onClick={() => downloadSvg(structure)} disabled={!structure}>
            {ui.exportSvg}
          </button>
        </div>
      </div>
      <div className="diagram-lens-status" aria-label="Active Cost Lens">
        <span className="lens-status-mode">{activeMode === "pd" ? "PD" : english ? "Centralized" : "集中式"}</span>
        <span>{chip?.name || "Unknown GPU"}</span>
        <span>{phase}</span>
        <span>{activeNodes?.[activeMode === "pd" ? phase : "centralized"] || 1} {activeNodes?.[activeMode === "pd" ? phase : "centralized"] === 1 ? (english ? "node" : "节点") : (english ? "nodes" : "节点")} × {gpusPerNode} GPU</span>
        <span>TP{plan.tp} / PP{plan.pp || 1} / EP{plan.ep} / DP{plan.dp || 1}</span>
        <span>{english ? "Cost Lens" : "成本 Lens"}: {["vram", "compute", "memory", "kv"].filter((id) => activeLenses.has(id)).map((id) => id === "vram" ? "VRAM" : id === "kv" ? "KV Cache" : id[0].toUpperCase() + id.slice(1)).join(" · ") || "None"}</span>
      </div>
      <div className="diagram-legend" aria-label={language === "en" ? "Node and edge legend" : "节点与连线图例"}>{[["model", language === "en" ? "Container" : "容器"], ["embedding", "Embedding"], ["attention", "Attention"], ["mlp", "MLP / MoE"], ["output", language === "en" ? "Output" : "输出"]].map(([kind, label]) => <span key={kind}><i className={`legend-dot ${kind}`} />{label}</span>)}<span><i className="legend-line structure" />{language === "en" ? "Structure" : "结构"}</span><span><i className="legend-line dataflow" />{language === "en" ? "Data flow" : "数据流"}</span><span><i className="legend-line selected-path" />{language === "en" ? "Selected path" : "选中路径"}</span><span><i className="legend-dot selected-node" />{language === "en" ? "Path nodes" : "路径节点"}</span></div>
      <div className="diagram-edge-filter" role="group" aria-label={ui.edges}><span>{ui.edges}</span>{[["all", ui.allEdges], ["structure", ui.structureEdges], ["dataflow", ui.dataflowEdges]].map(([mode, label]) => <button key={mode} type="button" className={edgeMode === mode ? "active" : ""} aria-pressed={edgeMode === mode} onClick={() => setEdgeMode(mode)}>{label}</button>)}</div>
      {formulaLinks.length > 0 && <div className={`formula-strip${formulaOpen ? " is-open" : ""}`} aria-label={language === "en" ? "Formula index" : "公式索引"}>
        <button type="button" className="formula-strip-toggle" aria-expanded={formulaOpen} aria-controls="formula-index-items" onClick={() => setFormulaOpen((value) => !value)}>
          <span className="formula-strip-label">{language === "en" ? "Formula index" : "公式索引"}</span>
          <span className="formula-strip-count">{formulaLinks.length}</span>
          {activeFormulaPath && <span className="formula-strip-active">{formulaLinks.find((link) => link.path === activeFormulaPath)?.formulaId || "linked"}</span>}
          <span className="formula-strip-chevron" aria-hidden="true">{formulaOpen ? "−" : "+"}</span>
        </button>
        <div id="formula-index-items" className="formula-strip-links" hidden={!formulaOpen}>{formulaLinks.map((link) => <button key={link.path} data-node-path={link.path} className={activeFormulaPath === link.path ? "active" : ""} aria-pressed={activeFormulaPath === link.path} title={link.explanation || link.formulaId} onMouseEnter={() => setFormulaHoveredPath(link.path)} onMouseLeave={() => setFormulaHoveredPath(null)} onClick={() => onSelectNode?.(link.path)}>{link.formulaId}</button>)}</div>
      </div>}
      {!compactControls && chip && <div className="lens-coverage"><b>{chip.name}</b>{coverage.missing.length > 0 && <span>{english ? "Missing: " : "缺失："}{coverage.missing.join(english ? ", " : "、")}</span>}{coverage.warnings.map((warning) => <span key={warning}>{coverageWarningText(warning, english)}</span>)}{chip.source?.startsWith("http") && <a href={chip.source} target="_blank" rel="noreferrer">{english ? "Specification source" : "规格来源"}</a>}{comparisonMode === COMPARISON_MODE.CHIP && compareScenario?.chip && compareCoverage && <><b>{compareScenario.chip.name}</b>{compareCoverage.missing.length > 0 && <span>{english ? "Missing: " : "缺失："}{compareCoverage.missing.join(english ? ", " : "、")}</span>}{compareCoverage.warnings.map((warning) => <span key={`${compareScenario.chip.id}-${warning}`}>{coverageWarningText(warning, english)}</span>)}{compareScenario.chip.source?.startsWith("http") && <a href={compareScenario.chip.source} target="_blank" rel="noreferrer">{english ? "Specification source" : "规格来源"}</a>}</>}</div>}
      {structure && !nodeLensResult.ok && <div className="cost-plan-error">基准方案无效：{nodeLensResult.errors.join("；")}</div>}
      {structure && compareLensResult && !compareLensResult.ok && <div className="cost-plan-error">对比方案无效：{compareLensResult.errors.join("；")}</div>}
      {compareScenario && compareLensResult?.ok && <div className={`lens-flips${flips.length === 0 ? " empty" : ""}`}>{flips.length > 0 ? <>{english ? "Bound flips: " : "瓶颈类型翻转："}{flips.length} {english ? "nodes" : "个节点"}（{flips.slice(0, 4).map((flip) => `${flip.primary}→${flip.secondary}`).join(english ? ", " : "、")}{flips.length > 4 ? "…" : ""}）</> : (english ? "No bound flips under the current conditions" : "当前条件下没有瓶颈类型翻转")}</div>}
      {structure ? (compareScenario && compareLensResult?.ok ? <div className="diagram-compare">
        <DiagramPane
          label={scenarioLabel(ui.base, primaryScenario, language)}
          syncId="primary"
          {...diagramProps}
          nodeLens={nodeLens}
          comparisonPaths={flipPaths}
          onHoverPathChange={setDiagramHoveredPath}
        />
        <DiagramPane
          label={scenarioLabel(comparisonMode === COMPARISON_MODE.CHIP ? ui.chipComparison : ui.planComparison, compareScenario, language)}
          syncId="compare"
          {...diagramProps}
          nodeLens={compareNodeLens}
          comparisonPaths={flipPaths}
          onHoverPathChange={setDiagramHoveredPath}
        />
      </div> : <StructureDiagram
          {...diagramProps}
          nodeLens={nodeLens}
          onHoverPathChange={setDiagramHoveredPath}
          onToggleGroup={onToggleGroup}
          showGroupToggle
        />)
      : (
        <EmptyState />
      )}
    </section>
  );
}

export default ArchitectureTab;
