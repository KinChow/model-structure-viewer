import { useMemo, useState } from "react";
import StructureDiagram from "../diagram/StructureDiagram";
import EmptyState from "./EmptyState";
import { PUBLIC_CHIPS } from "../cost/chips/public.js";
import { collectFormulaLinks } from "../diagram/formulaLinks.js";
import { boundFlips, COMPARISON_MODE, resolveComparisonScenario } from "../diagram/compare.js";
import { buildNodeLens } from "../diagram/lens.js";
import ManualChipForm from "./ManualChipForm.jsx";
import { getChipCoverage } from "../cost/chips/coverage.js";

function downloadSvg(structure) {
  const svg = document.querySelector(".diagram-svg");
  if (!svg) return;
  const blob = new Blob([svg.outerHTML], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${structure?.summary?.model_family || "model"}-structure.svg`;
  link.click();
  URL.revokeObjectURL(url);
}

function chipLinkText(chip) {
  const bandwidth = chip?.interconnect?.intra_node?.bandwidth;
  return Number.isFinite(bandwidth) ? `${bandwidth / 1e9} GB/s` : "链路未知";
}

function chipOptionText(chip) {
  return `${chip.name}${chip.confidence === "local" ? " (local)" : ""}`;
}

function scenarioLabel(prefix, scenario) {
  if (!scenario) return prefix;
  const { chip, plan } = scenario;
  return `${prefix} · ${chip?.name || "未知芯片"} · TP ${plan.tp} / EP ${plan.ep} / Attention ${String(plan.attnMode).toUpperCase()} · ${chipLinkText(chip)}`;
}

function DiagramPane({ label, ...diagramProps }) {
  return (
    <div>
      <div className="diagram-compare-label">{label}</div>
      <StructureDiagram {...diagramProps} showGroupToggle={false} />
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
  chips = PUBLIC_CHIPS,
  onAddChip,
  compactControls = false,
}) {
  const [phase, setPhase] = useState("prefill");
  const [chipId, setChipId] = useState(chips[0]?.id || "");
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
  const formulaLinks = useMemo(() => collectFormulaLinks(structure?.root), [structure]);
  const chip = chips.find((entry) => entry.id === chipId) || chips[0];
  const candidateChip = chips.find((entry) => entry.id === compareChipId) || chips[1] || chips[0];
  const primaryScenario = useMemo(
    () => ({ chip, plan: { tp, ep, attnMode } }),
    [chip, tp, ep, attnMode],
  );
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
    () => buildNodeLens(structure, chip, { phase, plan: primaryScenario.plan, efficiency }),
    [structure, chip, phase, primaryScenario, efficiency],
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
    activeLenses,
  };
  return (
    <section className="diagram-panel">
      <div className="panel-toolbar">
        <h2>
          Architecture
          {searchActive && (
            <span className="hit-count inline"> · {hitCount} match{hitCount === 1 ? "" : "es"}</span>
          )}
        </h2>
        <div className="toolbar-actions">
          {!compactControls && <>
          <label className="lens-control">Lens<select value={chip?.id || ""} onChange={(event) => setChipId(event.target.value)}>{chips.map((entry) => <option key={entry.id} value={entry.id}>{chipOptionText(entry)}</option>)}</select></label>
          <label className="lens-control">阶段<select value={phase} onChange={(event) => setPhase(event.target.value)}><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label>
          <label className="lens-control">TP<input type="number" min="1" value={tp} onChange={(event) => setTp(Math.max(1, Number(event.target.value) || 1))} /></label>
          <label className="lens-control">EP<input type="number" min="1" value={ep} onChange={(event) => setEp(Math.max(1, Number(event.target.value) || 1))} /></label>
          <label className="lens-control">Attention<select value={attnMode} onChange={(event) => setAttnMode(event.target.value)}><option value="tp">TP</option><option value="dp">DP</option></select></label>
          <div className="lens-mode-switch" role="group" aria-label="对比模式">
            <span>对比</span>
            {[
              [COMPARISON_MODE.OFF, "关闭"],
              [COMPARISON_MODE.CHIP, "芯片"],
              [COMPARISON_MODE.PLAN, "方案"],
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
          {comparisonMode === COMPARISON_MODE.CHIP && <label className="lens-control">对比芯片<select value={candidateChip?.id || ""} onChange={(event) => setCompareChipId(event.target.value)}>{chips.map((entry) => <option key={entry.id} value={entry.id}>{chipOptionText(entry)}</option>)}</select></label>}
          {comparisonMode === COMPARISON_MODE.PLAN && <label className="lens-control">对比 TP<input type="number" min="1" value={compareTp} onChange={(event) => setCompareTp(Math.max(1, Number(event.target.value) || 1))} /></label>}
          {comparisonMode === COMPARISON_MODE.PLAN && <label className="lens-control">对比 EP<input type="number" min="1" value={compareEp} onChange={(event) => setCompareEp(Math.max(1, Number(event.target.value) || 1))} /></label>}
          {comparisonMode === COMPARISON_MODE.PLAN && <label className="lens-control">对比 Attention<select value={compareAttnMode} onChange={(event) => setCompareAttnMode(event.target.value)}><option value="tp">TP</option><option value="dp">DP</option></select></label>}
          <label className="lens-control">ηF<input type="number" min="0.1" max="1" step="0.05" value={etaFlops} onChange={(event) => setEtaFlops(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.7)))} /></label>
          <label className="lens-control">ηHBM<input type="number" min="0.1" max="1" step="0.05" value={etaHbm} onChange={(event) => setEtaHbm(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.9)))} /></label>
          <label className="lens-control">ηComm<input type="number" min="0.1" max="1" step="0.05" value={etaComm} onChange={(event) => setEtaComm(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.8)))} /></label>
          <ManualChipForm onAdd={(entry) => { onAddChip?.(entry); setChipId(entry.id); }} />
          </>}
          {compactControls && <>
            <button type="button" onClick={onExpandAllGroups}>展开全部</button>
            <button type="button" onClick={onCollapseAllGroups}>收起全部</button>
          </>}
          <button onClick={() => onZoomChange(Math.max(0.7, zoom - 0.1))}>−</button>
          <button onClick={onFit}>Fit</button>
          <button onClick={() => onZoomChange(Math.min(1.4, zoom + 0.1))}>+</button>
          <button onClick={() => downloadSvg(structure)} disabled={!structure}>
            SVG
          </button>
        </div>
      </div>
      <div className="diagram-lens-status" aria-label="Active Cost Lens">Cost Lens · {["vram", "compute", "memory", "kv"].filter((id) => activeLenses.has(id)).map((id) => id === "vram" ? "VRAM" : id === "kv" ? "KV Cache" : id[0].toUpperCase() + id.slice(1)).join(" · ") || "None"}</div>
      {formulaLinks.length > 0 && <div className="formula-strip" aria-label="公式索引"><span className="formula-strip-label">公式</span>{formulaLinks.map((link) => <button key={link.path} data-node-path={link.path} className={activeFormulaPath === link.path ? "active" : ""} aria-pressed={activeFormulaPath === link.path} title={link.explanation || link.formulaId} onMouseEnter={() => setFormulaHoveredPath(link.path)} onMouseLeave={() => setFormulaHoveredPath(null)} onClick={() => onSelectNode?.(link.path)}>{link.formulaId}</button>)}</div>}
      {!compactControls && chip && <div className="lens-coverage"><b>{chip.name}</b>{coverage.missing.length > 0 && <span>缺失：{coverage.missing.join("、")}</span>}{coverage.warnings.map((warning) => <span key={warning}>{warning}</span>)}{chip.source?.startsWith("http") && <a href={chip.source} target="_blank" rel="noreferrer">规格来源</a>}{comparisonMode === COMPARISON_MODE.CHIP && compareScenario?.chip && compareCoverage && <><b>{compareScenario.chip.name}</b>{compareCoverage.missing.length > 0 && <span>缺失：{compareCoverage.missing.join("、")}</span>}{compareCoverage.warnings.map((warning) => <span key={`${compareScenario.chip.id}-${warning}`}>{warning}</span>)}{compareScenario.chip.source?.startsWith("http") && <a href={compareScenario.chip.source} target="_blank" rel="noreferrer">规格来源</a>}</>}</div>}
      {structure && !nodeLensResult.ok && <div className="cost-plan-error">基准方案无效：{nodeLensResult.errors.join("；")}</div>}
      {structure && compareLensResult && !compareLensResult.ok && <div className="cost-plan-error">对比方案无效：{compareLensResult.errors.join("；")}</div>}
      {compareScenario && compareLensResult?.ok && <div className={`lens-flips${flips.length === 0 ? " empty" : ""}`}>{flips.length > 0 ? <>瓶颈类型翻转：{flips.length} 个节点（{flips.slice(0, 4).map((flip) => `${flip.primary}→${flip.secondary}`).join("、")}{flips.length > 4 ? "…" : ""}）</> : "当前条件下没有瓶颈类型翻转"}</div>}
      {structure ? (compareScenario && compareLensResult?.ok ? <div className="diagram-compare">
        <DiagramPane
          label={scenarioLabel("基准", primaryScenario)}
          {...diagramProps}
          nodeLens={nodeLens}
          comparisonPaths={flipPaths}
          onHoverPathChange={setDiagramHoveredPath}
        />
        <DiagramPane
          label={scenarioLabel(comparisonMode === COMPARISON_MODE.CHIP ? "芯片对比" : "方案对比", compareScenario)}
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
