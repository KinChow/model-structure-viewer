import { useMemo, useState } from "react";
import StructureDiagram from "../diagram/StructureDiagram";
import EmptyState from "./EmptyState";
import { normalizeConfig } from "../structure/config/normalize.js";
import { computeNodeCosts } from "../cost/compute.js";
import { classifyRoofline } from "../cost/roofline.js";
import { nodeCommunicationBytes } from "../cost/comm.js";
import { PUBLIC_CHIPS } from "../cost/chips/public.js";
import { collectFormulaLinks } from "../diagram/formulaLinks.js";
import { boundFlips } from "../diagram/compare.js";
import ManualChipForm from "./ManualChipForm.jsx";

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
  chips = PUBLIC_CHIPS,
  onAddChip,
}) {
  const [phase, setPhase] = useState("prefill");
  const [chipId, setChipId] = useState(chips[0]?.id || "");
  const [tp, setTp] = useState(1);
  const [ep, setEp] = useState(1);
  const [attnMode, setAttnMode] = useState("tp");
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [compareChipId, setCompareChipId] = useState(chips[1]?.id || chips[0]?.id || "");
  const [etaFlops, setEtaFlops] = useState(0.7);
  const [etaHbm, setEtaHbm] = useState(0.9);
  const [etaComm, setEtaComm] = useState(0.8);
  const [formulaHoveredPath, setFormulaHoveredPath] = useState(null);
  const formulaLinks = useMemo(() => collectFormulaLinks(structure?.root), [structure]);
  const chip = chips.find((entry) => entry.id === chipId) || chips[0];
  const compareChip = chips.find((entry) => entry.id === compareChipId) || chips[1] || chips[0];
  const nodeLens = useMemo(() => {
    if (!structure?.root || !structure.extra_config || !chip) return {};
    const config = normalizeConfig(structure.extra_config);
    const rows = computeNodeCosts(structure.root, config, { batch: 1, sequence: 2048, phase });
    return Object.fromEntries(rows.map((row) => [row.path, classifyRoofline({
      macs: row.macs,
      weightBytes: row.weightBytes,
      actInBytes: 0,
      actOutBytes: 0,
      commBytes: nodeCommunicationBytes(row.node, config, { tp, ep, attnMode }, { batch: 1, tokens: phase === "decode" ? 1 : 2048, bytesPerElement: 2 }),
    }, chip, { efficiency: { flops: etaFlops, hbm: etaHbm, intra_node_comm: etaComm } })]));
  }, [structure, chip, phase, tp, ep, attnMode, etaFlops, etaHbm, etaComm]);
  const compareNodeLens = useMemo(() => {
    if (!compareEnabled || !structure?.root || !structure.extra_config || !compareChip) return {};
    const config = normalizeConfig(structure.extra_config);
    const rows = computeNodeCosts(structure.root, config, { batch: 1, sequence: 2048, phase });
    return Object.fromEntries(rows.map((row) => [row.path, classifyRoofline({
      macs: row.macs, weightBytes: row.weightBytes, actInBytes: 0, actOutBytes: 0,
      commBytes: nodeCommunicationBytes(row.node, config, { tp, ep, attnMode }, { batch: 1, tokens: phase === "decode" ? 1 : 2048, bytesPerElement: 2 }),
    }, compareChip, { efficiency: { flops: etaFlops, hbm: etaHbm, intra_node_comm: etaComm } })]));
  }, [structure, compareChip, compareEnabled, phase, tp, ep, attnMode, etaFlops, etaHbm, etaComm]);
  const flips = useMemo(() => compareEnabled ? boundFlips(nodeLens, compareNodeLens) : [], [compareEnabled, nodeLens, compareNodeLens]);
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
          <label className="lens-control">Lens<select value={chip?.id || ""} onChange={(event) => setChipId(event.target.value)}>{chips.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}{entry.confidence === "local" ? " (local)" : ""}</option>)}</select></label>
          <label className="lens-control">阶段<select value={phase} onChange={(event) => setPhase(event.target.value)}><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label>
          <label className="lens-control">TP<input type="number" min="1" value={tp} onChange={(event) => setTp(Math.max(1, Number(event.target.value) || 1))} /></label>
          <label className="lens-control">EP<input type="number" min="1" value={ep} onChange={(event) => setEp(Math.max(1, Number(event.target.value) || 1))} /></label>
          <label className="lens-control">Attention<select value={attnMode} onChange={(event) => setAttnMode(event.target.value)}><option value="tp">TP</option><option value="dp">DP</option></select></label>
          <label className="lens-control"><input type="checkbox" checked={compareEnabled} onChange={(event) => setCompareEnabled(event.target.checked)} />双卡对比</label>
          {compareEnabled && <label className="lens-control">对比卡<select value={compareChip?.id || ""} onChange={(event) => setCompareChipId(event.target.value)}>{chips.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select></label>}
          <label className="lens-control">ηF<input type="number" min="0.1" max="1" step="0.05" value={etaFlops} onChange={(event) => setEtaFlops(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.7)))} /></label>
          <label className="lens-control">ηHBM<input type="number" min="0.1" max="1" step="0.05" value={etaHbm} onChange={(event) => setEtaHbm(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.9)))} /></label>
          <label className="lens-control">ηComm<input type="number" min="0.1" max="1" step="0.05" value={etaComm} onChange={(event) => setEtaComm(Math.min(1, Math.max(0.1, Number(event.target.value) || 0.8)))} /></label>
          <ManualChipForm onAdd={(entry) => { onAddChip?.(entry); setChipId(entry.id); }} />
          <button onClick={() => onZoomChange(Math.max(0.7, zoom - 0.1))}>−</button>
          <button onClick={onFit}>Fit</button>
          <button onClick={() => onZoomChange(Math.min(1.4, zoom + 0.1))}>+</button>
          <button onClick={() => downloadSvg(structure)} disabled={!structure}>
            SVG
          </button>
        </div>
      </div>
      {formulaLinks.length > 0 && <div className="formula-strip" aria-label="公式索引"><span className="formula-strip-label">公式</span>{formulaLinks.map((link) => <button key={link.path} title={link.explanation || link.formulaId} onMouseEnter={() => setFormulaHoveredPath(link.path)} onMouseLeave={() => setFormulaHoveredPath(null)} onClick={() => onSelectNode?.(link.path)}>{link.formulaId}</button>)}</div>}
      {compareEnabled && flips.length > 0 && <div className="lens-flips">bound 翻转：{flips.length} 个节点（{flips.slice(0, 4).map((flip) => `${flip.primary}→${flip.secondary}`).join("、")}{flips.length > 4 ? "…" : ""}）</div>}
      {structure ? (compareEnabled ? <div className="diagram-compare"><div><div className="diagram-compare-label">{chip?.name || "主卡"}</div><StructureDiagram
          structure={structure}
          zoom={zoom}
          fitNonce={fitNonce}
          selectedPath={selectedPath}
          matchedPaths={matchedPaths}
          expandedGroups={expandedGroups}
          nodeLens={nodeLens}
          externalHoveredPath={formulaHoveredPath}
          searchActive={searchActive}
          onSelectNode={onSelectNode}
          showGroupToggle={false}
        /></div><div><div className="diagram-compare-label">{compareChip?.name || "对比卡"}</div><StructureDiagram
          structure={structure}
          zoom={zoom}
          fitNonce={fitNonce}
          selectedPath={selectedPath}
          matchedPaths={matchedPaths}
          expandedGroups={expandedGroups}
          nodeLens={compareNodeLens}
          externalHoveredPath={formulaHoveredPath}
          searchActive={searchActive}
          onSelectNode={onSelectNode}
          showGroupToggle={false}
        /></div></div> : <StructureDiagram
          structure={structure}
          zoom={zoom}
          fitNonce={fitNonce}
          selectedPath={selectedPath}
          matchedPaths={matchedPaths}
          expandedGroups={expandedGroups}
          nodeLens={nodeLens}
          externalHoveredPath={formulaHoveredPath}
          searchActive={searchActive}
          onSelectNode={onSelectNode}
          showGroupToggle={false}
        />)
      : (
        <EmptyState />
      )}
    </section>
  );
}

export default ArchitectureTab;
