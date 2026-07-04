import StructureDiagram from "../diagram/StructureDiagram";
import EmptyState from "./EmptyState";

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
}) {
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
          <button onClick={() => onZoomChange(Math.max(0.7, zoom - 0.1))}>−</button>
          <button onClick={onFit}>Fit</button>
          <button onClick={() => onZoomChange(Math.min(1.4, zoom + 0.1))}>+</button>
          <button onClick={() => downloadSvg(structure)} disabled={!structure}>
            SVG
          </button>
        </div>
      </div>
      {structure ? (
        <StructureDiagram
          structure={structure}
          zoom={zoom}
          fitNonce={fitNonce}
          selectedPath={selectedPath}
          matchedPaths={matchedPaths}
          expandedGroups={expandedGroups}
          searchActive={searchActive}
          onSelectNode={onSelectNode}
          showGroupToggle={false}
        />
      ) : (
        <EmptyState />
      )}
    </section>
  );
}

export default ArchitectureTab;
