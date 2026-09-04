import { useEffect, useId, useMemo, useRef, useState } from "react";
import { layoutDiagram } from "./layout";
import { fitDiagramViewport, sameDiagramViewport } from "./viewport";
import { isEdgeRelated, isPathRelated } from "./hover";

function formatMetric(seconds) {
  if (!Number.isFinite(seconds)) return null;
  if (seconds >= 1) return `${seconds.toFixed(2)}s`;
  return `${(seconds * 1000).toFixed(1)}ms`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "-";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${Math.round(bytes)} B`;
}

function StructureDiagram({
  structure,
  zoom,
  fitNonce = 0,
  selectedPath,
  matchedPaths,
  expandedGroups,
  searchActive,
  onSelectNode,
  onToggleGroup,
  nodeLens,
  externalHoveredPath,
  comparisonPaths,
  onHoverPathChange,
  showGroupToggle = true,
  activeLenses = new Set(),
  onFit,
}) {
  const nodes = useMemo(
    () => layoutDiagram(structure.root, expandedGroups),
    [structure, expandedGroups]
  );
  const nodesByPath = useMemo(() => new Map(nodes.map((node) => [node.path, node])), [nodes]);
  const contentWidth = Math.max(1, ...nodes.map((node) => node.x + node.width + 28));
  const contentHeight = Math.max(1, ...nodes.map((node) => node.y + node.height + 28));
  const frameRef = useRef(null);
  const scrollRef = useRef(null);
  const panRef = useRef({ active: false, moved: false, x: 0, y: 0, left: 0, top: 0 });
  const fitNonceRef = useRef(fitNonce);
  const [viewport, setViewport] = useState(() =>
    fitDiagramViewport({
      contentWidth,
      contentHeight,
      viewportWidth: 960,
      viewportHeight: 470,
      zoom,
    })
  );
  const matched = matchedPaths instanceof Set ? matchedPaths : new Set();
  const [hoveredPath, setHoveredPath] = useState(null);
  const activeHoveredPath = externalHoveredPath ?? hoveredPath;
  const activeRelationPath = activeHoveredPath ?? selectedPath;
  const markerId = `diagram-arrow-${useId().replaceAll(":", "")}`;

  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || target?.isContentEditable) return;
      const selected = nodes.find((node) => node.path === selectedPath);
      if (event.key === "Escape") {
        event.preventDefault();
        onSelectNode?.(null);
      } else if (event.key === "0") {
        event.preventDefault();
        onFit?.();
      } else if ((event.key === "e" || event.key === "E") && selected?.isCollapsible && !selected.isExpanded) {
        event.preventDefault();
        onToggleGroup?.(selected.path);
      } else if ((event.key === "c" || event.key === "C") && selected?.isCollapsible && selected.isExpanded) {
        event.preventDefault();
        onToggleGroup?.(selected.path);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [nodes, selectedPath, onFit, onToggleGroup]);

  useEffect(() => {
    if (!selectedPath) return;
    const scroll = scrollRef.current;
    if (!scroll) return;
    const selectedNode = [...scroll.querySelectorAll("g[data-node-path]")].find((element) => element.getAttribute("data-node-path") === selectedPath);
    selectedNode?.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
  }, [selectedPath, nodes]);

  useEffect(() => {
    const element = frameRef.current;
    if (!element) return;
    const update = () => {
      const nextViewport = fitDiagramViewport({
          contentWidth,
          contentHeight,
          viewportWidth: element.clientWidth,
          viewportHeight: element.clientHeight,
          zoom,
        });
      const shouldResetScroll = fitNonceRef.current !== fitNonce;
      setViewport((prev) => (sameDiagramViewport(prev, nextViewport) ? prev : nextViewport));
      if (shouldResetScroll && scrollRef.current) {
        scrollRef.current.scrollLeft = 0;
        scrollRef.current.scrollTop = 0;
        fitNonceRef.current = fitNonce;
      }
    };
    update();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [contentWidth, contentHeight, fitNonce, zoom]);

  const width = viewport.canvasWidth;
  const height = viewport.canvasHeight;
  const contentTransform = `translate(${viewport.offsetX}, ${viewport.offsetY}) scale(${viewport.scale})`;

  function startPan(event) {
    if (event.target.closest("button, a, input, select, textarea")) return;
    const scroll = scrollRef.current;
    if (!scroll) return;
    panRef.current = { active: true, moved: false, x: event.clientX, y: event.clientY, left: scroll.scrollLeft, top: scroll.scrollTop };
    scroll.setPointerCapture?.(event.pointerId);
  }

  function movePan(event) {
    const pan = panRef.current;
    const scroll = scrollRef.current;
    if (!pan.active || !scroll) return;
    const dx = event.clientX - pan.x;
    const dy = event.clientY - pan.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) pan.moved = true;
    if (!pan.moved) return;
    event.preventDefault();
    scroll.scrollLeft = pan.left - dx;
    scroll.scrollTop = pan.top - dy;
  }

  function endPan(event) {
    const scroll = scrollRef.current;
    if (scroll?.hasPointerCapture?.(event.pointerId)) scroll.releasePointerCapture(event.pointerId);
    panRef.current.active = false;
  }

  function preventClickAfterPan(event) {
    if (!panRef.current.moved) return;
    event.stopPropagation();
    panRef.current.moved = false;
  }

  return (
    <div className="diagram-frame" ref={frameRef} data-active-lenses={[...activeLenses].join(",") }>
      <div className="diagram-scroll" ref={scrollRef} onPointerDown={startPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan} onClickCapture={preventClickAfterPan}>
        <div className="diagram-zoom" style={{ width, height }}>
          <svg
            className="diagram-svg"
            viewBox={`0 0 ${width} ${height}`}
            width={width}
            height={height}
            role="img"
            aria-label="Model architecture diagram"
          >
            <defs>
              <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="var(--diagram-arrow)" />
              </marker>
            </defs>
            <g transform={contentTransform}>
              {nodes.filter((node) => node.containerFrame).sort((a, b) => a.depth - b.depth).map((node) => {
                const frame = node.containerFrame;
                return <g key={`frame-${node.path}`} className="diagram-container-frame" data-container-path={node.path}>
                  <rect x={frame.x} y={frame.y} width={frame.width} height={frame.height} rx="12" />
                  <text x={frame.x + 10} y={frame.y + 15}>{frame.label}</text>
                </g>;
              })}
              {nodes.flatMap((node) =>
                node.children.map((child) => {
                  const target = nodesByPath.get(child);
                  if (!target) return null;
                  return (
                    <path
                      key={`${node.path}-${child}`}
                      d={`M ${node.x + node.width} ${node.y + node.height / 2} C ${node.x + node.width + 36} ${
                        node.y + node.height / 2
                      }, ${target.x - 36} ${target.y + target.height / 2}, ${target.x} ${target.y + target.height / 2}`}
                      fill="none"
                      className={activeRelationPath && isEdgeRelated(node.path, target.path, activeRelationPath) ? "diagram-edge related" : "diagram-edge"}
                      stroke="var(--diagram-arrow)"
                      strokeWidth={activeRelationPath && isEdgeRelated(node.path, target.path, activeRelationPath) ? "2.8" : "1.5"}
                      markerEnd={`url(#${markerId})`}
                    />
                  );
                })
              )}
              {nodes.map((node) => {
                const isSelected = selectedPath === node.path;
                const isAncestor = Boolean(selectedPath && selectedPath.startsWith(`${node.path}.`));
                const isMatch = matched.has(node.path);
                const isDimmed = searchActive && !isMatch;
                const lensEnabled = activeLenses.size > 0;
                const bound = lensEnabled ? nodeLens?.[node.path]?.bound || "unknown" : "unknown";
                const metrics = lensEnabled ? nodeLens?.[node.path]?.metrics || {} : {};
                const lensValues = [
                  activeLenses.has("compute") && ["C", metrics.computeSeconds],
                  activeLenses.has("memory") && ["M", metrics.memoryBytes, true],
                  activeLenses.has("vram") && ["V", metrics.vramBytes, true],
                ].filter(Boolean).map(([label, value, isBytes]) => value == null ? `${label} -` : `${label} ${isBytes ? formatBytes(value) : formatMetric(value)}`);
                const isHovered = activeHoveredPath === node.path;
                const isRelated = activeRelationPath && isPathRelated(node.path, activeRelationPath);
                const comparisonActive = comparisonPaths instanceof Set && comparisonPaths.size > 0;
                const isComparisonChange = comparisonActive && comparisonPaths.has(node.path);
                const isComparisonStable = comparisonActive && !isComparisonChange;
                const classes = [
                  "diagram-node",
                  node.typeClass,
                  `bound-${bound}`,
                  isHovered ? "hovered" : "",
                  isRelated ? "related" : "",
                  isSelected ? "selected" : "",
                  isAncestor ? "ancestor" : "",
                  isMatch ? "match" : "",
                  isDimmed ? "dimmed" : "",
                  isComparisonChange ? "comparison-change" : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <g
                    key={node.path}
                    data-node-path={node.path}
                    transform={`translate(${node.x}, ${node.y})`}
                    className={`${classes}${node.isCollapsible ? " diagram-group-node" : ""}${node.isExpanded ? " diagram-group-open" : " diagram-group-closed"}${isComparisonStable ? " comparison-stable" : ""}`}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => {
                      setHoveredPath(node.path);
                      onHoverPathChange?.(node.path);
                    }}
                    onMouseLeave={() => {
                      setHoveredPath(null);
                      onHoverPathChange?.(null);
                    }}
                    onClick={() => onSelectNode && onSelectNode(node.path)}
                  >
                    <title>{[node.fullName, node.path, node.repeat ? `×${node.repeat}` : null, node.node?.attributes?.formula_id].filter(Boolean).join(" · ")}</title>
                    <rect width={node.width} height={node.height} rx="10" className={classes} />
                    <foreignObject x="0" y="0" width={node.width} height={node.height}>
                      <div xmlns="http://www.w3.org/1999/xhtml" className="diagram-node-content" onMouseDown={(event) => { if (!event.target.closest("button, a, input, select, textarea")) onSelectNode?.(node.path); }} onClick={() => onSelectNode?.(node.path)}>
                        <div className="diagram-node-header">
                          <span className="diagram-title" title={node.fullName}>
                            {node.displayName}
                          </span>
                          {showGroupToggle && node.isCollapsible && (
                            <button
                              className="layer-group-toggle"
                              onClick={(e) => {
                                e.stopPropagation();
                                onToggleGroup && onToggleGroup(node.path);
                              }}
                              title={node.isExpanded ? "Collapse" : "Expand"}
                              aria-label={node.isExpanded ? "Collapse" : "Expand"}
                            >
                              {node.isExpanded ? "−" : "+"}
                            </button>
                          )}
                        </div>
                        <div className="diagram-node-badges">
                          {node.repeat && (
                            <span className="diagram-repeat" title={`${node.repeat} repeated layers`}>×{node.repeat}</span>
                          )}
                          {node.node?.attributes?.range && (
                            <span className="diagram-range" title="Layer range">{node.node.attributes.range}</span>
                          )}
                          {node.node?.attributes?.formula_id && (
                            <span className="diagram-formula" title={node.node.attributes.explanation || node.node.attributes.formula_id}>
                              {node.node.attributes.formula_id}
                            </span>
                          )}
                          {node.isCollapsible && <span className="diagram-children-count">{node.node.children.length} {node.node.children.length === 1 ? "child" : "children"}</span>}
                          {lensEnabled && nodeLens?.[node.path] && <span className="diagram-bound">{bound}</span>}
                        </div>
                        {node.metaLines.length > 0 && (
                          <ul className="diagram-meta">
                            {node.metaLines.map((line) => (
                              <li key={line} title={line}>
                                {line}
                              </li>
                            ))}
                          </ul>
                        )}
                        {lensValues.length > 0 && <div className="diagram-lens-values">{lensValues.map((value) => <span key={value} className="diagram-lens-value">{value}</span>)}</div>}
                      </div>
                    </foreignObject>
                  </g>
                );
              })}
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}

export default StructureDiagram;
