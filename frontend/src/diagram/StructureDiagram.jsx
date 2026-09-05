import { useEffect, useId, useMemo, useRef, useState } from "react";
import { layoutGraph } from "./layout.js";
import { layoutGraphWithElk } from "./elkLayout.js";
import { fitDiagramViewport, sameDiagramViewport, zoomForWheel } from "./viewport.js";
import { isGraphEdgeRelated, isPathRelated } from "./hover.js";
import { edgeStrokeWidth } from "./edgeStyle.js";

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

function edgePath(edge, source, target) {
  const section = edge.sections?.[0];
  if (section?.startPoint && section?.endPoint) {
    const points = [section.startPoint, ...(section.bendPoints || []), section.endPoint];
    return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  }
  return `M ${source.x + source.width} ${source.y + source.height / 2} C ${source.x + source.width + 36} ${source.y + source.height / 2}, ${target.x - 36} ${target.y + target.height / 2}, ${target.x} ${target.y + target.height / 2}`;
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
  edgeMode = "all",
  onZoomChange,
  onFit,
  focusMode = false,
  onExitFocus,
  scrollSync,
  scrollSyncId,
  language = "en",
}) {
  const english = language === "en";
  const baseGraph = useMemo(
    () => layoutGraph(structure.root, expandedGroups),
    [structure, expandedGroups]
  );
  const [graph, setGraph] = useState(baseGraph);
  useEffect(() => {
    let active = true;
    setGraph(baseGraph);
    layoutGraphWithElk(baseGraph).then((nextGraph) => {
      if (active) setGraph(nextGraph);
    }).catch(() => {
      // Keep the synchronous graph fallback if layout calculation is unavailable.
    });
    return () => { active = false; };
  }, [baseGraph]);
  const { nodes, edges, containerFrames } = graph;
  const nodesByPath = useMemo(() => new Map(nodes.map((node) => [node.path, node])), [nodes]);
  const contentWidth = Math.max(1, ...nodes.map((node) => node.x + node.width + 28));
  const contentHeight = Math.max(1, ...nodes.map((node) => node.y + node.height + 28));
  const stageBands = useMemo(() => {
    const labels = {
      input: english ? "Input" : "输入",
      representation: english ? "Representation" : "表示层",
      decoder: english ? "Decoder" : "解码器",
      output: english ? "Output" : "输出",
    };
    return ["input", "representation", "decoder", "output"].flatMap((stage) => {
      const members = nodes.filter((node) => node.stage === stage);
      if (members.length === 0) return [];
      const left = Math.min(...members.map((node) => node.x)) - 24;
      const right = Math.max(...members.map((node) => node.x + node.width)) + 24;
      const countLabel = english ? `${members.length} node${members.length === 1 ? "" : "s"}` : `${members.length} 个节点`;
      return [{ stage, label: `${labels[stage]} · ${countLabel}`, x: left, width: right - left }];
    });
  }, [english, nodes]);
  const frameRef = useRef(null);
  const scrollRef = useRef(null);
  const panRef = useRef({ active: false, moved: false, x: 0, y: 0, left: 0, top: 0 });
  const [scrollPosition, setScrollPosition] = useState({ left: 0, top: 0 });
  const [miniMapOpen, setMiniMapOpen] = useState(true);
  useEffect(() => {
    if (!scrollSync?.group || !scrollSyncId) return undefined;
    scrollSync.group.set(scrollSyncId, scrollRef);
    return () => scrollSync.group.delete(scrollSyncId);
  }, [scrollSync, scrollSyncId]);

  function handleScroll(event) {
    const next = { left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop };
    setScrollPosition(next);
    if (!scrollSync?.group || scrollSync.group.busy) return;
    scrollSync.group.busy = true;
    scrollSync.group.forEach((ref, id) => {
      if (id !== scrollSyncId && ref.current) {
        ref.current.scrollLeft = next.left;
        ref.current.scrollTop = next.top;
      }
    });
    scrollSync.group.busy = false;
  }
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
  const activeStage = nodesByPath.get(activeRelationPath)?.stage;
  const markerId = `diagram-arrow-${useId().replaceAll(":", "")}`;
  const flowMarkerId = `${markerId}-flow`;

  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || target?.isContentEditable) return;
      const selected = nodes.find((node) => node.path === selectedPath);
      if (event.key === "Escape") {
        event.preventDefault();
        if (focusMode) onExitFocus?.();
        else onSelectNode?.(null);
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
  }, [nodes, selectedPath, onFit, onToggleGroup, focusMode, onExitFocus]);

  useEffect(() => {
    if (!selectedPath) return;
    const scroll = scrollRef.current;
    if (!scroll) return;
    const selectedNode = [...scroll.querySelectorAll("g[data-node-path]")].find((element) => element.getAttribute("data-node-path") === selectedPath);
    if (!selectedNode) return;
    // Keep selection framing inside the diagram; scrollIntoView can move the
    // page itself and break the fixed cost bar / mobile inspector layout.
    const scrollRect = scroll.getBoundingClientRect();
    const nodeRect = selectedNode.getBoundingClientRect();
    scroll.scrollBy({
      left: (nodeRect.left + nodeRect.width / 2) - (scrollRect.left + scrollRect.width / 2),
      top: (nodeRect.top + nodeRect.height / 2) - (scrollRect.top + scrollRect.height / 2),
      behavior: "smooth",
    });
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
  const miniMapWidth = 148;
  const miniMapHeight = 92;
  const miniScale = Math.min(miniMapWidth / Math.max(contentWidth, 1), miniMapHeight / Math.max(contentHeight, 1));
  const miniViewport = {
    x: Math.max(0, (scrollPosition.left - viewport.offsetX) / viewport.scale) * miniScale,
    y: Math.max(0, (scrollPosition.top - viewport.offsetY) / viewport.scale) * miniScale,
    width: Math.min(miniMapWidth, (scrollRef.current?.clientWidth || miniMapWidth) / viewport.scale * miniScale),
    height: Math.min(miniMapHeight, (scrollRef.current?.clientHeight || miniMapHeight) / viewport.scale * miniScale),
  };

  function jumpFromMiniMap(event) {
    const scroll = scrollRef.current;
    if (!scroll) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / miniScale;
    const y = (event.clientY - rect.top) / miniScale;
    scroll.scrollLeft = Math.max(0, x * viewport.scale + viewport.offsetX - scroll.clientWidth / 2);
    scroll.scrollTop = Math.max(0, y * viewport.scale + viewport.offsetY - scroll.clientHeight / 2);
  }

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

  function handleWheel(event) {
    const scroll = scrollRef.current;
    if (!scroll || (!event.ctrlKey && !event.metaKey)) return;
    event.preventDefault();
    onZoomChange?.(zoomForWheel(zoom, event.deltaY));
  }

  return (
    <div className="diagram-frame" ref={frameRef} data-active-lenses={[...activeLenses].join(",") }>
      <div className="diagram-scroll" ref={scrollRef} onScroll={handleScroll} onWheel={handleWheel} onPointerDown={startPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan} onClickCapture={preventClickAfterPan}>
        <div className="diagram-zoom" style={{ width, height }}>
          <svg
            className="diagram-svg"
            viewBox={`0 0 ${width} ${height}`}
            width={width}
            height={height}
            role="img"
            aria-label={english ? "Model architecture diagram" : "模型架构图"}
          >
            <defs>
              <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="var(--diagram-arrow)" />
              </marker>
              <marker id={flowMarkerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="#d08a3a" />
              </marker>
            </defs>
            <g transform={contentTransform}>
              <g className="diagram-stage-bands" aria-hidden="true">
                {stageBands.map((band) => <g key={band.stage} className={`diagram-stage-band stage-${band.stage}${activeStage && activeStage === band.stage ? " active" : ""}${activeStage && activeStage !== band.stage ? " dimmed" : ""}`}>
                  <rect x={band.x} y={0} width={band.width} height={contentHeight} rx="10" />
                  <text x={band.x + 12} y={18}>{band.label}</text>
                </g>)}
              </g>
              {containerFrames.sort((a, b) => nodesByPath.get(a.id).depth - nodesByPath.get(b.id).depth).map((frame) => {
                const depth = nodesByPath.get(frame.id)?.depth || 0;
                return <g key={`frame-${frame.id}`} className={`diagram-container-frame depth-${Math.min(depth, 3)}${frame.kind === "graph-group" ? " graph-group-frame" : ""}`} data-container-path={frame.id}>
                  <rect x={frame.x} y={frame.y} width={frame.width} height={frame.height} rx="12" />
                  <text x={frame.x + 10} y={frame.y + 15}>{frame.label}</text>
                </g>;
              })}
              {edges.filter((edge) => edgeMode === "all" || edge.kind === edgeMode).map((edge) => {
                  const node = nodesByPath.get(edge.source);
                  const target = nodesByPath.get(edge.target);
                  if (!node || !target) return null;
                  const related = activeRelationPath && (edge.kind === "dataflow"
                    ? edge.source === activeRelationPath || edge.target === activeRelationPath
                    : isGraphEdgeRelated(edge.source, edge.target, activeRelationPath));
                  const searchRelated = !searchActive || matched.has(edge.source) || matched.has(edge.target)
                    || [...matched].some((path) => isPathRelated(edge.source, path) || isPathRelated(edge.target, path));
                  return (
                    <path
                      key={edge.id}
                      data-edge-id={edge.id}
                      d={edgePath(edge, node, target)}
                      fill="none"
                      className={`diagram-edge ${edge.kind === "dataflow" ? "dataflow" : "structure"}${edge.evidence === "module-order" ? " mainflow" : ""}${related ? " related" : ""}${searchRelated ? "" : " search-dimmed"}`}
                      stroke="var(--diagram-arrow)"
                      strokeWidth={related ? "2.8" : edgeStrokeWidth(edge, node)}
                      markerEnd={`url(#${edge.kind === "dataflow" ? flowMarkerId : markerId})`}
                    >
                      <title>{edge.kind === "dataflow" ? (edge.evidence === "module-order" ? (english ? "Model stage flow" : "模型阶段流") : (english ? "Data flow (matching tensor shapes)" : "数据流（Tensor Shape 匹配）")) : (english ? "Module structure" : "模块结构")} · {node.fullName} → {target.fullName}</title>
                    </path>
                  );
                })}
              {nodes.map((node) => {
                const isSelected = selectedPath === node.path;
                const isAncestor = Boolean(selectedPath && selectedPath.startsWith(`${node.path}.`));
                const isMatch = matched.has(node.path);
                const isDimmed = searchActive && !isMatch;
                const lensEnabled = activeLenses.size > 0;
                const bound = lensEnabled ? nodeLens?.[node.path]?.bound || "unknown" : "unknown";
                const metrics = lensEnabled ? nodeLens?.[node.path]?.metrics || {} : {};
                const lensValues = [
                  activeLenses.has("compute") && ["compute", "C", metrics.computeSeconds],
                  activeLenses.has("memory") && ["memory", "M", metrics.memoryBytes, true],
                  activeLenses.has("vram") && ["vram", "V", metrics.vramBytes, true],
                ].filter(Boolean).map(([id, label, value, isBytes]) => ({ id, text: value == null ? `${label} -` : `${label} ${isBytes ? formatBytes(value) : formatMetric(value)}` }));
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
                    role="button"
                    tabIndex={0}
                    aria-label={`${node.fullName} at ${node.path}`}
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
                    onKeyDown={(event) => {
                      if ((event.key === "Enter" || event.key === " ") && !event.target.closest("button, a, input, select, textarea")) {
                        event.preventDefault();
                        onSelectNode?.(node.path);
                      }
                    }}
                  >
                    <title>{[node.fullName, node.path, node.repeat ? `×${node.repeat}` : null, node.node?.attributes?.formula_id].filter(Boolean).join(" · ")}</title>
                    <rect width={node.width} height={node.height} rx="10" className={classes} />
                    <circle cx="0" cy={node.height / 2} r="3.5" className="diagram-port input" />
                    <circle cx={node.width} cy={node.height / 2} r="3.5" className="diagram-port output" />
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
                              title={node.isExpanded ? (english ? "Collapse" : "收起") : (english ? "Expand" : "展开")}
                              aria-label={node.isExpanded ? (english ? "Collapse" : "收起") : (english ? "Expand" : "展开")}
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
                        {lensValues.length > 0 && <div className="diagram-lens-values">{lensValues.map(({ id, text }) => <span key={id} className={`diagram-lens-value lens-${id}`}>{text}</span>)}</div>}
                      </div>
                    </foreignObject>
                  </g>
                );
              })}
            </g>
          </svg>
        </div>
      </div>
      <button type="button" className="diagram-minimap-toggle" aria-label={miniMapOpen ? (english ? "Hide structure overview" : "隐藏结构概览") : (english ? "Show structure overview" : "显示结构概览")} title={miniMapOpen ? (english ? "Hide structure overview" : "隐藏结构概览") : (english ? "Show structure overview" : "显示结构概览")} onClick={() => setMiniMapOpen((value) => !value)}>{miniMapOpen ? "×" : "map"}</button>
      {miniMapOpen && <button type="button" className="diagram-minimap" aria-label={english ? "Structure overview" : "结构概览"} title={english ? "Click to navigate the structure" : "点击定位结构"} onClick={jumpFromMiniMap}>
        <svg viewBox={`0 0 ${miniMapWidth} ${miniMapHeight}`} role="img" aria-label={english ? "Structure overview map" : "结构概览图"}>
          <g transform={`scale(${miniScale})`}>
            {containerFrames.map((frame) => <rect key={`mini-frame-${frame.id}`} x={frame.x} y={frame.y} width={frame.width} height={frame.height} className="mini-frame" />)}
            {nodes.map((node) => <rect key={`mini-node-${node.path}`} x={node.x} y={node.y} width={node.width} height={node.height} className={`mini-node ${node.typeClass}${selectedPath === node.path ? " selected" : ""}${selectedPath && selectedPath.startsWith(`${node.path}.`) ? " ancestor" : ""}`} />)}
            <rect x={miniViewport.x / miniScale} y={miniViewport.y / miniScale} width={miniViewport.width / miniScale} height={miniViewport.height / miniScale} className="mini-viewport" />
          </g>
        </svg>
      </button>}
    </div>
  );
}

export default StructureDiagram;
