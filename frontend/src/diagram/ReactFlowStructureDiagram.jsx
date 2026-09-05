import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  BaseEdge,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  getBezierPath,
  getSmoothStepPath,
  useReactFlow,
} from "@xyflow/react";
import { SmartEdgeProvider, useSmartEdgePath } from "@tisoap/react-flow-smart-edge";
import { layoutGraph } from "./layout.js";
import { layoutGraphWithElk } from "./elkLayout.js";
import { isPathRelated, relatedDataflowEdgeIds } from "./hover.js";
import { edgeStrokeWidth } from "./edgeStyle.js";

const EMPTY_SET = new Set();
const DATAFLOW_MARKER = { type: MarkerType.ArrowClosed, width: 10, height: 10, color: "#d08a3a" };
const HoverContext = createContext({ activeRelationPath: null, onHover: null });

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

function formatShape(shape) {
  if (shape == null) return null;
  if (Array.isArray(shape)) return `[${shape.join(", ")}]`;
  if (typeof shape === "string") return shape;
  try {
    return JSON.stringify(shape);
  } catch {
    return String(shape);
  }
}

function parentPath(path) {
  const index = path.lastIndexOf(".");
  return index > 0 ? path.slice(0, index) : null;
}

function absoluteNodeBox(node, getNode) {
  let x = node.position?.x || 0;
  let y = node.position?.y || 0;
  let parentId = node.parentId;
  while (parentId) {
    const parent = getNode(parentId);
    if (!parent) break;
    x += parent.position?.x || 0;
    y += parent.position?.y || 0;
    parentId = parent.parentId;
  }
  return {
    x,
    y,
    width: node.measured?.width || node.width || 0,
    height: node.measured?.height || node.height || 0,
  };
}

function MsvNode({ data, selected }) {
  const { node, english, showGroupToggle, onSelect, onToggle, nodeLens, activeLenses, matched, searchActive, comparisonPaths } = data;
  const hover = useContext(HoverContext);
  const activeRelationPath = hover.activeRelationPath;
  const verticalFlow = node.depth > 1;
  const isOpenGroup = node.isCollapsible && node.isExpanded;
  const height = isOpenGroup ? 28 : node.height;
  const related = activeRelationPath && isPathRelated(node.path, activeRelationPath);
  const isMatch = matched.has(node.path);
  const lensEnabled = activeLenses.size > 0;
  const bound = lensEnabled ? nodeLens?.[node.path]?.bound || "unknown" : "unknown";
  const metrics = lensEnabled ? nodeLens?.[node.path]?.metrics || {} : {};
  const lensValues = [
    activeLenses.has("compute") && ["compute", "C", metrics.computeSeconds],
    activeLenses.has("memory") && ["memory", "M", metrics.memoryBytes, true],
    activeLenses.has("vram") && ["vram", "V", metrics.vramBytes, true],
  ].filter(Boolean).map(([id, label, value, bytes]) => ({
    id,
    text: value == null ? `${label} -` : `${label} ${bytes ? formatBytes(value) : formatMetric(value)}`,
  }));
  const classes = [
    "rf-model-node",
    node.typeClass,
    selected ? "selected" : "",
    related ? "related" : "",
    comparisonPaths?.size > 0 && comparisonPaths.has(node.path) ? "comparison-change" : "",
    comparisonPaths?.size > 0 && !comparisonPaths.has(node.path) ? "comparison-stable" : "",
    searchActive && !isMatch ? "dimmed" : "",
    isOpenGroup ? "open-group" : "closed-group",
  ].filter(Boolean).join(" ");
  return <div className={classes} style={{ width: node.width, height }} onMouseEnter={() => hover.onHover?.(node.path)} onMouseLeave={() => hover.onHover?.(null)} onClick={(event) => {
    if (event.target.closest("button, a, input, select, textarea")) return;
    onSelect(node.path);
  }}>
    <Handle id="target" type="target" position={verticalFlow ? Position.Top : Position.Left} className="rf-port" isConnectable={false} />
    <Handle id="source" type="source" position={verticalFlow ? Position.Bottom : Position.Right} className="rf-port" isConnectable={false} />
    <div className="rf-node-content">
      <div className="rf-node-header">
        <span className="rf-node-title" title={node.fullName}>{node.displayName}</span>
        {showGroupToggle && node.isCollapsible && <button type="button" className="layer-group-toggle" onClick={(event) => { event.stopPropagation(); onToggle(node.path); }} aria-label={node.isExpanded ? (english ? "Collapse" : "收起") : (english ? "Expand" : "展开")}>{node.isExpanded ? "−" : "+"}</button>}
      </div>
      <div className="rf-node-badges">
        {node.repeat && <span className="diagram-repeat">×{node.repeat}</span>}
        {node.node?.attributes?.range && <span className="diagram-range">{node.node.attributes.range}</span>}
        {node.node?.attributes?.formula_id && <span className="diagram-formula">{node.node.attributes.formula_id}</span>}
        {node.isCollapsible && <span className="diagram-children-count">{node.node.children.length} {english ? (node.node.children.length === 1 ? "child" : "children") : "个子模块"}</span>}
        {lensEnabled && nodeLens?.[node.path] && <span className="diagram-bound">{bound}</span>}
      </div>
      {!isOpenGroup && node.metaLines.length > 0 && <ul className="rf-node-meta">{node.metaLines.map((line) => <li key={line} title={line}>{line}</li>)}</ul>}
      {!isOpenGroup && lensValues.length > 0 && <div className="diagram-lens-values">{lensValues.map(({ id, text }) => <span key={id} className={`diagram-lens-value lens-${id}`}>{text}</span>)}</div>}
    </div>
  </div>;
}

function MsvGroupFrame({ data }) {
  const node = data.node;
  const verticalFlow = (data.depth || 0) > 1;
  const anchorStyle = data.edgeAnchorOffset == null ? undefined : { top: data.edgeAnchorOffset };
  return <div className={`rf-group-frame depth-${Math.min(data.depth || 0, 4)}`} title={data.label} onClick={(event) => {
    if (event.target.closest("button")) return;
    data.onSelect?.(node?.path);
  }}>
    <Handle id="target" type="target" position={verticalFlow ? Position.Top : Position.Left} className="rf-port rf-group-port" style={anchorStyle} isConnectable={false} />
    <Handle id="source" type="source" position={verticalFlow ? Position.Bottom : Position.Right} className="rf-port rf-group-port" style={anchorStyle} isConnectable={false} />
    <div className="rf-group-header">
      {data.showGroupToggle && node && <button type="button" className="layer-group-toggle" onClick={(event) => { event.stopPropagation(); data.onToggle?.(node.path); }} aria-label={data.english ? "Collapse" : "收起"}>−</button>}
      <strong>{data.label}</strong>
      {data.classLabel && <span className="rf-group-class">{data.classLabel}</span>}
    </div>
  </div>;
}

function MsvStageBand({ data }) {
  return <div className={`rf-stage-band stage-${data.stage}`}><span>{data.label}</span></div>;
}

const RF_NODE_TYPES = { msvNode: MsvNode, groupFrame: MsvGroupFrame, stageBand: MsvStageBand };
const RF_EDGE_TYPES = { msvEdge: MsvEdge, msvNativeEdge: MsvNativeEdge };

function edgeClassName(data) {
  return `rf-edge ${data?.kind || "dataflow"}${data?.evidence === "module-order" ? " module-order" : ""}${data?.evidence === "semantic-flow" ? " semantic-flow" : ""}${data?.related ? " related" : ""}`;
}

function edgeStyle(style, data) {
  const mainFlow = data?.evidence === "module-order" || data?.evidence === "semantic-flow";
  return { ...style, strokeWidth: data?.related ? 2.8 : data?.width, strokeDasharray: mainFlow ? undefined : "7 4" };
}

function MsvNativeEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, data }) {
  const [path] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={edgeStyle(style, data)} className={edgeClassName(data)} />;
}

function MsvEdge(props) {
  const { id, source, target, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, data } = props;
  const { route } = useSmartEdgePath({
    ...props,
    // A routed MLA side branch can contain several obstacle-avoidance
    // waypoints. Smoothstep keeps those turns legible instead of bending all
    // waypoints into a visually confusing Bezier loop.
    preset: "smoothstep",
    options: { gridRatio: 12, nodePadding: 8, borderRadius: 6 },
  });
  const [fallbackPath] = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: 6 });
  const path = route && route.kind !== "clear" ? route.svgPathString : fallbackPath;
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={edgeStyle(style, data)} className={edgeClassName(data)} />;
}

function ReactFlowCanvas({ graph, props }) {
  const { fitBounds, fitView, setCenter, setViewport, getNode, getNodes, getViewport, zoomTo } = useReactFlow();
  const lastZoom = useRef(props.zoom);
  const lastFitNonce = useRef(props.fitNonce);
  const lastModelKey = useRef(null);
  useEffect(() => {
    if (!props.scrollSync?.group || !props.scrollSyncId) return undefined;
    const entry = { setViewport: (viewport) => setViewport(viewport, { duration: 0 }) };
    props.scrollSync.group.set(props.scrollSyncId, entry);
    return () => props.scrollSync.group.delete(props.scrollSyncId);
  }, [props.scrollSync, props.scrollSyncId, setViewport]);
  // Structure is represented by nested containers. Only execution/dataflow
  // edges are drawn, so parent-child containment cannot be mistaken for data.
  const renderEdges = useMemo(() => graph.edges.filter((edge) => edge.kind === "dataflow"), [graph.edges]);
  const matched = props.matchedPaths instanceof Set ? props.matchedPaths : EMPTY_SET;
  const activeRelationPath = props.externalHoveredPath ?? props.hoveredPath ?? props.selectedPath;
  const selectNode = useCallback((path) => {
    const modelNode = graph.nodes.find((node) => node.path === path);
    props.onSelectNode?.(path);
    if (modelNode?.isCollapsible && !modelNode.isExpanded) props.onToggleGroup?.(path);
  }, [graph.nodes, props.onSelectNode, props.onToggleGroup]);
  const relatedDataflowEdges = useMemo(() => relatedDataflowEdgeIds(graph.edges, activeRelationPath), [graph.edges, activeRelationPath]);
  const nodes = useMemo(() => {
    const nodeByPath = new Map(graph.nodes.map((node) => [node.path, node]));
    const frameByPath = new Map(graph.containerFrames.map((frame) => [frame.id, frame]));
    const nearestFrame = (path) => {
      let current = parentPath(path);
      while (current) {
        if (frameByPath.has(current)) return current;
        current = parentPath(current);
      }
      return null;
    };
    const originForFrame = (frameId) => {
      const frame = frameByPath.get(frameId);
      return frame ? { x: frame.x, y: frame.y } : { x: 0, y: 0 };
    };
    const frames = graph.containerFrames.map((frame) => {
      const parentFrame = nearestFrame(frame.id);
      const parentOrigin = originForFrame(parentFrame);
      return {
        id: `frame-${frame.id}`,
        type: "groupFrame",
        position: { x: frame.x - parentOrigin.x, y: frame.y - parentOrigin.y },
        parentId: parentFrame ? `frame-${parentFrame}` : undefined,
        width: frame.width,
        height: frame.height,
        measured: { width: frame.width, height: frame.height },
        style: { width: frame.width, height: frame.height },
        data: { ...frame, node: nodeByPath.get(frame.id), english: props.english, showGroupToggle: props.showGroupToggle, onSelect: selectNode, onToggle: props.onToggleGroup },
        selected: props.selectedPath === frame.id,
        selectable: true,
        draggable: false,
        connectable: false,
        zIndex: -10,
      };
    });
    const modelNodes = graph.nodes.filter((node) => !frameByPath.has(node.path)).map((node) => {
      const parentFrame = nearestFrame(node.path);
      const parentOrigin = originForFrame(parentFrame);
      const nodeHeight = node.isCollapsible && node.isExpanded ? 28 : node.height;
      return {
        id: node.path,
        type: "msvNode",
        position: { x: node.x - parentOrigin.x, y: node.y - parentOrigin.y },
        parentId: parentFrame ? `frame-${parentFrame}` : undefined,
        width: node.width,
        height: nodeHeight,
        measured: { width: node.width, height: nodeHeight },
        style: { width: node.width, height: nodeHeight },
        data: { node, english: props.english, showGroupToggle: props.showGroupToggle, onSelect: selectNode, onToggle: props.onToggleGroup, nodeLens: props.nodeLens, activeLenses: props.activeLenses, matched, searchActive: props.searchActive, comparisonPaths: props.comparisonPaths },
        selected: props.selectedPath === node.path,
        draggable: false,
      };
    });
    return [...frames, ...modelNodes];
  }, [graph, props.english, props.showGroupToggle, props.onToggleGroup, props.nodeLens, props.activeLenses, props.searchActive, props.comparisonPaths, props.selectedPath, matched, selectNode]);
  const edges = useMemo(() => {
    const framePaths = new Set(graph.containerFrames.map((frame) => frame.id));
    const targetId = (path) => framePaths.has(path) ? `frame-${path}` : path;
    return renderEdges.map((edge) => ({
      id: edge.id,
      source: targetId(edge.source),
      target: targetId(edge.target),
      sourceHandle: "source",
      targetHandle: "target",
      type: framePaths.has(edge.source) || framePaths.has(edge.target) ? "msvNativeEdge" : "msvEdge",
      markerEnd: DATAFLOW_MARKER,
      data: { ...edge, originalSource: edge.source, originalTarget: edge.target, flowDirection: (parentPath(edge.source)?.split(".").length || 0) > 1 ? "vertical" : "horizontal", related: relatedDataflowEdges.has(edge.id), width: edgeStrokeWidth(edge, graph.nodes.find((n) => n.path === edge.source)) },
    }));
  }, [renderEdges, relatedDataflowEdges, graph.nodes, graph.containerFrames]);
  const modelKey = graph.nodes.find((node) => node.path === "root")?.fullName || graph.nodes[0]?.fullName || "";
  // Fit once after the real ELK layout arrives, on model changes, or when the
  // user explicitly requests it. Expanding a nested module must preserve the
  // current viewport so the selected-module focus below can take over.
  useEffect(() => {
    if (!graph.layoutReady) return;
    const modelChanged = modelKey !== lastModelKey.current;
    const fitRequested = props.fitNonce !== lastFitNonce.current;
    if (!modelChanged && !fitRequested) return;
    lastModelKey.current = modelKey;
    lastFitNonce.current = props.fitNonce;
    fitView({ padding: 0.12, duration: 260 });
  }, [graph.layoutReady, modelKey, props.fitNonce, fitView]);
  useEffect(() => {
    if (props.zoom === lastZoom.current) return;
    const ratio = props.zoom / Math.max(lastZoom.current, 0.1);
    if (ratio > 0) {
      zoomTo(Math.max(0.1, Math.min(2.5, getViewport().zoom * ratio)), { duration: 120 });
    }
    lastZoom.current = props.zoom;
  }, [props.zoom, getViewport, zoomTo]);
  useEffect(() => {
    if (!props.selectedPath) return;
    const depth = props.selectedPath.split(".").length - 1;
    const timer = window.setTimeout(() => {
      const isFrame = graph.containerFrames.some((frame) => frame.id === props.selectedPath);
      const node = getNode(isFrame ? `frame-${props.selectedPath}` : props.selectedPath);
      if (!node) return;
      // Node positions are relative inside compound parents. Reconstruct the
      // absolute point from the parent chain instead of trusting a measured
      // absolute cache, which can still describe the previous layout during
      // an expand/collapse transition.
      let absoluteX = node.position.x;
      let absoluteY = node.position.y;
      let parentId = node.parentId;
      while (parentId) {
        const parent = getNode(parentId);
        if (!parent) break;
        absoluteX += parent.position.x;
        absoluteY += parent.position.y;
        parentId = parent.parentId;
      }
      // A deep compound module is the user's current reading context. Focus
      // that frame at a readable zoom instead of fitting all of its siblings,
      // which makes every operator card too small to inspect.
      if (isFrame && depth >= 2) {
        const box = absoluteNodeBox(node, getNode);
        void fitBounds({ x: box.x, y: box.y, width: box.width, height: box.height }, {
          duration: 260,
          padding: 0.14,
          minZoom: 0.5,
          maxZoom: 0.9,
        });
        return;
      }
      const siblings = getNodes().filter((candidate) => candidate.parentId === node.parentId && candidate.type !== "stageBand");
      if (siblings.length > 1) {
        const boxes = siblings.map((candidate) => absoluteNodeBox(candidate, getNode));
        const left = Math.min(...boxes.map((box) => box.x));
        const top = Math.min(...boxes.map((box) => box.y));
        const right = Math.max(...boxes.map((box) => box.x + box.width));
        const bottom = Math.max(...boxes.map((box) => box.y + box.height));
        void fitBounds({ x: left, y: top, width: right - left, height: bottom - top }, { padding: 0.16, duration: 260 });
        return;
      }
      setCenter(absoluteX + (node.measured?.width || node.width || 220) / 2, absoluteY + (node.measured?.height || node.height || 76) / 2, { duration: 260, zoom: depth >= 2 ? 1.05 : undefined });
    }, 360);
    return () => window.clearTimeout(timer);
  }, [props.selectedPath, graph, fitBounds, getNode, getNodes, setCenter]);
  function handleMove(_, viewport) {
    const group = props.scrollSync?.group;
    if (!group || !props.scrollSyncId || group.busy) return;
    group.busy = true;
    group.forEach((entry, id) => { if (id !== props.scrollSyncId) entry.setViewport(viewport); });
    group.busy = false;
  }
  const hoverContext = useMemo(() => ({ activeRelationPath, onHover: props.onHoverPathChange }), [activeRelationPath, props.onHoverPathChange]);
  return <HoverContext.Provider value={hoverContext}><SmartEdgeProvider nodes={nodes}>
    <ReactFlow nodes={nodes} edges={edges} nodeTypes={RF_NODE_TYPES} edgeTypes={RF_EDGE_TYPES} minZoom={0.1} maxZoom={2.5} onMove={handleMove} onNodeClick={(_, node) => selectNode(node.id.replace(/^frame-/, ""))} onEdgeClick={(_, edge) => { if (edge.data?.kind === "dataflow") selectNode(edge.data.originalTarget || edge.target.replace(/^frame-/, "")); }} onPaneClick={() => props.onHoverPathChange?.(null)}>
      <Background gap={20} size={1} color={props.english ? "#d7e1ea" : "#253042"} />
      <MiniMap pannable zoomable nodeColor={(node) => node.type === "groupFrame" ? "#8291a2" : "#93a0b2"} />
      <Controls position="top-left" showInteractive={false} />
    </ReactFlow>
  </SmartEdgeProvider></HoverContext.Provider>;
}

export default function ReactFlowStructureDiagram(props) {
  const baseGraph = useMemo(() => layoutGraph(props.structure.root, props.expandedGroups), [props.structure, props.expandedGroups]);
  const [graph, setGraph] = useState(baseGraph);
  const [hoveredPath, setHoveredPath] = useState(null);
  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || target?.isContentEditable) return;
      const selected = graph.nodes.find((node) => node.path === props.selectedPath);
      if (event.key === "Escape") {
        event.preventDefault();
        if (props.focusMode) props.onExitFocus?.();
        else props.onSelectNode?.(null);
      } else if (event.key === "0") {
        event.preventDefault();
        props.onFit?.();
      } else if ((event.key === "e" || event.key === "E" || event.key === "c" || event.key === "C") && selected?.isCollapsible) {
        const shouldExpand = event.key.toLowerCase() === "e";
        if (selected.isExpanded !== shouldExpand) {
          event.preventDefault();
          props.onToggleGroup?.(selected.path);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [graph.nodes, props]);
  useEffect(() => {
    let active = true;
    // Keep the current compound graph visible while ELK computes the next
    // one. Publishing the provisional graph first creates a visible second
    // layer during expand/collapse because React Flow measures both states.
    layoutGraphWithElk(baseGraph).then((next) => { if (active) setGraph(next); }).catch(() => {});
    return () => { active = false; };
  }, [baseGraph]);
  return <div className="diagram-frame react-flow-diagram" data-active-lenses={[...props.activeLenses].join(",")}><ReactFlowProvider><ReactFlowCanvas graph={graph} props={{ ...props, english: props.language === "en", hoveredPath, onHoverPathChange: (path) => { setHoveredPath(path); props.onHoverPathChange?.(path); } }} /></ReactFlowProvider></div>;
}
