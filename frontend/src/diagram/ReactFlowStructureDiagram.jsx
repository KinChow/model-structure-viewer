import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  useReactFlow,
} from "@xyflow/react";
import { layoutGraph } from "./layout.js";
import { layoutGraphWithElk } from "./elkLayout.js";
import { isGraphEdgeRelated, isPathRelated, relatedDataflowEdgeIds } from "./hover.js";
import { edgeStrokeWidth } from "./edgeStyle.js";

const EMPTY_SET = new Set();

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

function nodeGeometry(node) {
  const position = node?.internals?.positionAbsolute || node?.positionAbsolute || node?.position;
  return {
    x: position?.x || 0,
    y: position?.y || 0,
    width: node?.measured?.width || node?.width || 0,
    height: node?.measured?.height || node?.height || 0,
  };
}

function MsvNode({ data, selected }) {
  const { node, english, showGroupToggle, onSelect, onToggle, onHover, nodeLens, activeLenses, activeRelationPath, matched, searchActive, comparisonPaths } = data;
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
  return <div className={classes} style={{ width: node.width, height }} onMouseEnter={() => onHover?.(node.path)} onMouseLeave={() => onHover?.(null)} onClick={(event) => {
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
  return <div className={`rf-group-frame depth-${Math.min(data.depth || 0, 4)}`} title={data.label} onClick={(event) => {
    if (event.target.closest("button")) return;
    data.onSelect?.(node?.path);
  }}>
    <Handle id="target" type="target" position={verticalFlow ? Position.Top : Position.Left} className="rf-port rf-group-port" isConnectable={false} />
    <Handle id="source" type="source" position={verticalFlow ? Position.Bottom : Position.Right} className="rf-port rf-group-port" isConnectable={false} />
    <div className="rf-group-header">
      {data.showGroupToggle && node && <button type="button" className="layer-group-toggle" onClick={(event) => { event.stopPropagation(); data.onToggle?.(node.path); }} aria-label={data.english ? "Collapse" : "收起"}>−</button>}
      <strong>{data.label}</strong>
    </div>
  </div>;
}

function MsvStageBand({ data }) {
  return <div className={`rf-stage-band stage-${data.stage}`}><span>{data.label}</span></div>;
}

const RF_NODE_TYPES = { msvNode: MsvNode, groupFrame: MsvGroupFrame, stageBand: MsvStageBand };
const RF_EDGE_TYPES = { msvEdge: MsvEdge };

function MsvEdge({ source, target, markerEnd, style, data }) {
  const { getNode } = useReactFlow();
  const sourceNode = getNode(source);
  const targetNode = getNode(target);
  if (!sourceNode || !targetNode) return null;
  const sourceBox = nodeGeometry(sourceNode);
  const targetBox = nodeGeometry(targetNode);
  const vertical = data?.flowDirection === "vertical";
  const sourceFrame = source.startsWith("frame-");
  const targetFrame = target.startsWith("frame-");
  const headerOffset = 24;
  const geometry = vertical
    ? {
      sourceX: sourceBox.x + sourceBox.width / 2,
      sourceY: sourceBox.y + sourceBox.height,
      sourcePosition: Position.Bottom,
      targetX: targetBox.x + targetBox.width / 2,
      targetY: targetBox.y + (targetFrame ? headerOffset : 0),
      targetPosition: Position.Top,
    }
    : {
      sourceX: sourceBox.x + sourceBox.width,
      sourceY: sourceBox.y + (sourceFrame ? headerOffset : sourceBox.height / 2),
      sourcePosition: Position.Right,
      targetX: targetBox.x,
      targetY: targetBox.y + (targetFrame ? headerOffset : targetBox.height / 2),
      targetPosition: Position.Left,
    };
  const [fallback] = getBezierPath(geometry);
  const className = `rf-edge ${data?.kind || "dataflow"}${data?.evidence === "module-order" ? " module-order" : ""}${data?.related ? " related" : ""}`;
  return <BaseEdge path={fallback} markerEnd={data?.kind === "dataflow" ? markerEnd : undefined} style={{ ...style, strokeWidth: data?.related ? 2.8 : data?.width, strokeDasharray: data?.kind === "dataflow" && data?.evidence !== "module-order" ? "7 4" : undefined }} className={className} />;
}

function ReactFlowCanvas({ graph, props }) {
  const { fitView, setCenter, setViewport, getNode, getViewport, zoomTo } = useReactFlow();
  const lastZoom = useRef(props.zoom);
  useEffect(() => {
    if (!props.scrollSync?.group || !props.scrollSyncId) return undefined;
    const entry = { setViewport: (viewport) => setViewport(viewport, { duration: 0 }) };
    props.scrollSync.group.set(props.scrollSyncId, entry);
    return () => props.scrollSync.group.delete(props.scrollSyncId);
  }, [props.scrollSync, props.scrollSyncId, setViewport]);
  // Containment is the structure view. Drawing parent -> child edges on top
  // of nested frames duplicates that relationship and creates crossings.
  // The canvas therefore renders only sibling execution/dataflow edges.
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
      return {
        id: node.path,
        type: "msvNode",
        position: { x: node.x - parentOrigin.x, y: node.y - parentOrigin.y },
        parentId: parentFrame ? `frame-${parentFrame}` : undefined,
        style: { width: node.width, height: node.isCollapsible && node.isExpanded ? 28 : node.height },
        data: { node, english: props.english, showGroupToggle: props.showGroupToggle, onSelect: selectNode, onToggle: props.onToggleGroup, onHover: props.onHoverPathChange, nodeLens: props.nodeLens, activeLenses: props.activeLenses, activeRelationPath, matched, searchActive: props.searchActive, comparisonPaths: props.comparisonPaths },
        selected: props.selectedPath === node.path,
        draggable: false,
      };
    });
    return [...frames, ...modelNodes];
  }, [graph, props, activeRelationPath, matched, selectNode]);
  const edges = useMemo(() => {
    const framePaths = new Set(graph.containerFrames.map((frame) => frame.id));
    const targetId = (path) => framePaths.has(path) ? `frame-${path}` : path;
    return renderEdges.filter((edge) => props.edgeMode === "all" || edge.kind === props.edgeMode).map((edge) => ({
      id: edge.id,
      source: targetId(edge.source),
      target: targetId(edge.target),
      sourceHandle: "source",
      targetHandle: "target",
      type: "msvEdge",
      markerEnd: { type: MarkerType.ArrowClosed, color: edge.kind === "dataflow" ? "#d08a3a" : "#8291a2" },
      data: { ...edge, originalSource: edge.source, originalTarget: edge.target, flowDirection: (parentPath(edge.source)?.split(".").length || 0) > 1 ? "vertical" : "horizontal", related: edge.kind === "dataflow" ? relatedDataflowEdges.has(edge.id) : isGraphEdgeRelated(edge.source, edge.target, activeRelationPath), width: edgeStrokeWidth(edge, graph.nodes.find((n) => n.path === edge.source)), sections: edge.sections },
    }));
  }, [renderEdges, props.edgeMode, activeRelationPath, relatedDataflowEdges, graph.nodes, graph.containerFrames]);
  // ELK keeps the same node count while replacing the provisional positions;
  // fitting only on node-count changes leaves the canvas focused on the
  // provisional layout after the compound layout resolves.
  useEffect(() => {
    if (!graph.layoutReady) return;
    fitView({ padding: 0.12, duration: 260 });
  }, [props.fitNonce, graph, fitView]);
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
      let absoluteX = node.internals?.positionAbsolute?.x ?? node.position.x;
      let absoluteY = node.internals?.positionAbsolute?.y ?? node.position.y;
      if (!node.internals?.positionAbsolute) {
        let parentId = node.parentId;
        while (parentId) {
          const parent = getNode(parentId);
          if (!parent) break;
          absoluteX += parent.position.x;
          absoluteY += parent.position.y;
          parentId = parent.parentId;
        }
      }
      setCenter(absoluteX + (node.measured?.width || node.width || 220) / 2, absoluteY + (node.measured?.height || node.height || 76) / 2, { duration: 260, zoom: depth >= 2 ? 1.05 : undefined });
    }, 360);
    return () => window.clearTimeout(timer);
  }, [props.selectedPath, graph, getNode, setCenter]);
  function handleMove(_, viewport) {
    const group = props.scrollSync?.group;
    if (!group || !props.scrollSyncId || group.busy) return;
    group.busy = true;
    group.forEach((entry, id) => { if (id !== props.scrollSyncId) entry.setViewport(viewport); });
    group.busy = false;
  }
  return <ReactFlow nodes={nodes} edges={edges} nodeTypes={RF_NODE_TYPES} edgeTypes={RF_EDGE_TYPES} minZoom={0.1} maxZoom={2.5} onMove={handleMove} onNodeClick={(_, node) => selectNode(node.id.replace(/^frame-/, ""))} onEdgeClick={(_, edge) => { if (edge.data?.kind === "dataflow") selectNode(edge.data.originalTarget || edge.target.replace(/^frame-/, "")); }} onPaneClick={() => props.onHoverPathChange?.(null)}>
    <Background gap={20} size={1} color={props.english ? "#d7e1ea" : "#253042"} />
    <MiniMap pannable zoomable nodeColor={(node) => node.type === "groupFrame" ? "#8291a2" : "#93a0b2"} />
    <Controls position="top-left" showInteractive={false} />
  </ReactFlow>;
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
      } else if (["1", "2", "3"].includes(event.key)) {
        event.preventDefault();
        props.onEdgeModeChange?.({ 1: "all", 2: "structure", 3: "dataflow" }[event.key]);
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
