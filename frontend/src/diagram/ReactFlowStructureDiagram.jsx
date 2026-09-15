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
  useStore,
} from "@xyflow/react";
import { SmartEdgeProvider, useSmartEdgePath } from "@tisoap/react-flow-smart-edge";
import { layoutGraph } from "./layout.js";
import { layoutGraphWithElk } from "./elkLayout.js";
import { isPathRelated, relatedDataflowEdgeIds } from "./hover.js";
import { edgePresentation } from "./edgeStyle.js";
import { formatBytes, formatMetric } from "../formatters.js";
import { nodeBadges } from "./nodeBadges.js";
import { t } from "../i18n/format.js";

const EMPTY_SET = new Set();
const DATAFLOW_MARKER = { type: MarkerType.ArrowClosed, width: 10, height: 10, color: "#d08a3a" };
const PROGRAMMATIC_MIN_ZOOM = 0.35;
const PROGRAMMATIC_MAX_ZOOM = 1.2;
const HEADER_CENTER_Y = 14;
const HoverContext = createContext({ activeRelationPath: null, onHover: null });


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
  const bound = lensEnabled ? nodeLens?.[node.path]?.bound || null : null;
  const metrics = lensEnabled ? nodeLens?.[node.path]?.metrics || {} : {};
  const lensValues = [
    activeLenses.has("compute") && ["compute", "C", metrics.macsPerToken],
    activeLenses.has("memory") && ["memory", "M", metrics.memoryBytes, true],
    activeLenses.has("vram") && ["vram", "V", metrics.vramBytes, true],
  ].filter(Boolean).map(([id, label, value, bytes]) => ({
    id,
    text: value == null ? `${label} -` : `${label} ${bytes ? formatBytes(value, { includeKib: true }) : formatMetric(value)}`,
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
  return <div className={classes} style={{ width: node.width, height }} data-bound={lensEnabled ? (bound || "unknown") : undefined} onMouseEnter={() => hover.onHover?.(node.path)} onMouseLeave={() => hover.onHover?.(null)} onClick={(event) => {
    if (event.target.closest("button, a, input, select, textarea")) return;
    onSelect(node.path);
  }}>
    <Handle id="target" type="target" position={verticalFlow ? Position.Top : Position.Left} className="rf-port" isConnectable={false} />
    <Handle id="source" type="source" position={verticalFlow ? Position.Bottom : Position.Right} className="rf-port" isConnectable={false} />
    <div className="rf-node-content">
      <div className="rf-node-header">
        <span className="rf-node-title" title={node.fullName}>{node.displayName}</span>
        {showGroupToggle && node.isCollapsible && <button type="button" className="layer-group-toggle" onClick={(event) => { event.stopPropagation(); onToggle(node.path); }} aria-label={node.isExpanded ? t(english ? "en" : "zh", "diagram.collapse") : t(english ? "en" : "zh", "diagram.expand")}>{node.isExpanded ? "−" : "+"}</button>}
      </div>
      <div className="rf-node-badges">
        {nodeBadges(node.node || node, english ? "en" : "zh").map((badge) => (
          <span key={badge.kind} className={badge.kind === "children" ? "diagram-children-count" : "diagram-repeat"}>{badge.text}</span>
        ))}
        {node.node?.attributes?.range && <span className="diagram-range">{node.node.attributes.range}</span>}
        {node.node?.attributes?.operator_id && <span className="diagram-formula">{node.node.attributes.operator_id}</span>}
        {/* M11-P1-7：bound=unknown 显式呈现（虚线灰徽标），不再以"不渲染"冒充未开 Lens */}
        {lensEnabled && (activeLenses.has("compute") || activeLenses.has("memory")) && bound && (bound !== "unknown"
          ? <span className="diagram-bound">{bound}</span>
          : <span className="diagram-bound diagram-bound-unknown" title={t(english ? "en" : "zh", "diagram.boundUnknown")}>{t(english ? "en" : "zh", "diagram.boundUnknownShort")}</span>)}
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
      <strong>{data.label}</strong>
      {data.classLabel && <span className="rf-group-class">{data.classLabel}</span>}
      {data.showGroupToggle && node && <button type="button" className="layer-group-toggle" onClick={(event) => { event.stopPropagation(); data.onToggle?.(node.path); }} aria-label={t(data.english ? "en" : "zh", "diagram.collapse")}>−</button>}
    </div>
  </div>;
}

function MsvStageBand({ data }) {
  return <div className={`rf-stage-band stage-${data.stage}`}><span>{data.label}</span></div>;
}

const RF_NODE_TYPES = { msvNode: MsvNode, groupFrame: MsvGroupFrame, stageBand: MsvStageBand };
const RF_EDGE_TYPES = { msvEdge: MsvEdge, msvNativeEdge: MsvNativeEdge };

function edgeClassName(data) {
  // §2.2：类名由 edgeStyle.edgePresentation 统一决策（declared 实线；
  // module-order / shape-match 推断弱化）。semantic-flow 已退役。
  return `rf-edge ${data?.kind || "dataflow"}${data?.presentationClass || ""}${data?.related ? " related" : ""}`;
}

function edgeStyle(style, data) {
  // 虚线/实线由 CSS 按 evidence class 决定，此处只管宽度和关联态。
  return { ...style, strokeWidth: data?.related ? 2.8 : data?.width };
}

function MsvNativeEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, data }) {
  const [path] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={edgeStyle(style, data)} className={edgeClassName(data)} data-evidence={data?.evidence}><title>{data?.hint}</title></BaseEdge>;
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
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={edgeStyle(style, data)} className={edgeClassName(data)} data-evidence={data?.evidence}><title>{data?.hint}</title></BaseEdge>;
}

function reactFlowId(path, containerFrames) {
  return containerFrames.some((frame) => frame.id === path) ? `frame-${path}` : path;
}

function boxInViewport(box, viewport, pane) {
  const width = pane?.width || 0;
  const height = pane?.height || 0;
  if (!width || !height || !box.width || !box.height) return false;
  const left = box.x * viewport.zoom + viewport.x;
  const top = box.y * viewport.zoom + viewport.y;
  const right = left + box.width * viewport.zoom;
  const bottom = top + box.height * viewport.zoom;
  return left >= 0 && top >= 0 && right <= width && bottom <= height;
}

function ReactFlowCanvas({ graph, props }) {
  const { fitView, setCenter, setViewport, getNode, getViewport, zoomTo } = useReactFlow();
  const paneWidth = useStore((state) => state.width);
  const paneHeight = useStore((state) => state.height);
  const lastZoom = useRef(props.zoom);
  const lastFitNonce = useRef(props.fitNonce);
  const lastModelKey = useRef(null);
  const lastFocusedPath = useRef(null);
  const lastFocusedSignature = useRef(null);
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
    return renderEdges.map((edge) => {
      const presentation = edgePresentation(edge, graph.nodes.find((n) => n.path === edge.source), { english: props.english });
      return {
        id: edge.id,
        source: targetId(edge.source),
        target: targetId(edge.target),
        sourceHandle: "source",
        targetHandle: "target",
        type: framePaths.has(edge.source) || framePaths.has(edge.target) ? "msvNativeEdge" : "msvEdge",
        markerEnd: DATAFLOW_MARKER,
        // data-evidence：测试与调试的数据契约（W6-2 e2e 依赖）
        data: { ...edge, evidence: presentation.evidence, originalSource: edge.source, originalTarget: edge.target, flowDirection: (parentPath(edge.source)?.split(".").length || 0) > 1 ? "vertical" : "horizontal", related: relatedDataflowEdges.has(edge.id), width: presentation.width, presentationClass: presentation.className, hint: presentation.hint },
      };
    });
  }, [renderEdges, relatedDataflowEdges, graph.nodes, graph.containerFrames, props.english]);
  const modelKey = graph.nodes.find((node) => node.path === "root")?.fullName || graph.nodes[0]?.fullName || "";
  const layoutSignature = useMemo(
    () => [...graph.containerFrames, ...graph.nodes].map((node) => `${node.path || node.id}:${node.x || 0}:${node.y || 0}:${node.width || 0}:${node.height || 0}`).join("|"),
    [graph.containerFrames, graph.nodes],
  );
  useEffect(() => {
    if (!graph.layoutReady) return;
    const modelChanged = modelKey !== lastModelKey.current;
    const fitRequested = props.fitNonce !== lastFitNonce.current;
    if (!modelChanged && !fitRequested) return;
    lastModelKey.current = modelKey;
    lastFitNonce.current = props.fitNonce;
    lastFocusedPath.current = props.selectedPath;
    lastFocusedSignature.current = layoutSignature;
    void fitView({
      padding: 0.12,
      duration: 260,
      minZoom: PROGRAMMATIC_MIN_ZOOM,
      maxZoom: PROGRAMMATIC_MAX_ZOOM,
    });
  }, [graph.layoutReady, modelKey, props.fitNonce, layoutSignature, fitView]);
  useEffect(() => {
    if (props.zoom === lastZoom.current) return;
    const ratio = props.zoom / Math.max(lastZoom.current, 0.1);
    if (ratio > 0) {
      zoomTo(Math.max(0.1, Math.min(2.5, getViewport().zoom * ratio)), { duration: 120 });
    }
    lastZoom.current = props.zoom;
  }, [props.zoom, getViewport, zoomTo]);
  useEffect(() => {
    if (!graph.layoutReady || !props.selectedPath) return;
    if (props.fitNonce !== lastFitNonce.current) return;
    const alreadyFocused = lastFocusedPath.current === props.selectedPath && lastFocusedSignature.current === layoutSignature;
    if (alreadyFocused) return;
    const node = getNode(reactFlowId(props.selectedPath, graph.containerFrames));
    if (!node) return;
    const box = absoluteNodeBox(node, getNode);
    if (!box.width || !box.height) return;
    lastFocusedPath.current = props.selectedPath;
    lastFocusedSignature.current = layoutSignature;
    const viewport = getViewport();
    const headerBox = { x: box.x, y: box.y, width: box.width, height: Math.min(box.height, HEADER_CENTER_Y * 2) };
    if (boxInViewport(headerBox, viewport, { width: paneWidth, height: paneHeight })) return;
    setCenter(box.x + box.width / 2, box.y + HEADER_CENTER_Y, { duration: 200, zoom: viewport.zoom });
  }, [graph.layoutReady, layoutSignature, props.selectedPath, props.fitNonce, getNode, getViewport, setCenter, graph.containerFrames, paneWidth, paneHeight]);
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
  const baseGraph = useMemo(() => layoutGraph(props.structure, props.expandedGroups), [props.structure, props.expandedGroups]);
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
  return <div className="diagram-frame react-flow-diagram" data-graph-version={graph.graphVersion || "legacy"} data-active-lenses={[...props.activeLenses].join(",")}><ReactFlowProvider><ReactFlowCanvas graph={graph} props={{ ...props, english: props.language === "en", hoveredPath, onHoverPathChange: (path) => { setHoveredPath(path); props.onHoverPathChange?.(path); } }} /></ReactFlowProvider></div>;
}
