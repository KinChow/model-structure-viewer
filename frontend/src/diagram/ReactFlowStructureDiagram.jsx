import { useEffect, useMemo, useRef, useState } from "react";
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
  getSmoothStepPath,
  useReactFlow,
} from "@xyflow/react";
import { layoutGraph } from "./layout.js";
import { layoutGraphWithElk } from "./elkLayout.js";
import { isGraphEdgeRelated, isPathRelated, relatedDataflowEdgeIds } from "./hover.js";
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

function edgePathFromSections(sections) {
  const section = sections?.[0];
  if (!section?.startPoint || !section?.endPoint) return null;
  const points = [section.startPoint, ...(section.bendPoints || []), section.endPoint];
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
}

function MsvNode({ data, selected }) {
  const { node, english, showGroupToggle, onSelect, onToggle, nodeLens, activeLenses, activeRelationPath, matched, searchActive } = data;
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
    searchActive && !isMatch ? "dimmed" : "",
    isOpenGroup ? "open-group" : "closed-group",
  ].filter(Boolean).join(" ");
  return <div className={classes} style={{ width: node.width, height }} onClick={(event) => {
    if (event.target.closest("button, a, input, select, textarea")) return;
    onSelect(node.path);
  }}>
    <Handle type="target" position={Position.Left} className="rf-port" isConnectable={false} />
    <Handle type="source" position={Position.Right} className="rf-port" isConnectable={false} />
    <Handle type="target" position={Position.Top} className="rf-port" isConnectable={false} />
    <Handle type="source" position={Position.Bottom} className="rf-port" isConnectable={false} />
    <div className="rf-node-content">
      <div className="rf-node-header">
        <span className="rf-node-title" title={node.fullName}>{node.displayName}</span>
        {showGroupToggle && node.isCollapsible && <button type="button" className="layer-group-toggle" onClick={(event) => { event.stopPropagation(); onToggle(node.path); }} aria-label={node.isExpanded ? (english ? "Collapse" : "收起") : (english ? "Expand" : "展开")}>{node.isExpanded ? "−" : "+"}</button>}
      </div>
      <div className="rf-node-badges">
        {node.repeat && <span className="diagram-repeat">×{node.repeat}</span>}
        {node.node?.attributes?.range && <span className="diagram-range">{node.node.attributes.range}</span>}
        {node.node?.attributes?.formula_id && <span className="diagram-formula">{node.node.attributes.formula_id}</span>}
        {node.isCollapsible && <span className="diagram-children-count">{node.node.children.length} {node.node.children.length === 1 ? "child" : "children"}</span>}
        {lensEnabled && nodeLens?.[node.path] && <span className="diagram-bound">{bound}</span>}
      </div>
      {!isOpenGroup && node.metaLines.length > 0 && <ul className="rf-node-meta">{node.metaLines.map((line) => <li key={line} title={line}>{line}</li>)}</ul>}
      {!isOpenGroup && lensValues.length > 0 && <div className="diagram-lens-values">{lensValues.map(({ id, text }) => <span key={id} className={`diagram-lens-value lens-${id}`}>{text}</span>)}</div>}
    </div>
  </div>;
}

function MsvGroupFrame({ data }) {
  return <div className="rf-group-frame" title={data.label}><strong>{data.label}</strong></div>;
}

function MsvStageBand({ data }) {
  return <div className={`rf-stage-band stage-${data.stage}`}><span>{data.label}</span></div>;
}

const RF_NODE_TYPES = { msvNode: MsvNode, groupFrame: MsvGroupFrame, stageBand: MsvStageBand };
const RF_EDGE_TYPES = { msvEdge: MsvEdge };

function MsvEdge({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, data }) {
  const route = edgePathFromSections(data?.sections);
  const [fallback] = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const className = `rf-edge ${data?.kind || "dataflow"}${data?.evidence === "module-order" ? " module-order" : ""}${data?.related ? " related" : ""}`;
  return <BaseEdge path={route || fallback} markerEnd={data?.kind === "dataflow" ? markerEnd : undefined} style={{ ...style, strokeWidth: data?.related ? 2.8 : data?.width, strokeDasharray: data?.kind === "dataflow" && data?.evidence !== "module-order" ? "7 4" : undefined }} className={className} />;
}

function ReactFlowCanvas({ graph, props }) {
  const { fitView, setCenter, getNode, zoomIn, zoomOut } = useReactFlow();
  const lastZoom = useRef(props.zoom);
  const renderEdges = useMemo(() => graph.edges.filter((edge) => edge.kind !== "structure" || edge.source === "root" || edge.evidence === "module-order"), [graph.edges]);
  const matched = props.matchedPaths instanceof Set ? props.matchedPaths : new Set();
  const activeRelationPath = props.externalHoveredPath ?? props.hoveredPath ?? props.selectedPath;
  const selectNode = (path) => {
    const modelNode = graph.nodes.find((node) => node.path === path);
    props.onSelectNode?.(path);
    if (modelNode?.isCollapsible && !modelNode.isExpanded) props.onToggleGroup?.(path);
  };
  const relatedDataflowEdges = useMemo(() => relatedDataflowEdgeIds(graph.edges, activeRelationPath), [graph.edges, activeRelationPath]);
  const nodes = useMemo(() => {
    const stageBands = ["input", "representation", "decoder", "output"].flatMap((stage) => {
      const members = graph.nodes.filter((node) => node.stage === stage);
      if (!members.length) return [];
      return [{ id: `stage-${stage}`, type: "stageBand", position: { x: Math.min(...members.map((n) => n.x)) - 24, y: 0 }, style: { width: Math.max(...members.map((n) => n.x + n.width)) - Math.min(...members.map((n) => n.x)) + 48, height: Math.max(...graph.nodes.map((n) => n.y + n.height)) + 48 }, data: { stage, label: `${stage[0].toUpperCase()}${stage.slice(1)} · ${members.length} node${members.length === 1 ? "" : "s"}` }, selectable: false, draggable: false, connectable: false, zIndex: -20 }];
    });
    const frames = graph.containerFrames.map((frame) => ({ id: `frame-${frame.id}`, type: "groupFrame", position: { x: frame.x, y: frame.y }, style: { width: frame.width, height: frame.height }, data: frame, selectable: false, draggable: false, connectable: false, zIndex: -10 }));
    const modelNodes = graph.nodes.map((node) => ({ id: node.path, type: "msvNode", position: { x: node.x, y: node.y }, style: { width: node.width, height: node.isCollapsible && node.isExpanded ? 28 : node.height }, data: { node, english: props.english, showGroupToggle: props.showGroupToggle, onSelect: selectNode, onToggle: props.onToggleGroup, nodeLens: props.nodeLens, activeLenses: props.activeLenses, activeRelationPath, matched, searchActive: props.searchActive }, selected: props.selectedPath === node.path, draggable: false }));
    return [...stageBands, ...frames, ...modelNodes];
  }, [graph, props, activeRelationPath, matched, selectNode]);
  const edges = useMemo(() => renderEdges.filter((edge) => props.edgeMode === "all" || edge.kind === props.edgeMode).map((edge) => ({ id: edge.id, source: edge.source, target: edge.target, type: "msvEdge", markerEnd: { type: MarkerType.ArrowClosed, color: edge.kind === "dataflow" ? "#d08a3a" : "#8291a2" }, data: { ...edge, related: edge.kind === "dataflow" ? relatedDataflowEdges.has(edge.id) : isGraphEdgeRelated(edge.source, edge.target, activeRelationPath), width: edgeStrokeWidth(edge, graph.nodes.find((n) => n.path === edge.source)), sections: edge.sections } })) , [renderEdges, props.edgeMode, activeRelationPath, relatedDataflowEdges, graph.nodes]);
  useEffect(() => { fitView({ padding: 0.12, duration: 260 }); }, [props.fitNonce, graph.nodes.length, fitView]);
  useEffect(() => {
    if (props.zoom === lastZoom.current) return;
    const ratio = props.zoom / Math.max(lastZoom.current, 0.1);
    if (ratio > 0) {
      if (ratio > 1) zoomIn({ factor: ratio });
      else zoomOut({ factor: 1 / ratio });
    }
    lastZoom.current = props.zoom;
  }, [props.zoom, zoomIn, zoomOut]);
  useEffect(() => {
    if (!props.selectedPath) return;
    const node = getNode(props.selectedPath);
    if (!node) return;
    setCenter(node.position.x + (node.measured?.width || node.width || 220) / 2, node.position.y + (node.measured?.height || node.height || 76) / 2, { duration: 260 });
  }, [props.selectedPath, getNode, setCenter]);
  return <ReactFlow nodes={nodes} edges={edges} nodeTypes={RF_NODE_TYPES} edgeTypes={RF_EDGE_TYPES} fitView minZoom={0.1} maxZoom={2.5} onNodeClick={(_, node) => selectNode(node.id)} onPaneClick={() => props.onHoverPathChange?.(null)}>
    <Background gap={20} size={1} color={props.english ? "#d7e1ea" : "#253042"} />
    <MiniMap pannable zoomable nodeColor={(node) => node.type === "groupFrame" ? "#8291a2" : "#93a0b2"} />
    <Controls showInteractive={false} />
  </ReactFlow>;
}

export default function ReactFlowStructureDiagram(props) {
  const baseGraph = useMemo(() => layoutGraph(props.structure.root, props.expandedGroups), [props.structure, props.expandedGroups]);
  const [graph, setGraph] = useState(baseGraph);
  const hoveredPath = useRef(null);
  useEffect(() => {
    let active = true;
    setGraph(baseGraph);
    layoutGraphWithElk(baseGraph).then((next) => { if (active) setGraph(next); }).catch(() => {});
    return () => { active = false; };
  }, [baseGraph]);
  return <div className="diagram-frame react-flow-diagram" data-active-lenses={[...props.activeLenses].join(",")}><ReactFlowProvider><ReactFlowCanvas graph={graph} props={{ ...props, english: props.language === "en", hoveredPath: hoveredPath.current }} /></ReactFlowProvider></div>;
}
