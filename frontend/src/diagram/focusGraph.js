import { layoutGraph } from "./layout.js";

function parentPath(path) {
  const index = path.lastIndexOf(".");
  return index > 0 ? path.slice(0, index) : null;
}

export function focusTopLevelPath(path) {
  if (!path || path === "root") return null;
  const parts = String(path).split(".");
  return parts.length >= 2 ? parts.slice(0, 2).join(".") : null;
}

function isDescendant(path, ancestor) {
  return path === ancestor || path.startsWith(`${ancestor}.`);
}

function nearestIncluded(path, included) {
  let current = path;
  const seen = new Set();
  while (current && !included.has(current)) {
    if (seen.has(current)) return null;
    seen.add(current);
    current = parentPath(current);
  }
  return current;
}

function syntheticBoundaryPath(index) {
  return `root.${9000 + index}`;
}

/**
 * Build a bounded detail graph for one top-level module.
 *
 * The overview remains the complete ELK graph. Focus mode deliberately keeps
 * only the selected module subtree plus synthetic boundary terminals. Thus a
 * deep expansion is solved locally and cannot move unrelated overview nodes.
 */
export function buildFocusedLayoutGraph(structure, expandedGroups, focusPath) {
  const focusRoot = focusTopLevelPath(focusPath);
  if (!focusRoot) return layoutGraph(structure, expandedGroups);

  const expanded = new Set(expandedGroups instanceof Set ? expandedGroups : []);
  expanded.add("root");
  expanded.add(focusRoot);
  const laidOut = layoutGraph(structure, expanded);
  const topLevelNodes = laidOut.nodes.filter((node) => node.path.split(".").length === 2);
  const topLevelPaths = new Set(topLevelNodes.map((node) => node.path));
  const included = new Set(
    laidOut.nodes
      .filter((node) => isDescendant(node.path, focusRoot))
      .map((node) => node.path),
  );
  const focusedNodes = laidOut.nodes.filter((node) => included.has(node.path));
  const focusedPaths = new Set(focusedNodes.map((node) => node.path));
  const boundaryNodes = new Map();
  let boundaryIndex = 0;

  const ensureBoundary = (topLevel, direction) => {
    if (!topLevelPaths.has(topLevel) || topLevel === focusRoot) return null;
    const key = `${topLevel}:${direction}`;
    if (!boundaryNodes.has(key)) {
      const source = topLevelNodes.find((node) => node.path === topLevel);
      const path = syntheticBoundaryPath(boundaryIndex++);
      boundaryNodes.set(key, {
        ...source,
        path,
        fullName: source?.fullName || source?.displayName || topLevel,
        displayName: `${direction === "in" ? "input" : "output"} · ${source?.displayName || topLevel}`,
        metaLines: ["external module boundary"],
        width: Math.max(220, source?.width || 220),
        height: 64,
        depth: 1,
        isCollapsible: false,
        isExpanded: false,
        synthetic: true,
        boundaryDirection: direction,
        originalPath: topLevel,
      });
    }
    return boundaryNodes.get(key).path;
  };

  const projected = new Map();
  const addEdge = (edge, source, target) => {
    if (!source || !target || source === target || source === "root" || target === "root") return;
    const key = JSON.stringify([source, target, edge.kind, edge.relation, edge.label, edge.evidence]);
    const existing = projected.get(key);
    if (existing) {
      existing.originalEdges = [...(existing.originalEdges || [existing.id]), edge.id];
      return;
    }
    const changed = source !== edge.source || target !== edge.target;
    projected.set(key, changed
      ? {
        ...edge,
        source,
        target,
        originalSource: edge.originalSource || edge.source,
        originalTarget: edge.originalTarget || edge.target,
        originalEdges: [edge.id],
      }
      : { ...edge });
  };

  for (const edge of laidOut.edges) {
    const sourceInside = nearestIncluded(edge.source, focusedPaths);
    const targetInside = nearestIncluded(edge.target, focusedPaths);
    const sourceTop = focusTopLevelPath(edge.source);
    const targetTop = focusTopLevelPath(edge.target);
    const sourceExternal = sourceTop && sourceTop !== focusRoot && topLevelPaths.has(sourceTop);
    const targetExternal = targetTop && targetTop !== focusRoot && topLevelPaths.has(targetTop);
    if (targetInside && sourceExternal) {
      addEdge(edge, ensureBoundary(sourceTop, "in"), targetInside);
    } else if (sourceInside && targetExternal) {
      addEdge(edge, sourceInside, ensureBoundary(targetTop, "out"));
    } else if (sourceInside && targetInside) {
      addEdge(edge, sourceInside, targetInside);
    }
  }

  const rootNode = laidOut.nodes.find((node) => node.path === "root");
  const nodes = [rootNode, ...focusedNodes, ...boundaryNodes.values()].filter(Boolean);
  const nodePaths = new Set(nodes.map((node) => node.path));
  const containerFrames = laidOut.containerFrames.filter((frame) =>
    frame.id === "root" || nodePaths.has(frame.id),
  );
  return {
    ...laidOut,
    nodes,
    edges: [...projected.values()],
    containerFrames,
    focusPath: focusRoot,
    focusMode: true,
  };
}

const FOCUS_FRAME_X = 420;
const FOCUS_FRAME_Y = 120;
const FOCUS_INPUT_X = 80;
const FOCUS_OUTPUT_X = 1240;
const FOCUS_BOUNDARY_Y = 180;

function shiftPoints(points, dx, dy) {
  return points?.map((point) => ({ x: point.x + dx, y: point.y + dy }));
}

/**
 * Anchor a focused viewport after ELK has solved its local graph.
 *
 * This is a viewport normalization, not a second layout algorithm: the local
 * ELK geometry and routes are preserved, while the focused shell and external
 * terminals get stable canvas anchors. Expanding a descendant therefore does
 * not translate the final-norm/output terminal.
 */
export function normalizeFocusedElkGraph(graph) {
  if (!graph.focusMode) return graph;
  const frame = graph.containerFrames.find((candidate) => candidate.id === graph.focusPath);
  if (!frame) return graph;
  const dx = FOCUS_FRAME_X - frame.x;
  const dy = FOCUS_FRAME_Y - frame.y;
  const boundaryNodes = graph.nodes.filter((node) => node.synthetic);
  const incoming = boundaryNodes.filter((node) => node.boundaryDirection === "in");
  const outgoing = boundaryNodes.filter((node) => node.boundaryDirection === "out");
  const positions = new Map();
  incoming.forEach((node, index) => positions.set(node.path, {
    x: FOCUS_INPUT_X,
    y: FOCUS_BOUNDARY_Y + index * 90,
  }));
  outgoing.forEach((node, index) => positions.set(node.path, {
    x: FOCUS_OUTPUT_X,
    y: FOCUS_BOUNDARY_Y + index * 90,
  }));
  const nodes = graph.nodes
    .filter((node) => node.path !== "root")
    .map((node) => {
      const fixed = positions.get(node.path);
      return fixed
        ? { ...node, ...fixed }
        : { ...node, x: (node.x || 0) + dx, y: (node.y || 0) + dy };
    });
  const frames = graph.containerFrames
    .filter((candidate) => candidate.id !== "root")
    .map((candidate) => ({
      ...candidate,
      x: candidate.x + dx,
      y: candidate.y + dy,
    }));
  const frameAfter = frames.find((candidate) => candidate.id === graph.focusPath);
  const pointFor = (path, side) => {
    if (path === graph.focusPath && frameAfter) {
      return {
        x: frameAfter.x + (side === "out" ? frameAfter.width : 0),
        y: frameAfter.y + frameAfter.height / 2,
      };
    }
    const node = nodes.find((candidate) => candidate.path === path);
    if (node) {
      return {
        x: node.x + (side === "out" ? node.width : 0),
        y: node.y + node.height / 2,
      };
    }
    return null;
  };
  const edges = graph.edges.map((edge) => {
    const sourceBoundary = positions.has(edge.source);
    const targetBoundary = positions.has(edge.target);
    if (sourceBoundary || targetBoundary) {
      const source = pointFor(edge.source, "out");
      const target = pointFor(edge.target, "in");
      const routePoints = source && target ? [source, target] : undefined;
      return {
        ...edge,
        routePoints,
        bendPoints: routePoints ? [] : undefined,
      };
    }
    return {
      ...edge,
      routePoints: shiftPoints(edge.routePoints, dx, dy),
      bendPoints: shiftPoints(edge.bendPoints, dx, dy),
    };
  });
  return { ...graph, nodes, containerFrames: frames, edges };
}
