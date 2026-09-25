import { buildElkHierarchyEdges } from "./elkHierarchyEdges.js";
import { buildLayoutProjection, VIRTUAL_ROOT } from "./layoutProjection.js";

let elkPromise;

function getElk() {
  if (!elkPromise) {
    elkPromise = (typeof Worker === "undefined"
      ? import("elkjs/lib/elk.bundled.js")
      : import("elkjs/lib/elk-api.js")).then(({ default: Elk }) => {
      if (typeof Worker === "undefined") return new Elk();
      return new Elk({
        workerFactory: () => new Worker(new URL("elkjs/lib/elk-worker.min.js", import.meta.url), { type: "classic" }),
      });
    });
  }
  return elkPromise;
}

const BASE = {
  "elk.algorithm": "layered",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.spacing.nodeNode": "28",
  "elk.spacing.edgeNode": "18",
  "elk.spacing.edgeEdge": "10",
  "elk.layered.spacing.nodeNodeBetweenLayers": "44",
  "elk.layered.spacing.edgeNodeBetweenLayers": "18",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "10",
  "elk.layered.layering.strategy": "LONGEST_PATH",
};

function nodeType(node) {
  return String(node?.node?.type || node?.typeClass || "").toLowerCase();
}

function parentPath(path) {
  const index = path.lastIndexOf(".");
  return index < 0 ? null : path.slice(0, index);
}

function nodeHeight(node) {
  return node.isCollapsible && node.isExpanded ? 28 : node.height;
}

/**
 * Experimental all-ELK layout.
 *
 * Unlike the production compatibility layout, this function gives ELK one
 * compound graph containing the visible Graph IR and layout-only lanes. No
 * post-layout coordinate mutation and no secondary router are allowed here.
 * It is intentionally not wired into the UI until its POC gates pass.
 */
export async function layoutGraphWithElkOnly(graph) {
  const elk = await getElk();
  const projection = buildLayoutProjection(graph);
  const nodeByPath = new Map(graph.nodes.map((node) => [node.path, node]));
  const virtualById = new Map(projection.virtualNodes.map((node) => [node.id, node]));
  const parentByPath = projection.layoutParentByPath;
  const childrenByParent = projection.layoutChildrenByParent;
  const hierarchy = buildElkHierarchyEdges(graph, {
    layoutParentByPath: parentByPath,
    virtualNodeIds: projection.virtualNodes.map((node) => node.id),
  });

  const rawParent = (path) => parentByPath.get(path) ?? parentPath(path);
  const childrenOf = (owner) => childrenByParent.get(owner) || [];
  const isVirtual = (id) => virtualById.has(id);
  const directionOf = (id) => {
    if (id === "root") return "DOWN";
    if (isVirtual(id)) return "RIGHT";
    return "DOWN";
  };
  const sidesOf = (direction) => direction === "RIGHT"
    ? { in: "WEST", out: "EAST" }
    : { in: "NORTH", out: "SOUTH" };
  const portId = (path, direction) => `${path}::${direction}`;

  const directEdgesByOwner = new Map();
  const addDirect = (owner, edge) => {
    const list = directEdgesByOwner.get(owner) || [];
    list.push(edge);
    directEdgesByOwner.set(owner, list);
  };
  for (const edge of graph.edges || []) {
    if (edge.kind !== "dataflow" && edge.kind !== "module-order") continue;
    const sourceOwner = rawParent(edge.source);
    const targetOwner = rawParent(edge.target);
    if (sourceOwner && sourceOwner === targetOwner) addDirect(sourceOwner, edge);
  }
  for (const [owner, parts] of hierarchy.partsByOwner) {
    for (const part of parts) addDirect(owner, part);
  }

  const addPorts = (shape, id, parentDirection) => {
    const sides = sidesOf(parentDirection);
    const bridgePorts = (hierarchy.portsByNode.get(id) || []).map(({ id: bridgeId, direction: bridgeDirection }) => ({
      id: bridgeId,
      layoutOptions: { "elk.port.side": bridgeDirection === "in" ? sides.in : sides.out },
    }));
    shape.ports = [
      { id: portId(id, "in"), layoutOptions: { "elk.port.side": sides.in } },
      { id: portId(id, "out"), layoutOptions: { "elk.port.side": sides.out } },
      ...bridgePorts,
    ];
    shape.layoutOptions = {
      ...(shape.layoutOptions || {}),
      "elk.portConstraints": "FIXED_SIDE",
      "elk.portAlignment.default": "CENTER",
    };
    return shape;
  };

  const makeEdge = (edge) => {
    const source = edge.source.includes("::") ? edge.source : portId(edge.source, "out");
    const target = edge.target.includes("::") ? edge.target : portId(edge.target, "in");
    return { id: edge.id, sources: [source], targets: [target] };
  };

  function shapeFor(id) {
    const virtual = virtualById.get(id);
    const node = nodeByPath.get(id);
    const direction = directionOf(id);
    const children = childrenOf(id);
    const projectedEdges = directEdgesByOwner.get(id) || [];
    const realPairs = new Set(
      projectedEdges
        .filter((edge) => edge.source && edge.target && !edge.source.includes("::") && !edge.target.includes("::"))
        .map((edge) => `${edge.source}=>${edge.target}`),
    );
    const orderEdges = children.slice(0, -1)
      .map((source, index) => ({
        id: `__elk_order__${id}__${index}`,
        source,
        target: children[index + 1],
      }))
      .filter((edge) => !realPairs.has(`${edge.source}=>${edge.target}`));
    const childShapes = children.map((child) => shapeFor(child));
    const shape = {
      id,
      ...(node && !children.length ? {
        width: node.width,
        height: nodeHeight(node),
      } : {}),
      layoutOptions: {
        ...BASE,
        "elk.direction": direction,
        "elk.padding": virtual
          ? "[top=28,left=28,bottom=28,right=28]"
          : "[top=54,left=40,bottom=40,right=40]",
      },
      children: childShapes,
      edges: [
        ...projectedEdges.map(makeEdge),
        ...orderEdges.map(makeEdge),
      ],
    };
    addPorts(shape, id, directionOf(rawParent(id)));
    return shape;
  }

  const model = shapeFor("root");
  const result = await elk.layout({
    id: "__elk_only_root__",
    layoutOptions: { ...BASE, "elk.direction": "DOWN", "elk.padding": "24" },
    children: [model],
  });

  const positions = new Map();
  const dimensions = new Map();
  const frames = [];
  const routes = new Map();
  const walk = (shape, offsetX = 0, offsetY = 0) => {
    const x = offsetX + (shape.x || 0);
    const y = offsetY + (shape.y || 0);
    if (shape.id !== "__elk_only_root__" && shape.id !== VIRTUAL_ROOT) {
      positions.set(shape.id, { x, y });
      dimensions.set(shape.id, { width: shape.width || 0, height: shape.height || 0 });
    }
    if (shape.id !== "__elk_only_root__" && shape.edges) {
      for (const edge of shape.edges) {
        if (edge.id.startsWith("__elk_order__")) continue;
        const points = (edge.sections || []).flatMap((section) => [
          section.startPoint,
          ...(section.bendPoints || []),
          section.endPoint,
        ].filter(Boolean).map((point) => ({ x: x + point.x, y: y + point.y })));
        if (points.length > 1) routes.set(edge.id, points);
      }
    }
    if (shape.id !== "__elk_only_root__" && !isVirtual(shape.id) && shape.children?.length) {
      const node = nodeByPath.get(shape.id);
      frames.push({
        id: shape.id,
        x,
        y,
        width: shape.width,
        height: shape.height,
        label: shape.id === "root"
          ? "model"
          : `${node.displayName} · ${node.node?.type || "module"}${node.repeat > 1 ? ` · ×${node.repeat}` : ""}`,
        classLabel: shape.id === "root" ? (node.node?.attributes?.class || node.node?.name || null) : null,
        depth: node.depth,
        edgeAnchorOffset: node.depth === 1 ? 70 : null,
        kind: "graph-group",
      });
    }
    for (const child of shape.children || []) walk(child, x, y);
  };
  walk(result);

  const compoundRoutes = new Map();
  for (const [edgeId, segments] of hierarchy.segmentsByEdge) {
    const parts = segments.map((segment) => routes.get(segment));
    if (parts.some((part) => !part)) continue;
    compoundRoutes.set(edgeId, parts.flatMap((part, index) => index ? part.slice(1) : part));
  }

  return {
    ...graph,
    layoutReady: true,
    nodes: graph.nodes.map((node) => ({ ...node, ...(positions.get(node.path) || {}) })),
    edges: graph.edges.map((edge) => {
      const points = compoundRoutes.get(edge.id) || routes.get(edge.id);
      return points ? { ...edge, bendPoints: points.slice(1, -1), routePoints: points } : { ...edge };
    }),
    containerFrames: frames,
    layoutEngine: "elk-only-poc",
  };
}
