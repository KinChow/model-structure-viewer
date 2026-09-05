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
const BASE_LAYOUT = {
  "elk.algorithm": "layered",
  "elk.layered.spacing.nodeNodeBetweenLayers": "44",
  "elk.spacing.nodeNode": "24",
};

function parentPath(path) {
  const index = path.lastIndexOf(".");
  return index < 0 ? null : path.slice(0, index);
}

function directChildren(node, nodeByPath) {
  if (!node.isExpanded) return [];
  return (node.node?.children || [])
    .map((_, index) => nodeByPath.get(`${node.path}.${index}`))
    .filter(Boolean);
}

function layoutHeight(node) {
  return node.isCollapsible && node.isExpanded ? 28 : node.height;
}

/**
 * Compound graph layout: top-level modules read left-to-right while module
 * internals read top-to-bottom, keeping the canvas graph-first and readable.
 */
export async function layoutGraphWithElk(graph) {
  const elk = await getElk();
  const nodeByPath = new Map(graph.nodes.map((node) => [node.path, node]));
  const directEdges = (path) => graph.edges
    .filter((edge) => (edge.kind === "structure" || edge.kind === "dataflow")
      && parentPath(edge.source) === path && parentPath(edge.target) === path)
    .map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] }));

  function makeShape(node, depth) {
    const children = directChildren(node, nodeByPath);
    if (children.length === 0) return { id: node.path, width: node.width, height: layoutHeight(node) };
    const orderEdges = children.slice(0, -1).map((child, index) => ({
      id: `__order__${node.path}__${index}`,
      sources: [child.path],
      targets: [children[index + 1].path],
    }));
    return {
      id: node.path,
      layoutOptions: {
        ...BASE_LAYOUT,
        "elk.direction": "DOWN",
        "elk.padding": "[top=32,left=24,bottom=24,right=24]",
      },
      children: children.map((child) => makeShape(child, depth + 1)),
      edges: [...directEdges(node.path), ...orderEdges],
    };
  }

  const root = nodeByPath.get("root");
  const topChildren = root ? directChildren(root, nodeByPath) : [];
  const topEdges = graph.edges
    .filter((edge) => (edge.kind === "structure" && edge.source === "root" && edge.target !== "root")
      || (edge.kind === "dataflow" && parentPath(edge.source) === "root" && parentPath(edge.target) === "root"))
    .map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] }));
  const layoutRoot = {
    id: "__graph_root__",
    layoutOptions: { ...BASE_LAYOUT, "elk.direction": "RIGHT", "elk.padding": "32" },
    children: [root, ...topChildren].filter(Boolean).map((node) => node.path === "root"
      ? { id: "root", width: node.width, height: layoutHeight(node) }
      : makeShape(node, 1)),
    edges: topEdges,
  };

  const result = await elk.layout(layoutRoot);
  const positions = new Map();
  const routedEdges = new Map();
  const groupFrames = [];
  function walk(shape, offsetX = 0, offsetY = 0) {
    const x = offsetX + (shape.x || 0);
    const y = offsetY + (shape.y || 0);
    if (shape.id !== "__graph_root__") positions.set(shape.id, { x, y });
    if (shape.id !== "__graph_root__" && shape.children?.length && nodeByPath.has(shape.id)) {
      const node = nodeByPath.get(shape.id);
      groupFrames.push({
        id: shape.id,
        x: x - 16,
        y: y - 22,
        width: shape.width + 32,
        height: shape.height + 38,
        label: `${node.displayName} · ${node.node?.type || "module"}${node.repeat > 1 ? ` · ×${node.repeat}` : ""}`,
        depth: node.depth,
        kind: "graph-group",
      });
    }
    for (const edge of shape.edges || []) {
      routedEdges.set(edge.id, (edge.sections || []).map((section) => ({
        ...section,
        startPoint: { x: section.startPoint.x + offsetX, y: section.startPoint.y + offsetY },
        endPoint: { x: section.endPoint.x + offsetX, y: section.endPoint.y + offsetY },
        bendPoints: (section.bendPoints || []).map((point) => ({ x: point.x + offsetX, y: point.y + offsetY })),
      })));
    }
    for (const child of shape.children || []) walk(child, x, y);
  }
  walk(result);

  return {
    ...graph,
    nodes: graph.nodes.map((node) => ({ ...node, ...(positions.get(node.path) || {}) })),
    edges: graph.edges.map((edge) => ({ ...edge, sections: routedEdges.get(edge.id) || [] })),
    containerFrames: groupFrames,
  };
}
