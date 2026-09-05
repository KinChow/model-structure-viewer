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

function isExternalRootNode(node) {
  const type = String(node.node?.type || node.typeClass || "").toLowerCase();
  const name = String(node.node?.name || node.displayName || "").toLowerCase();
  return type === "output" || type === "head" || /(^|[._ -])(lm[_ -]?head|classifier|score)$/.test(name);
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
  const directEdges = (path, allowedIds) => graph.edges
    .filter((edge) => (edge.kind === "structure" || edge.kind === "dataflow")
      && parentPath(edge.source) === path && parentPath(edge.target) === path
      && (!allowedIds || (allowedIds.has(edge.source) && allowedIds.has(edge.target))))
    .map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] }));

  function makeShape(node, depth) {
    const allChildren = directChildren(node, nodeByPath);
    const children = node.path === "root" ? allChildren.filter((child) => !isExternalRootNode(child)) : allChildren;
    const childIds = new Set(children.map((child) => child.path));
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
        // Keep the model's top-level modules in a readable pipeline. Once a
        // module is opened, its implementation is a vertical sibling flow.
        "elk.direction": depth === 0 ? "RIGHT" : "DOWN",
        "elk.padding": "[top=32,left=24,bottom=24,right=24]",
      },
      children: children.map((child) => makeShape(child, depth + 1)),
      edges: [...directEdges(node.path, childIds), ...orderEdges],
    };
  }

  // The model itself is the outermost compound node. Placing `root` beside
  // its children makes the model header look disconnected and loses the
  // parent/child containment that modelmap uses.
  const root = nodeByPath.get("root");
  const rootChildren = root ? directChildren(root, nodeByPath) : [];
  const externalRootChildren = rootChildren.filter(isExternalRootNode);
  const externalIds = new Set(externalRootChildren.map((child) => child.path));
  const modelShape = root ? makeShape(root, 0) : null;
  const topEdges = root
    ? graph.edges
      .filter((edge) => (edge.kind === "structure" || edge.kind === "dataflow")
        && rootChildren.some((child) => child.path === edge.source)
        && rootChildren.some((child) => child.path === edge.target)
        && (externalIds.has(edge.source) || externalIds.has(edge.target)))
      .map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] }))
    : [];
  const layoutRoot = root
    ? {
      id: "__graph_root__",
      layoutOptions: { ...BASE_LAYOUT, "elk.direction": "RIGHT", "elk.padding": "32" },
      children: [modelShape, ...externalRootChildren.map((child) => makeShape(child, 1))],
      edges: topEdges,
    }
    : { id: "__graph_root__", layoutOptions: { ...BASE_LAYOUT, "elk.direction": "RIGHT" }, children: [] };

  const result = await elk.layout(layoutRoot);
  // Cross-boundary edges can make ELK place an external head above the
  // compound model. The canvas root has a deliberate left-to-right contract:
  // keep the model container on the left and its external siblings to the
  // right, centered against the model's height.
  if (result.id === "__graph_root__") {
    const modelResult = result.children?.find((child) => child.id === "root");
    if (modelResult) {
      let externalX = (modelResult.x || 0) + (modelResult.width || 0) + 80;
      const modelY = modelResult.y || 0;
      for (const child of result.children || []) {
        if (child.id === "root") continue;
        child.x = externalX;
        child.y = modelY + Math.max(0, ((modelResult.height || 0) - (child.height || 0)) / 2);
        externalX += (child.width || 0) + 80;
      }
    }
  }
  // ELK centers short siblings against a large expanded compound node. That
  // is technically valid, but it pushes embedding/norm/head far below the
  // container headers and makes the top-level execution chain look broken.
  // Keep the root pipeline on one baseline; nested containers retain ELK's
  // own placement.
  const modelLayout = result.id === "root" ? result : result.children?.find((child) => child.id === "root");
  if (modelLayout?.children?.length) {
    const topLevelY = Math.min(...modelLayout.children.map((child) => child.y || 0));
    for (const child of modelLayout.children) child.y = topLevelY;
  }
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
        label: node.id === "root"
          ? "model"
          : `${node.displayName} · ${node.node?.type || "module"}${node.repeat > 1 ? ` · ×${node.repeat}` : ""}`,
        classLabel: node.id === "root" ? (node.node?.attributes?.class || node.node?.name || null) : null,
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
    layoutReady: true,
    nodes: graph.nodes.map((node) => ({ ...node, ...(positions.get(node.path) || {}) })),
    edges: graph.edges.map((edge) => ({ ...edge, sections: routedEdges.get(edge.id) || [] })),
    containerFrames: groupFrames,
  };
}
