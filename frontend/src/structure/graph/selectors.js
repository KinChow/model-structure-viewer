function graphNodes(graph) {
  return Array.isArray(graph?.nodes) ? graph.nodes : [];
}

export function graphNodeAt(graph, path) {
  if (!path) return null;
  return graphNodes(graph).find((node) => node.id === path) || null;
}

export function graphChildren(graph, parentId) {
  return graphNodes(graph)
    .filter((node) => node.parent_id === parentId)
    .sort((left, right) => (left.order || 0) - (right.order || 0) || left.id.localeCompare(right.id));
}

export function graphViewNode(graph, path) {
  const node = typeof path === "string" ? graphNodeAt(graph, path) : path;
  if (!node) return null;
  return {
    ...node,
    path: node.id,
    id: node.canonical_id || node.module_id || node.id,
    children: graphChildren(graph, node.id).map((child) => graphViewNode(graph, child)),
  };
}

export function graphRoot(graph) {
  return graphViewNode(graph, graph?.root_id || "root");
}

export function graphPaths(graph) {
  return graphNodes(graph).map((node) => node.id);
}
