export function childRepeatMultiplier(node, inheritedMultiplier = 1, { repeatHandled = false } = {}) {
  const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
  const childHasExplicitRepeat = (node?.children || []).some((child) => Number.isFinite(child?.repeat));
  return inheritedMultiplier * (repeatHandled || childHasExplicitRepeat ? 1 : repeat);
}

export function graphNodeToNode(graphNode) {
  return {
    id: graphNode.canonical_id || graphNode.module_id || graphNode.id,
    name: graphNode.name || graphNode.module_id || graphNode.id,
    type: graphNode.type || "module",
    repeat: graphNode.repeat ?? undefined,
    attributes: graphNode.attributes || {},
    source_fields: graphNode.source_fields || [],
    confidence: graphNode.confidence || "high",
    children: [],
    params: graphNode.params ?? undefined,
    weight_shapes: graphNode.weight_shapes || undefined,
    dtype: graphNode.dtype || undefined,
    input_shape: graphNode.input_shape || undefined,
    output_shape: graphNode.output_shape || undefined,
    value_source: graphNode.value_source || undefined,
    tensor_names: graphNode.tensor_names || undefined,
  };
}

export function walkGraph(graph, visit) {
  if (!graph?.nodes?.length) return;
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const childrenByParent = new Map();
  for (const node of graph.nodes) {
    if (node.parent_id == null) continue;
    const children = childrenByParent.get(node.parent_id) || [];
    children.push(node);
    childrenByParent.set(node.parent_id, children);
  }
  for (const children of childrenByParent.values()) {
    children.sort((left, right) => (left.order || 0) - (right.order || 0) || left.id.localeCompare(right.id));
  }
  function walk(nodeId, multiplier = 1) {
    const graphNode = byId.get(nodeId);
    if (!graphNode) return;
    const children = childrenByParent.get(nodeId) || [];
    // Keep the child index available to cost ownership decisions. Graph IR
    // remains the source of truth; this shallow projection is only the
    // traversal view used by the cost calculators.
    const node = {
      ...graphNodeToNode(graphNode),
      children: children.map(graphNodeToNode),
    };
    visit({ node, path: graphNode.id, multiplier });
    const childHasExplicitRepeat = children.some((child) => Number.isFinite(child.repeat));
    const childMultiplier = childRepeatMultiplier(node, multiplier, { repeatHandled: childHasExplicitRepeat });
    for (const child of children) walk(child.id, childMultiplier);
  }
  walk(graph.root_id || graph.nodes.find((node) => node.parent_id == null)?.id || "root");
}

export function walkStructure(root, visit, graph = null) {
  if (graph?.nodes?.length) {
    walkGraph(graph, visit);
    return;
  }
  function walk(node, path = "root", multiplier = 1) {
    visit({ node, path, multiplier });
    const childMultiplier = childRepeatMultiplier(node, multiplier);
    (node?.children || []).forEach((child, index) => {
      walk(child, `${path}.${index}`, childMultiplier);
    });
  }
  if (root) walk(root);
}
