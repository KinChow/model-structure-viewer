/** Project the graph node index back to the legacy hierarchy view. */
export function projectGraphToTree(graph) {
  const byId = new Map((graph?.nodes || []).map((node) => [node.id, {
    id: node.canonical_id || node.module_id || node.id,
    name: node.name || node.module_id || node.id,
    type: node.type || "module",
    repeat: node.repeat ?? undefined,
    attributes: node.attributes || {},
    source_fields: node.source_fields || [],
    confidence: node.confidence || "high",
    children: [],
    params: node.params ?? undefined,
    weight_shapes: node.weight_shapes || undefined,
    dtype: node.dtype || undefined,
    input_shape: node.input_shape || undefined,
    output_shape: node.output_shape || undefined,
    value_source: node.value_source || undefined,
    tensor_names: node.tensor_names || undefined,
  }]));
  for (const node of [...(graph?.nodes || [])].sort((left, right) => (
    (left.parent_id || "").localeCompare(right.parent_id || "")
      || (left.order || 0) - (right.order || 0)
      || left.id.localeCompare(right.id)
  ))) {
    if (node.parent_id == null) continue;
    byId.get(node.parent_id)?.children.push(byId.get(node.id));
  }
  return byId.get(graph?.root_id || "root") || null;
}
