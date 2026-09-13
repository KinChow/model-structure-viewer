export function childRepeatMultiplier(node, inheritedMultiplier = 1, { repeatHandled = false } = {}) {
  const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
  const childHasExplicitRepeat = (node?.children || []).some((child) => Number.isFinite(child?.repeat));
  return inheritedMultiplier * (repeatHandled || childHasExplicitRepeat ? 1 : repeat);
}

/** 驻留容量用的 repeat：`repeat=0` 仍占显存（原则 §3.8，对标 vLLM named_parameters）。
 *  一份模板 × N 读 `attributes.modules`；`mtp.0/1/2` 这种已展开 stage 不再乘。 */
export function residentRepeat(node) {
  const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
  if (repeat !== 0) return repeat;
  const children = node?.children || [];
  if (children.some((child) => Number.isFinite(child?.repeat))) return 1;
  const stages = children.filter((child) => /(?:^|\.)\d+$/.test(String(child.id || ""))).length;
  if (stages > 1) return 1;
  const modules = node?.attributes?.modules;
  return Number.isFinite(modules) && modules > 0 ? modules : 1;
}

export function childResidentRepeat(node, inheritedMultiplier = 1, { repeatHandled = false } = {}) {
  const childHasExplicitRepeat = (node?.children || []).some((child) => Number.isFinite(child?.repeat));
  return inheritedMultiplier * (repeatHandled || childHasExplicitRepeat ? 1 : residentRepeat(node));
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

// P7（步骤 7）：walkStructure 的 tree root 入参退役——Graph IR 是唯一遍历路径。
// 旧签名 walkStructure(root, visit, graph) 的树回退分支已删除（原 walkGraph
// 本体并入）；多倍率语义（childRepeatMultiplier / 显式子 repeat 压过父
// repeat）与旧图分支逐位一致。
export function walkStructure(graph, visit) {
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
  function walk(nodeId, multiplier = 1, resident = 1) {
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
    visit({ node, path: graphNode.id, multiplier, resident });
    const childHasExplicitRepeat = children.some((child) => Number.isFinite(child.repeat));
    const childMultiplier = childRepeatMultiplier(node, multiplier, { repeatHandled: childHasExplicitRepeat });
    const childResident = childResidentRepeat(node, resident, { repeatHandled: childHasExplicitRepeat });
    for (const child of children) walk(child.id, childMultiplier, childResident);
  }
  walk(graph.root_id || graph.nodes.find((node) => node.parent_id == null)?.id || "root");
}
