/**
 * Resolve builder-declared child references into stable graph edges.
 * References use a child id suffix (for example `gate_proj`) so builders do
 * not need to know the materialized tree path used by the renderer.
 */
export function materializeDeclaredEdges(item) {
  const declarations = item?.node?.attributes?.dataflow_edges;
  if (!Array.isArray(declarations) || declarations.length === 0) return null;
  const byReference = new Map();
  for (const child of item.childItems || []) {
    const id = String(child.node?.id || "");
    if (!id) continue;
    byReference.set(id, child);
    const parts = id.split(".");
    const suffix = parts.at(-1);
    if (suffix && !byReference.has(suffix)) byReference.set(suffix, child);
    // 末两段引用（如 `indexer.q_proj`）：消除与同名末段（`q_proj`）的歧义，
    // 供嵌套子算子（indexer.*）精确连线。首个占位者优先，避免覆盖。
    if (parts.length >= 2) {
      const twoSeg = parts.slice(-2).join(".");
      if (!byReference.has(twoSeg)) byReference.set(twoSeg, child);
    }
  }
  const edges = [];
  for (const declaration of declarations) {
    if (!Array.isArray(declaration) || declaration.length !== 2) return null;
    const [sourceRef, targetRef] = declaration;
    const source = byReference.get(sourceRef);
    const target = byReference.get(targetRef);
    if (!source || !target || source.path === target.path) return null;
    edges.push({
      id: `${source.path}=>${target.path}`,
      source: source.path,
      target: target.path,
      kind: "dataflow",
      evidence: "declared",
    });
  }
  if (edges.length === 0) return null;
  // 关系标注（如 CED 全局 KV 投影）走单独的 dataflow_edge_relations 属性，避免污染
  // dataflow_edges 的显示（3 元组会在属性面板渲染成 [object Object]）。按 ref 后缀解析后
  // 打到对应边上：kind 仍为 dataflow（进入渲染过滤），relation 决定虚线/提示样式。
  const relations = item?.node?.attributes?.dataflow_edge_relations;
  if (Array.isArray(relations)) {
    const edgeById = new Map(edges.map((edge) => [edge.id, edge]));
    for (const rel of relations) {
      const source = byReference.get(rel?.from);
      const target = byReference.get(rel?.to);
      if (!source || !target) continue;
      const edge = edgeById.get(`${source.path}=>${target.path}`);
      if (!edge) continue;
      if (rel.relation) edge.relation = rel.relation;
      if (rel.label) edge.label = rel.label;
      if (rel.evidence) edge.evidence = rel.evidence;
    }
  }
  return edges;
}
