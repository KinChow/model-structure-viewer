/**
 * Resolve builder-declared child references into stable graph edges.
 * References use a child id suffix (for example `gate_proj`) so builders do
 * not need to know the materialized tree path used by the renderer.
 */
export function materializeDeclaredEdges(item, canonicalItems = new Map(), modelId = "unknown") {
  const declarations = item?.node?.attributes?.dataflow_edges;
  if (declarations === undefined) return null;
  const fail = message => { throw new Error(`Invalid dataflow declaration for ${modelId} in ${item.node?.id || item.path}: ${message}`); };
  if (!Array.isArray(declarations)) fail("expected an array");
  const byReference = new Map();
  const add = (key, child) => {
    const hits = byReference.get(key) || new Set();
    hits.add(child); byReference.set(key, hits);
  };
  for (const child of item.childItems || []) {
    const id = String(child.node?.id || "");
    if (!id) continue;
    const parts = id.split(".");
    add(id, child);
    add(parts.at(-1), child);
    if (parts.length > 1) add(parts.slice(-2).join("."), child);
  }
  const resolve = ref => {
    if (typeof ref !== "string") fail("endpoint must be a string");
    // 局部完整路径优先（避免 DSpark 的 norm 误绑根 norm），再接受全图 canonical ID。
    // 后缀仅在直接子节点中查找；多义引用必须补齐路径。
    const scoped = canonicalItems.get(`${item.node.id}.${ref}`);
    if (scoped?.length > 1) fail(`ambiguous scoped endpoint ${ref}`);
    if (scoped?.length === 1) return scoped[0];
    const exact = canonicalItems.get(ref);
    if (exact?.length > 1) fail(`ambiguous canonical endpoint ${ref}`);
    if (exact?.length === 1) return exact[0];
    const hits = byReference.get(ref);
    if (hits?.size > 1) fail(`ambiguous endpoint ${ref}`);
    if (hits?.size === 1) return [...hits][0];
    fail(`unresolved endpoint ${ref}`);
  };
  const edges = [];
  for (const declaration of declarations) {
    if (!Array.isArray(declaration) || declaration.length !== 2) fail("expected [from, to]");
    const [sourceRef, targetRef] = declaration;
    let source, target;
    try {
      source = resolve(sourceRef);
      target = resolve(targetRef);
    } catch (error) {
      throw new Error(`${error.message}; endpoints ${JSON.stringify(declaration)}`);
    }
    if (source.path === target.path) fail(`self edge ${sourceRef}`);
    edges.push({
      id: `${source.path}=>${target.path}`,
      source: source.path,
      target: target.path,
      kind: "dataflow",
      evidence: "declared",
    });
  }
  // 关系标注（如 CED 全局 KV 投影）走单独的 dataflow_edge_relations 属性，避免污染
  // dataflow_edges 的显示（3 元组会在属性面板渲染成 [object Object]）。按 ref 后缀解析后
  // 打到对应边上：kind 仍为 dataflow（进入渲染过滤），relation 决定虚线/提示样式。
  const relations = item?.node?.attributes?.dataflow_edge_relations;
  if (relations !== undefined && !Array.isArray(relations)) fail("expected relation array");
  if (Array.isArray(relations)) {
    const edgeById = new Map(edges.map((edge) => [edge.id, edge]));
    for (const rel of relations) {
      const source = resolve(rel?.from);
      const target = resolve(rel?.to);
      const edge = edgeById.get(`${source.path}=>${target.path}`);
      if (!edge) fail(`relation without declared edge ${rel.from} => ${rel.to}`);
      if (rel.relation) edge.relation = rel.relation;
      if (rel.label) edge.label = rel.label;
      if (rel.evidence) edge.evidence = rel.evidence;
    }
  }
  return edges;
}
