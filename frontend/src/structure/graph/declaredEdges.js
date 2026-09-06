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
    const suffix = id.split(".").at(-1);
    if (suffix && !byReference.has(suffix)) byReference.set(suffix, child);
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
  return edges.length > 0 ? edges : null;
}
