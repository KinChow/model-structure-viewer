const PATH_WRAPPERS = new Set(["model", "language_model"]);

export function canonicalModulePath(value) {
  const parts = String(value || "").split(".").filter(Boolean);
  while (parts.length > 0 && PATH_WRAPPERS.has(parts[0])) parts.shift();
  if (parts[0] === "layers" || parts[0] === "text_decoder") parts[0] = "decoder";
  if (parts[0] === "visual" || parts[0] === "vision") parts[0] = "vision_tower";
  return parts.join(".");
}

/** Build a minimal Graph IR index from checkpoint skeleton facts. */
export function skeletonTruthGraph(skeleton) {
  const nodes = [];
  const edges = [];
  function visit(node, path, parentId = null, order = 0) {
    nodes.push({
      id: path,
      canonical_id: node.id,
      module_id: node.id,
      parent_id: parentId,
      order,
      name: node.name,
      type: node.type || "module",
      repeat: node.repeat ?? null,
      attributes: node.weight_dtypes && Object.keys(node.weight_dtypes).length ? { weight_dtypes: node.weight_dtypes } : {},
      source_fields: [],
      confidence: "high",
      params: node.params ?? null,
      weight_shapes: node.weight_shapes || null,
      dtype: node.dtype || null,
      input_shape: null,
      output_shape: null,
      value_source: "checkpoint",
      tensor_names: node.tensor_names || null,
    });
    const children = node.children || [];
    children.forEach((child, index) => visit(child, `${path}.${index}`, path, index));
    children.slice(0, -1).forEach((child, index) => edges.push({
      id: `${`${path}.${index}`}~${`${path}.${index + 1}`}`,
      source: `${path}.${index}`,
      target: `${path}.${index + 1}`,
      kind: "dataflow",
      evidence: "module-order",
      source_canonical_id: children[index].id,
      target_canonical_id: children[index + 1].id,
    }));
  }
  visit(skeleton, "root");
  return { version: 2, schema_version: 2, root_id: "root", nodes, edges };
}

/** Bind checkpoint node facts to the canonical Graph IR node index. */
export function bindTruthToGraph(graph, truthGraph) {
  const truthNodes = (truthGraph?.nodes || []).filter((node) => Number(node.params) > 0);
  const truthByPath = new Map();
  for (const node of truthNodes) {
    const key = canonicalModulePath(node.canonical_id || node.module_id || node.id);
    const entries = truthByPath.get(key) || [];
    entries.push(node);
    truthByPath.set(key, entries);
  }
  const used = new Set();
  const boundIds = [];
  const ambiguous = [];
  const nodes = (graph?.nodes || []).map((node) => {
    const key = canonicalModulePath(node.canonical_id || node.module_id || node.id);
    const candidates = (truthByPath.get(key) || []).filter((candidate) => !used.has(candidate.id));
    if (candidates.length > 1) {
      ambiguous.push({ template: node.canonical_id || node.module_id || node.id, candidates: candidates.map((candidate) => candidate.canonical_id || candidate.id) });
      return node;
    }
    const [truthNode] = candidates;
    if (!truthNode) return node;
    used.add(truthNode.id);
    boundIds.push(truthNode.canonical_id || truthNode.id);
    return {
      ...node,
      params: truthNode.params,
      weight_shapes: truthNode.weight_shapes,
      dtype: truthNode.dtype,
      tensor_names: truthNode.tensor_names,
      value_source: "checkpoint",
      attributes: {
        ...(node.attributes || {}),
        ...(truthNode.attributes?.weight_dtypes ? { weight_dtypes: truthNode.attributes.weight_dtypes } : {}),
      },
    };
  });
  return {
    graph: { ...graph, nodes },
    diagnostics: {
      graph_bound_tensors: boundIds.length,
      graph_truth_gaps: truthNodes.filter((node) => !used.has(node.id)).map((node) => node.canonical_id || node.id),
      graph_ambiguous_truth_matches: ambiguous,
    },
  };
}
