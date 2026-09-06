const PATH_WRAPPERS = new Set(["model", "language_model"]);
import { buildSkeleton } from "./skeleton.js";

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
      graph_truth_used_ids: [...used],
    },
  };
}

function gapNodeDefinition(node, used) {
  const children = (node.children || []).map((child) => gapNodeDefinition(child, used)).filter(Boolean);
  const ownGap = Number(node.params) > 0 && !used.has(node.id);
  if (!ownGap && children.length === 0) return null;
  return {
    node,
    ownGap,
    children,
  };
}

function appendGapDefinition(definition, parentId, path, order, nodes, edges) {
  const node = definition.node;
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
    params: definition.ownGap ? node.params : 0,
    weight_shapes: definition.ownGap ? node.weight_shapes : null,
    dtype: definition.ownGap ? node.dtype : null,
    input_shape: null,
    output_shape: null,
    value_source: definition.ownGap ? "checkpoint" : null,
    tensor_names: definition.ownGap ? node.tensor_names : null,
  });
  const childPaths = definition.children.map((child, index) => appendGapDefinition(child, path, `${path}.${index}`, index, nodes, edges));
  for (let index = 0; index + 1 < childPaths.length; index += 1) {
    const source = nodes.find((candidate) => candidate.id === childPaths[index]);
    const target = nodes.find((candidate) => candidate.id === childPaths[index + 1]);
    edges.push({
      id: `${source.id}~${target.id}`,
      source: source.id,
      target: target.id,
      source_canonical_id: source.canonical_id,
      target_canonical_id: target.canonical_id,
      kind: "dataflow",
      evidence: "module-order",
    });
  }
  return path;
}

export function appendGraphGaps(graph, skeleton, usedTruthIds) {
  const definition = gapNodeDefinition(skeleton, new Set(usedTruthIds));
  if (!definition) return graph;
  const rootNodes = graph.nodes.filter((node) => node.parent_id === graph.root_id);
  const gapPath = `${graph.root_id}.${rootNodes.length}`;
  const gapContainer = {
    id: gapPath,
    canonical_id: "checkpoint_gaps",
    module_id: "checkpoint_gaps",
    parent_id: graph.root_id,
    order: rootNodes.length,
    name: "Checkpoint extra modules",
    type: "checkpoint-gaps",
    repeat: null,
    attributes: { class: "CheckpointExtraModules" },
    source_fields: [],
    confidence: "high",
    params: null,
    weight_shapes: null,
    dtype: null,
    input_shape: null,
    output_shape: null,
    value_source: null,
    tensor_names: null,
  };
  const nodes = [...graph.nodes, gapContainer];
  const edges = [...graph.edges];
  const gapChildren = skeleton.params === 0 && skeleton.children?.length ? skeleton.children : [skeleton];
  const definitions = gapChildren.map((child) => gapNodeDefinition(child, new Set(usedTruthIds))).filter(Boolean);
  const childPaths = definitions.map((child, index) => appendGapDefinition(child, gapPath, `${gapPath}.${index}`, index, nodes, edges));
  for (let index = 0; index + 1 < childPaths.length; index += 1) {
    const source = nodes.find((candidate) => candidate.id === childPaths[index]);
    const target = nodes.find((candidate) => candidate.id === childPaths[index + 1]);
    edges.push({ id: `${source.id}~${target.id}`, source: source.id, target: target.id, source_canonical_id: source.canonical_id, target_canonical_id: target.canonical_id, kind: "dataflow", evidence: "module-order" });
  }
  if (rootNodes.length) {
    const source = rootNodes[rootNodes.length - 1];
    edges.push({ id: `${source.id}~${gapPath}`, source: source.id, target: gapPath, source_canonical_id: source.canonical_id, target_canonical_id: gapContainer.canonical_id, kind: "dataflow", evidence: "module-order" });
  }
  return { ...graph, nodes, edges };
}

export function enrichGraphWithTruth(graph, truth, { hasTemplate, modelName, canonicalArchitecture }) {
  if (!truth || !Array.isArray(truth.tensors) || truth.tensors.length === 0) {
    return { graph, diagnostics: { strategy: "no-truth" } };
  }
  const skeleton = buildSkeleton(truth.tensors);
  const truthGraph = skeletonTruthGraph(skeleton);
  if (!hasTemplate) {
    const root = truthGraph.nodes.find((node) => node.id === truthGraph.root_id);
    if (root) {
      root.canonical_id = "skeleton";
      root.module_id = "skeleton";
      root.name = modelName || canonicalArchitecture || "Model";
      root.type = "model";
    }
    return {
      graph: truthGraph,
      diagnostics: { strategy: "skeleton-truth", total_tensors: truth.tensors.length, parameter_total: truth.parameterTotal ?? null },
    };
  }
  const bound = bindTruthToGraph(graph, truthGraph);
  const enrichedGraph = appendGraphGaps(bound.graph, skeleton, bound.diagnostics.graph_truth_used_ids);
  return {
    graph: enrichedGraph,
    diagnostics: {
      strategy: "template+truth",
      bound_tensors: bound.diagnostics.graph_bound_tensors,
      total_tensors: truth.tensors.length,
      template_gaps: bound.diagnostics.graph_truth_gaps,
      ambiguous_truth_matches: bound.diagnostics.graph_ambiguous_truth_matches,
    },
  };
}
