import { buildSkeleton } from "./skeleton.js";

// 只剥 HF named_modules 根包装（model. / language_model.）。
// 对标 transformers PreTrainedModel 把主干挂在 self.model。
const PATH_WRAPPERS = new Set(["model", "language_model"]);

export function canonicalModulePath(value) {
  const parts = String(value || "").split(".").filter(Boolean);
  while (parts.length > 0 && PATH_WRAPPERS.has(parts[0])) parts.shift();
  return parts.join(".");
}

function pathBindKeys(modulePath) {
  const path = canonicalModulePath(modulePath);
  return path ? [path] : [];
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

/**
 * checkpoint 按模块路径绑到图节点。两端都剥 HF 根包装后做相等匹配。
 * 图 id 已是 HF `_modules` 名（layers / visual / vision_tower / language_model）。
 * 多候选记 ambiguous，不静默丢弃。
 */
export function bindTruthToGraph(graph, truthGraph) {
  const truthNodes = (truthGraph?.nodes || []).filter((node) => Number(node.params) > 0);
  const truthByPath = new Map();
  for (const node of truthNodes) {
    const truthId = node.canonical_id || node.module_id || node.id;
    for (const pathKey of pathBindKeys(truthId)) {
      const pathEntries = truthByPath.get(pathKey) || [];
      pathEntries.push(node);
      truthByPath.set(pathKey, pathEntries);
    }
  }
  const used = new Set();
  const boundIds = [];
  const ambiguous = [];
  const nodes = (graph?.nodes || []).map((node) => {
    const templateId = node.canonical_id || node.module_id || node.id;
    const seen = new Set();
    const candidates = [];
    for (const key of pathBindKeys(templateId)) {
      for (const candidate of truthByPath.get(key) || []) {
        if (used.has(candidate.id) || seen.has(candidate.id)) continue;
        seen.add(candidate.id);
        candidates.push(candidate);
      }
    }
    if (candidates.length > 1) {
      ambiguous.push({ template: templateId, candidates: candidates.map((candidate) => candidate.canonical_id || candidate.id) });
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

export function enrichGraphWithTruth(graph, truth, { hasBuilder, modelName, architecture }) {
  // 离线证据文件形态：truth.skeleton 是**已折叠**的 SkeletonNode（由
  // fetch-evidence --headers 从 safetensors 头部构建后入库，K3 原始张量
  // 表 59.7MB 折叠后小几个数量级，符合「仅轻量元数据入库」纪律）。
  // 在场时与 tensors 形态等价，压过 paramDtypes 推断层（聚合层已按
  // hasParameterCount 走 checkpoint 路径）。
  if (truth?.skeleton) {
    const truthGraph = skeletonTruthGraph(truth.skeleton);
    if (!hasBuilder) {
      const root = truthGraph.nodes.find((node) => node.id === truthGraph.root_id);
      if (root) {
        root.canonical_id = "skeleton";
        root.module_id = "skeleton";
        root.name = modelName || architecture || "Model";
        root.type = "model";
      }
      return {
        graph: truthGraph,
        diagnostics: { strategy: "skeleton-truth-file", total_tensors: truth.tensor_count ?? null, parameter_total: truth.parameterTotal ?? null },
      };
    }
    const bound = bindTruthToGraph(graph, truthGraph);
    const enrichedGraph = appendGraphGaps(bound.graph, truth.skeleton, bound.diagnostics.graph_truth_used_ids);
    return {
      graph: enrichedGraph,
      diagnostics: {
        strategy: "template+truth-file",
        bound_tensors: bound.diagnostics.graph_bound_tensors,
        total_tensors: truth.tensor_count ?? null,
        template_gaps: bound.diagnostics.graph_truth_gaps,
        ambiguous_truth_matches: bound.diagnostics.graph_ambiguous_truth_matches,
      },
    };
  }
  if (!truth || !Array.isArray(truth.tensors) || truth.tensors.length === 0) {
    if (Number.isFinite(truth?.parameterTotal) && truth.parameterTotal > 0) {
      // S3：只有总量、没有逐张量。图仍是模板（或空网络），不改骨架。
      return {
        graph,
        diagnostics: {
          strategy: hasBuilder ? "template+header-truth" : "header-truth",
          total_tensors: truth.tensor_count ?? null,
          parameter_total: truth.parameterTotal,
        },
      };
    }
    return { graph, diagnostics: { strategy: "no-truth" } };
  }
  const skeleton = buildSkeleton(truth.tensors);
  const truthGraph = skeletonTruthGraph(skeleton);
  if (!hasBuilder) {
    const root = truthGraph.nodes.find((node) => node.id === truthGraph.root_id);
    if (root) {
      root.canonical_id = "skeleton";
      root.module_id = "skeleton";
      root.name = modelName || architecture || "Model";
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
