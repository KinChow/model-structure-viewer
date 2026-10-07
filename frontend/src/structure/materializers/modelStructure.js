import { hasModelArchitecture } from "../registry/resolveArchitecture.js";
import { materializeStructureGraph } from "../graph/materializeStructureGraph.js";
import { enrichGraphWithTruth } from "../truth/graphTruth.js";
import { bindSourceRefToGraph } from "../source_ref/bindSourceRef.js";
import { checkpointPathAliases } from "../archs/index.js";
import { galleryAlignmentForGraph } from "../galleryAlignment.js";
import {
  logicalParameterSummaryFromTensors,
  logicalParameterTotalFromHeader,
  quantizationConfigOf,
} from "../../cost/quantBytes.js";

function structureNodeFromSpec(spec) {
  if (spec.kind === "operator") {
    return {
      id: spec.id,
      name: spec.name,
      type: "operator",
      attributes: {
        class: spec.name,
        operator_id: spec.operatorId,
        ...spec.attributes,
      },
      source_fields: Object.keys(spec.attributes || {}),
      confidence: "high",
      children: (spec.children || []).map(structureNodeFromSpec),
      // 节点扩展字段仅在 spec 携带时透传；图边由 Structure IR 单独承载。
      params: spec.params,
      weight_shapes: spec.weight_shapes,
      dtype: spec.dtype,
      input_shape: spec.input_shape,
      output_shape: spec.output_shape,
      value_source: spec.value_source,
      tensor_names: spec.tensor_names,
    };
  }
  return {
    id: spec.id,
    name: spec.name,
    type: spec.type,
    repeat: spec.repeat,
    attributes: spec.attributes || {},
    source_fields: Object.keys(spec.attributes || {}),
    confidence: "high",
    children: (spec.children || []).map(structureNodeFromSpec),
    // 节点扩展字段仅在 spec 携带时透传；图边由 Structure IR 单独承载。
    params: spec.params,
    weight_shapes: spec.weight_shapes,
    dtype: spec.dtype,
    input_shape: spec.input_shape,
    output_shape: spec.output_shape,
    value_source: spec.value_source,
    tensor_names: spec.tensor_names,
  };
}

function normalizeGraphEdges(graph) {
  if (!graph?.edges) return graph;
  return {
    ...graph,
    edges: graph.edges.map((edge) => ({
      ...edge,
      relation: edge.relation ?? null,
      label: edge.label ?? null,
    })),
  };
}

export function materializeModelStructure(ir) {
  const { network, normalized, resolved, options = {}, diagnostics = {} } = ir;
  const rawTruth = options.truth;
  const quant = quantizationConfigOf(normalized);
  const logicalRoleSummary = rawTruth
    ? logicalParameterSummaryFromTensors(rawTruth.tensors, quant)
    : null;
  const logicalParameterTotal = rawTruth
    ? logicalParameterTotalFromHeader(rawTruth, quant)
    : null;
  const truth = rawTruth && (logicalParameterTotal != null || logicalRoleSummary)
    ? {
        ...rawTruth,
        ...(logicalParameterTotal != null ? { logicalParameterTotal } : {}),
        ...(logicalRoleSummary && !rawTruth.logicalParameterCount
          ? { logicalParameterCount: logicalRoleSummary.logicalParameterCount }
          : {}),
      }
    : rawTruth;

  const hasBuilder = hasModelArchitecture(resolved?.architecture);
  const truthDiagnostics = truth ? { strategy: "graph-truth" } : { strategy: "no-truth" };
  let mergedDiagnostics = { ...diagnostics, ...truthDiagnostics };
  const effectiveStrategy = truthDiagnostics.strategy === "no-truth" || !truth
    ? ir.strategy
    : truthDiagnostics.strategy;
  // 根模块即 model（transformers/vLLM 惯例）：瞬态 StructureNode 树，仅用于
  // 物化——Graph IR 是唯一载荷（P7 步骤 7）。
  const model = {
    id: network.id,
    name: network.name,
    type: "model",
    attributes: {
      class: network.name,
      model_type: normalized.modelType,
      architecture: resolved.architecture || normalized.architecture,
      ...(network.attributes || {}),
    },
    source_fields: ["model_type", "architecture"],
    confidence: "high",
    children: network.children.map(structureNodeFromSpec),
  };
  let graph = materializeStructureGraph(model, { modelId: options.modelId || network.name });
  const graphTruth = enrichGraphWithTruth(graph, truth, {
    hasBuilder,
    modelName: network?.name,
    architecture: resolved?.architecture,
    truthPathAliases: checkpointPathAliases(normalized),
  });
  graph = normalizeGraphEdges(graphTruth.graph);
  const sourceRefBound = bindSourceRefToGraph(graph, options.sourceRef);
  graph = sourceRefBound.graph;
  mergedDiagnostics = {
    ...diagnostics,
    ...graphTruth.diagnostics,
    source_ref: sourceRefBound.diagnostics,
  };
  const graphRoot = graph.nodes.find((node) => node.id === graph.root_id);

  return {
    summary: {
      strategy: graphTruth.diagnostics.strategy === "no-truth" ? ir.strategy : graphTruth.diagnostics.strategy,
      model_family: graphRoot?.name || network.name,
      model_type: normalized.modelType,
      architecture: resolved.architecture || normalized.architecture || normalized.modelType,
      text_layers: normalized.layers,
      vision_layers: normalized.visionLayers,
      vision_hidden_size: normalized.visionHiddenSize,
      vision_output_size: normalized.visionOutputSize,
      hidden_size: normalized.hiddenSize,
      num_attention_heads: normalized.attentionHeads,
      num_key_value_heads: normalized.kvHeads,
      num_local_experts: normalized.experts,
      n_routed_experts: normalized.experts,
      num_experts_per_tok: normalized.expertsPerToken,
      max_position_embeddings: normalized.contextLength,
      // 真值：逻辑模型参数量；packed checkpoint 的 storage-elements
      // 单独保留，避免把 U8 weight_packed 误展示为参数量。
      parameters_total: truth?.logicalParameterTotal ?? truth?.parameterTotal ?? null,
      parameters_by_dtype: truth?.parameterCount ?? null,
      parameters_by_logical_dtype: truth?.logicalParameterCount ?? null,
      parameters_storage_total: truth?.parameterTotal ?? null,
      gallery_alignment: galleryAlignmentForGraph(graph, resolved.architecture || normalized.architecture),
    },
    source: {
      kind: options.source || "config",
      model_id: options.modelId,
      revision: options.revision,
      strategy: graphTruth.diagnostics.strategy === "no-truth" ? effectiveStrategy : graphTruth.diagnostics.strategy,
      checkpoint_truth: options.checkpointTruthStatus || (truth ? "available" : "not-requested"),
      checkpoint_truth_method: options.checkpointTruthMethod || truth?.method || null,
      checkpoint_truth_error: options.checkpointTruthError || null,
      config_endpoint: options.configEndpoint || null,
      checkpoint_truth_endpoint: options.checkpointTruthEndpoint || null,
      diagnostics: mergedDiagnostics,
    },
    graph,
    extra_config: normalized.raw,
  };
}
