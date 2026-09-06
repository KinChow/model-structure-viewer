import { TEMPLATE_FAMILIES } from "../truth/mergeSemantics.js";
import { materializeStructureGraph } from "../graph/materializeStructureGraph.js";
import { projectGraphToTree } from "../graph/projectGraphToTree.js";
import { enrichGraphWithTruth } from "../truth/graphTruth.js";

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
      children: [],
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

export function materializeModelStructure(ir) {
  const { network: templateNetwork, normalized, resolved, options = {}, diagnostics = {} } = ir;
  const truth = options.truth;

  const hasTemplate = TEMPLATE_FAMILIES.has(resolved?.canonicalArchitecture);
  const network = templateNetwork;
  const truthDiagnostics = truth ? { strategy: "graph-truth" } : { strategy: "no-truth" };
  let mergedDiagnostics = { ...diagnostics, ...truthDiagnostics };
  const effectiveStrategy = truthDiagnostics.strategy === "no-truth" || !truth
    ? ir.strategy
    : truthDiagnostics.strategy;
  const root = {
    id: network.id,
    name: network.name,
    type: "model",
    attributes: {
      class: network.name,
      model_type: normalized.modelType,
      canonical_architecture: resolved.canonicalArchitecture,
    },
    source_fields: ["model_type", "canonical_architecture"],
    confidence: "high",
    children: network.children.map(structureNodeFromSpec),
  };
  let graph = materializeStructureGraph(root);
  const graphTruth = enrichGraphWithTruth(graph, truth, {
    hasTemplate,
    modelName: templateNetwork?.name,
    canonicalArchitecture: resolved?.canonicalArchitecture,
  });
  graph = graphTruth.graph;
  mergedDiagnostics = { ...diagnostics, ...graphTruth.diagnostics };

  return {
    summary: {
      strategy: graphTruth.diagnostics.strategy === "no-truth" ? ir.strategy : graphTruth.diagnostics.strategy,
      model_family: projectGraphToTree(graph)?.name || templateNetwork.name,
      model_type: normalized.modelType,
      architecture: resolved.architecture || normalized.architecture || normalized.modelType,
      canonical_architecture: resolved.canonicalArchitecture,
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
      // 真值：模型级精确参数量（@huggingface/hub 计算）
      parameters_total: truth?.parameterTotal ?? null,
      parameters_by_dtype: truth?.parameterCount ?? null,
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
    root: projectGraphToTree(graph),
    graph,
    extra_config: normalized.raw,
  };
}
