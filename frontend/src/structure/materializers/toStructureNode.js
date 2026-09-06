import { enrichNetworkWithTruth, TEMPLATE_FAMILIES } from "../truth/mergeSemantics.js";
import { materializeStructureGraph } from "../graph/materializeStructureGraph.js";

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
      // IR v2 可选字段（仅当 spec 携带时透传）
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
    // IR v2 可选字段（仅当 spec 携带时透传）
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

  // 真值合并：有模板 → 绑定真值；无模板 → trie 树兜底；无真值 → 原样
  const hasTemplate = TEMPLATE_FAMILIES.has(resolved?.canonicalArchitecture);
  const { network, diagnostics: truthDiagnostics } = enrichNetworkWithTruth(
    templateNetwork,
    truth,
    { hasTemplate, modelName: templateNetwork?.name, canonicalArchitecture: resolved?.canonicalArchitecture },
  );
  const mergedDiagnostics = { ...diagnostics, ...truthDiagnostics };
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

  return {
    summary: {
      strategy: effectiveStrategy,
      model_family: network.name,
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
      strategy: effectiveStrategy,
      checkpoint_truth: options.checkpointTruthStatus || (truth ? "available" : "not-requested"),
      checkpoint_truth_method: options.checkpointTruthMethod || truth?.method || null,
      checkpoint_truth_error: options.checkpointTruthError || null,
      config_endpoint: options.configEndpoint || null,
      checkpoint_truth_endpoint: options.checkpointTruthEndpoint || null,
      diagnostics: mergedDiagnostics,
    },
    root,
    graph: materializeStructureGraph(root),
    extra_config: normalized.raw,
  };
}
