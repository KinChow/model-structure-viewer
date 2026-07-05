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
  };
}

export function materializeModelStructure(ir) {
  const { network, normalized, resolved, options = {}, diagnostics = {} } = ir;
  return {
    summary: {
      strategy: ir.strategy,
      model_family: network.name,
      model_type: normalized.modelType,
      architecture: resolved.architecture || normalized.architecture || normalized.modelType,
      canonical_architecture: resolved.canonicalArchitecture,
      text_layers: normalized.layers,
      vision_layers: normalized.visionLayers,
      hidden_size: normalized.hiddenSize,
      num_attention_heads: normalized.attentionHeads,
      num_key_value_heads: normalized.kvHeads,
      num_local_experts: normalized.experts,
      n_routed_experts: normalized.experts,
      num_experts_per_tok: normalized.expertsPerToken,
      max_position_embeddings: normalized.contextLength,
    },
    source: {
      kind: options.source || "config",
      model_id: options.modelId,
      revision: options.revision,
      strategy: ir.strategy,
      diagnostics,
    },
    root: {
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
    },
    extra_config: normalized.raw,
  };
}
