import { formulaForOperator } from "../../formulas/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

function cleanAttributes(attributes) {
  return Object.fromEntries(
    Object.entries(attributes).filter(([, value]) => value !== undefined && value !== null),
  );
}

export function operatorSpec(id, name, operatorId, attributes = {}, numericShapes = {}) {
  const formula = formulaForOperator(operatorId);
  return {
    kind: "operator",
    id,
    name,
    operatorId,
    input_shape: numericShapes.input,
    output_shape: numericShapes.output,
    attributes: cleanAttributes({
      formula_id: operatorId,
      formula: formula?.formula,
      explanation: formula?.explanation,
      inputs: formula?.inputs,
      outputs: formula?.outputs,
      ...attributes,
    }),
  };
}

export function attentionOperatorSpecs(prefix, attentionKind, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return [
    operatorSpec(`${prefix}.q_proj`, "q projection", "linear", shapeFlow(shapes.hidden, shapes.attentionQuery), { input: dims.hidden, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.k_proj`, "k projection", "linear", shapeFlow(shapes.hidden, shapes.attentionKey), { input: dims.hidden, output: dims.attentionKey }),
    operatorSpec(`${prefix}.v_proj`, "v projection", "linear", shapeFlow(shapes.hidden, shapes.attentionValue), { input: dims.hidden, output: dims.attentionValue }),
    operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, `${shapes.attentionQuery}, ${shapes.attentionKey}`),
      query_shape: shapes.attentionQuery,
      key_shape: shapes.attentionKey,
      position_shape: "[batch, sequence]",
    }, { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.scores`, "attention scores", "matmul", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, shapes.attentionScores),
      formula: "S = Q K^T / sqrt(d)",
      explanation: "用旋转后的 Q 与 K^T 计算注意力分数。",
      inputs: ["Q", "K"],
      outputs: ["S"],
      attention_kind: attentionKind,
      query_shape: shapes.attentionQuery,
      key_shape: shapes.attentionKey,
    }, { input: dims.attentionQuery, output: dims.attentionScores }),
    operatorSpec(
      `${prefix}.softmax`,
      "attention probabilities",
      "softmax",
      shapeFlow(shapes.attentionScores, shapes.attentionProbabilities), { input: dims.attentionScores, output: dims.attentionProbabilities },
    ),
    operatorSpec(`${prefix}.context`, "weighted value", "matmul", {
      ...shapeFlow(`${shapes.attentionProbabilities}, ${shapes.attentionValue}`, shapes.attentionContext),
      formula: "O = P V",
      explanation: "用注意力概率 P 对 V 做加权聚合。",
      inputs: ["probabilities", "V"],
      outputs: ["O"],
      probabilities_shape: shapes.attentionProbabilities,
      value_shape: shapes.attentionValue,
    }, { input: dims.attentionProbabilities, output: dims.attentionContext }),
    operatorSpec(`${prefix}.o_proj`, "output projection", "linear", shapeFlow(shapes.attentionContext, shapes.hidden), { input: dims.attentionContext, output: dims.hidden }),
  ];
}

export function linearAttentionOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  if (normalized.linearAttentionMode === "kimi_k3") {
    return canonicalKdaOperatorSpecs(prefix, normalized, "kimi_k3");
  }
  if (normalized.linearAttentionMode === "kimi") {
    return canonicalKdaOperatorSpecs(prefix, normalized, "kimi");
  }
  if (normalized.linearAttentionMode === "glm5_next") {
    return canonicalKdaOperatorSpecs(prefix, normalized, "glm5_next");
  }
  if (normalized.linearAttentionMode === "qwen4_exp") {
    return canonicalKdaOperatorSpecs(prefix, normalized, "qwen4_exp");
  }
  if (normalized.linearAttentionMode === "qwen4_exp") {
    return [
      operatorSpec(`${prefix}.in_proj_qkv`, "linear attention qkv projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
      operatorSpec(`${prefix}.in_proj_z`, "linear attention gate projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
      operatorSpec(`${prefix}.in_proj_b`, "linear attention decay projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
      operatorSpec(`${prefix}.in_proj_a`, "linear attention gate feature projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
      operatorSpec(`${prefix}.conv1d`, "short convolution", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
      operatorSpec(`${prefix}.norm`, "gated RMSNorm", "rmsnorm", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
      operatorSpec(`${prefix}.out_proj`, "output projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    ];
  }
  return [
    operatorSpec(`${prefix}.in_proj_qkv`, "linear attention qkv projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.in_proj_z`, "linear attention gate projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.in_proj_b`, "linear attention decay projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.short_conv`, "short convolution", "linear_attention", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.state_update`, "linear attention state update", "linear_attention", {
      ...shapeFlow(`${shapes.hidden}, state`, shapes.hidden), attention_kind: "linear",
    }, { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.output_gate`, "linear attention output gate", "linear_attention_gate", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
  ];
}

// KDA is one semantic structure. Framework-specific fused projections remain
// in attributes so vLLM/SGLang implementation details do not duplicate nodes.
function canonicalKdaOperatorSpecs(prefix, normalized, modelKind) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const keyHeads = normalized.linearKeyHeads || normalized.attentionHeads || 0;
  const valueHeads = normalized.linearValueHeads || normalized.attentionHeads || keyHeads;
  const keyDim = normalized.linearKeyDim || normalized.headDim || 0;
  const valueDim = normalized.linearValueDim || normalized.valueHeadDim || keyDim;
  const keyProjection = keyHeads * keyDim;
  const valueProjection = valueHeads * valueDim;
  const qkvFlat = 2 * keyProjection + valueProjection;
  const qkvShape = modelKind === "qwen4_exp"
    ? `[batch, sequence, Q/K=${keyHeads}x${keyDim}, V=${valueHeads}x${valueDim}]`
    : `[batch, sequence, linear heads=${keyHeads}, head dimension=${keyDim}]`;
  const betaShape = `[batch, sequence, value heads=${valueHeads}]`;
  const gateShape = modelKind === "qwen4_exp"
    ? `[batch, sequence, value heads=${valueHeads}, value dimension=${valueDim}]`
    : qkvShape;
  const stateShape = `[batch, value heads=${valueHeads}, state value dimension=${valueDim}, state key dimension=${keyDim}]`;
  const qkvDims = modelKind === "qwen4_exp" ? [-1, -1, qkvFlat] : [-1, -1, keyHeads, keyDim];
  const outputDims = modelKind === "qwen4_exp" ? [-1, -1, valueHeads, valueDim] : qkvDims;
  const betaDims = [-1, -1, valueHeads];
  const fullRank = modelKind === "kimi_k3";
  const qwen = modelKind === "qwen4_exp";
  const implementation = fullRank
    ? {
      input_projection: "fused_qkvg_proj",
      beta_projection: "b_proj",
      decay_projection: ["f_a_proj", "f_b_proj"],
      short_convolution: "qkv_conv1d",
      output_gate: "fused_qkvg_proj.g",
    }
    : qwen
      ? {
        input_projection: "in_proj_qkvz",
        beta_projection: "in_proj_ba.b",
        decay_projection: "in_proj_ba.a",
        short_convolution: "conv1d",
        output_gate: "in_proj_qkvz.z",
      }
      : {
      input_projection: "in_proj_qkvbfg_a",
      beta_projection: "in_proj_qkvbfg_a.beta",
      decay_projection: ["in_proj_qkvbfg_a.f_a", "f_b_proj"],
      short_convolution: ["q_conv1d", "k_conv1d", "v_conv1d"],
      output_gate: ["in_proj_qkvbfg_a.g_a", "g_b_proj"],
    };
  const projectionLayout = fullRank ? ["q", "k", "v", "g"] : qwen ? ["q", "k", "v", "z"] : ["q", "k", "v", "beta", "f_a", "g_a"];
  return [
    operatorSpec(`${prefix}.qkv_projection`, "QKV projection", "linear", {
      ...shapeFlow(shapes.hidden, qkvShape),
      semantic_role: "q_k_v_projection",
      implementation,
      projection_size: qwen ? { qk: keyProjection, v: valueProjection } : keyProjection,
      fused_projection_layout: projectionLayout,
    }, { input: dims.hidden, output: qkvDims }),
    operatorSpec(`${prefix}.beta_projection`, "beta projection", "linear", {
      ...shapeFlow(shapes.hidden, betaShape),
      semantic_role: "delta_beta",
      implementation: implementation.beta_projection,
      activation: "sigmoid_in_kda_kernel",
    }, { input: dims.hidden, output: betaDims }),
    operatorSpec(`${prefix}.decay_projection`, "forget/decay gate projection", "linear", {
      ...shapeFlow(shapes.hidden, gateShape),
      semantic_role: "forget_gate_logits",
      implementation: implementation.decay_projection,
      gate_lower_bound: normalized.linearLowerBound,
      projection_size: valueHeads,
    }, { input: dims.hidden, output: betaDims }),
    operatorSpec(`${prefix}.short_conv`, "qkv causal short convolution", "causal_conv1d", {
      ...shapeFlow(qkvShape, qkvShape),
      semantic_role: "q_k_v_short_convolution",
      implementation: implementation.short_convolution,
      branches: ["q", "k", "v"],
      kernel_size: normalized.linearConvKernelSize,
      activation: "silu",
      channel_layout: qwen ? { q: keyProjection, k: keyProjection, v: valueProjection } : undefined,
    }, { input: qkvDims, output: qkvDims }),
    operatorSpec(`${prefix}.state_update`, "KDA recurrent state", "gated_delta_attention", {
      ...shapeFlow(`${qkvShape}, ${betaShape}, ${stateShape}`, qwen ? gateShape : qkvShape),
      semantic_role: "gated_delta_recurrent_state",
      model_kind: modelKind,
      attention_kind: "linear",
      mode: "chunk_prefill_or_fused_recurrent",
      qk_l2norm: true,
      beta_activation: "sigmoid",
      safe_gate: !qwen,
      decay_activation: qwen ? "softplus" : "bounded_sigmoid",
      gate_lower_bound: normalized.linearLowerBound,
      decay_parameters: ["A_log", "dt_bias"],
      state_shape: stateShape,
    }, { input: qkvDims, output: outputDims }),
    operatorSpec(`${prefix}.output_gate_norm`, "gated RMSNorm", "gated_rmsnorm", {
      ...shapeFlow(qkvShape, gateShape),
      semantic_role: "gated_output_normalization",
      implementation: implementation.output_gate,
      gate_shape: gateShape,
      activation: "sigmoid",
    }, { input: outputDims, output: outputDims }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", {
      ...shapeFlow(gateShape, shapes.hidden),
      semantic_role: "attention_output_projection",
    }, { input: outputDims, output: dims.hidden }),
  ];
}

function kimiK3LinearAttentionOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const heads = normalized.linearKeyHeads || normalized.attentionHeads || 0;
  const headDim = normalized.linearKeyDim || normalized.headDim || 0;
  const projection = heads * headDim;
  const fusedWidth = 4 * projection;
  const fusedShape = `[batch, sequence, fused qkvg=${fusedWidth}]`;
  const qkvShape = `[batch, sequence, linear heads=${heads}, head dimension=${headDim}]`;
  const qkvFlatShape = `[batch, sequence, qkv=${3 * projection}]`;
  const betaShape = `[batch, sequence, linear heads=${heads}]`;
  const gateFeatureShape = `[batch, sequence, gate feature dimension=${headDim}]`;
  const stateShape = `[batch, linear heads=${heads}, state value dimension=${headDim}, state key dimension=${headDim}]`;
  const qkv = [-1, -1, heads, headDim];
  const flatProjection = [-1, -1, projection];
  const gateFeature = [-1, -1, headDim];
  return [
    operatorSpec(`${prefix}.fused_qkvg_proj`, "fused qkvg projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: ["q", "k", "v", "g"],
      projection_size: projection,
      gate_type: "full_rank",
    }, { input: dims.hidden, output: [-1, -1, fusedWidth] }),
    operatorSpec(`${prefix}.fused_qkvg_split`, "fused qkvg split", "kimi_fused_qkvg_split", {
      ...shapeFlow(fusedShape, `${qkvShape}, ${qkvShape}, ${qkvShape}, ${qkvShape}`),
      split_sizes: [projection, projection, projection, projection],
    }, { input: [-1, -1, fusedWidth], output: [-1, -1, 3 * projection] }),
    operatorSpec(`${prefix}.qkv_conv1d`, "merged qkv causal short convolution", "causal_conv1d", {
      ...shapeFlow(qkvFlatShape, qkvFlatShape),
      kernel_size: normalized.linearConvKernelSize,
      activation: "silu",
      branches: ["q", "k", "v"],
      merged_channels: 3 * projection,
    }, { input: [-1, -1, 3 * projection], output: [-1, -1, 3 * projection] }),
    operatorSpec(`${prefix}.b_proj`, "beta projection", "linear", {
      ...shapeFlow(shapes.hidden, betaShape),
      output_size: heads,
      activation: "sigmoid_in_kda_kernel",
    }, { input: dims.hidden, output: [-1, -1, heads] }),
    operatorSpec(`${prefix}.f_a_proj`, "forget gate feature projection", "linear", {
      ...shapeFlow(shapes.hidden, gateFeatureShape),
      output_size: headDim,
      replicated: true,
    }, { input: dims.hidden, output: gateFeature }),
    operatorSpec(`${prefix}.f_b_proj`, "forget gate projection", "linear", {
      ...shapeFlow(gateFeatureShape, qkvShape),
      output_size: projection,
      gate_role: "raw decay logits",
    }, { input: gateFeature, output: flatProjection }),
    operatorSpec(`${prefix}.A_log`, "A_log decay parameter", "kda_decay", {
      ...shapeFlow("[linear heads]", "[linear heads]"),
      parameter_role: "per-head log decay scale",
      checkpoint_shape: [headDim],
      runtime_shape: [heads],
    }, { input: [heads], output: [heads] }),
    operatorSpec(`${prefix}.dt_bias`, "dt bias parameter", "kda_decay", {
      ...shapeFlow("[linear heads, head dimension]", "[linear heads, head dimension]"),
      parameter_role: "per-head-per-channel decay bias",
      parameter_shape: [heads, headDim],
    }, { input: [heads, headDim], output: [heads, headDim] }),
    operatorSpec(`${prefix}.state_update`, "Kimi KDA recurrent state", "kimi_kda", {
      ...shapeFlow(`${qkvShape}, ${betaShape}, ${stateShape}`, qkvShape),
      attention_kind: "linear",
      mode: "chunk_prefill_or_fused_recurrent",
      qk_l2norm: true,
      beta_activation: "sigmoid",
      safe_gate: true,
      gate_lower_bound: normalized.linearLowerBound,
      state_shape: stateShape,
    }, { input: qkv, output: qkv }),
    operatorSpec(`${prefix}.o_norm`, "Kimi gated RMSNorm", "kimi_kda_output_gate", {
      ...shapeFlow(qkvShape, qkvShape),
      gate_shape: qkvShape,
      activation: "sigmoid",
    }, { input: qkv, output: qkv }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", shapeFlow(qkvShape, shapes.hidden), { input: qkv, output: dims.hidden }),
  ];
}

function glm5NextLinearAttentionOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const heads = normalized.linearKeyHeads || normalized.attentionHeads || 0;
  const headDim = normalized.linearKeyDim || normalized.headDim || 0;
  const projection = heads * headDim;
  const fusedWidth = 3 * projection + heads + 2 * headDim;
  const fusedShape = `[batch, sequence, fused qkvbfg_a=${fusedWidth}]`;
  const qkvShape = `[batch, sequence, linear heads=${heads}, head dimension=${headDim}]`;
  const betaShape = `[batch, sequence, linear heads=${heads}]`;
  const gateFeatureShape = `[batch, sequence, gate feature dimension=${headDim}]`;
  const stateShape = `[batch, linear heads=${heads}, state value dimension=${headDim}, state key dimension=${headDim}]`;
  const flatProjection = [-1, -1, projection];
  const qkv = [-1, -1, heads, headDim];
  const beta = [-1, -1, heads];
  const gateFeature = [-1, -1, headDim];
  return [
    operatorSpec(`${prefix}.in_proj_qkvbfg_a`, "fused qkvbfg_a projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: ["q", "k", "v", "beta", "f_a", "g_a"],
      qkv_projection_size: 3 * projection,
      beta_size: heads,
      gate_feature_size: headDim,
    }, { input: dims.hidden, output: [-1, -1, fusedWidth] }),
    operatorSpec(`${prefix}.qkvbfg_a_split`, "qkvbfg_a split", "split", {
      ...shapeFlow(fusedShape, `${qkvShape}, ${betaShape}, ${gateFeatureShape}, ${gateFeatureShape}`),
      split_sizes: [3 * projection, heads, headDim, headDim],
      qkv_split: [projection, projection, projection],
    }, { input: [-1, -1, fusedWidth], output: [-1, -1, 3 * projection] }),
    operatorSpec(`${prefix}.q_conv1d`, "q causal short convolution", "causal_conv1d", {
      ...shapeFlow(qkvShape, qkvShape),
      kernel_size: normalized.linearConvKernelSize,
      activation: "silu",
      branch: "q",
    }, { input: qkv, output: qkv }),
    operatorSpec(`${prefix}.k_conv1d`, "k causal short convolution", "causal_conv1d", {
      ...shapeFlow(qkvShape, qkvShape),
      kernel_size: normalized.linearConvKernelSize,
      activation: "silu",
      branch: "k",
    }, { input: qkv, output: qkv }),
    operatorSpec(`${prefix}.v_conv1d`, "v causal short convolution", "causal_conv1d", {
      ...shapeFlow(qkvShape, qkvShape),
      kernel_size: normalized.linearConvKernelSize,
      activation: "silu",
      branch: "v",
    }, { input: qkv, output: qkv }),
    operatorSpec(`${prefix}.f_b_proj`, "forget gate projection", "linear", {
      ...shapeFlow(gateFeatureShape, qkvShape),
      output_size: projection,
      gate_role: "raw decay logits",
    }, { input: gateFeature, output: flatProjection }),
    operatorSpec(`${prefix}.g_b_proj`, "output gate projection", "linear", {
      ...shapeFlow(gateFeatureShape, qkvShape),
      output_size: projection,
      gate_role: "gated RMSNorm input",
    }, { input: gateFeature, output: flatProjection }),
    operatorSpec(`${prefix}.A_log`, "A_log decay parameter", "kda_decay", {
      ...shapeFlow("[linear heads]", "[linear heads]"),
      parameter_role: "per-head log decay scale",
      parameter_shape: [heads],
    }, { input: [heads], output: [heads] }),
    operatorSpec(`${prefix}.dt_bias`, "dt bias parameter", "kda_decay", {
      ...shapeFlow("[linear heads, head dimension]", "[linear heads, head dimension]"),
      parameter_role: "per-head-per-channel decay bias",
      parameter_shape: [heads, headDim],
    }, { input: [heads, headDim], output: [heads, headDim] }),
    operatorSpec(`${prefix}.state_update`, "gated delta recurrent state", "gated_delta_attention", {
      ...shapeFlow(`${qkvShape}, ${betaShape}, ${stateShape}`, qkvShape),
      attention_kind: "linear",
      mode: "chunk_prefill_or_fused_recurrent",
      qk_l2norm: true,
      beta_activation: "sigmoid",
      safe_gate: true,
      gate_lower_bound: normalized.linearLowerBound,
      state_shape: stateShape,
    }, { input: qkv, output: qkv }),
    operatorSpec(`${prefix}.o_norm`, "gated RMSNorm", "gated_rmsnorm", {
      ...shapeFlow(qkvShape, qkvShape),
      gate_shape: qkvShape,
      activation: "sigmoid",
    }, { input: qkv, output: qkv }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", shapeFlow(qkvShape, shapes.hidden), { input: qkv, output: dims.hidden }),
  ];
}

function kdaLinearAttentionOperatorSpecs(prefix, normalized, stateFormula, gateFormula) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return [
    operatorSpec(`${prefix}.q_proj`, "q projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.k_proj`, "k projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.v_proj`, "v projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.q_conv1d`, "q short convolution", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.k_conv1d`, "k short convolution", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.v_conv1d`, "v short convolution", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.f_a_proj`, "gate feature projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.f_b_proj`, "gate expansion projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.b_proj`, "decay projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.state_update`, "gated delta state update", stateFormula, {
      ...shapeFlow(`${shapes.hidden}, state`, shapes.hidden), attention_kind: "linear", mode: "chunk_or_fused_recurrent",
    }, { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.g_proj`, "output gate", gateFormula, shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden }),
  ];
}

export function mlaAttentionOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const specs = [];
  if (normalized.qLoraRank != null) {
    specs.push(operatorSpec(`${prefix}.q_a_proj`, "query down projection", "mla_query_compress", shapeFlow(shapes.hidden, `[batch, sequence, q latent=${normalized.qLoraRank}]`), { input: dims.hidden, output: [-1, -1, normalized.qLoraRank] }));
    specs.push(operatorSpec(`${prefix}.q_b_proj`, "query up projection", "linear", shapeFlow(`[batch, sequence, q latent=${normalized.qLoraRank}]`, shapes.attentionQuery), { input: [-1, -1, normalized.qLoraRank], output: dims.attentionQuery }));
  } else {
    specs.push(operatorSpec(`${prefix}.q_proj`, "q projection", "linear", shapeFlow(shapes.hidden, shapes.attentionQuery), { input: dims.hidden, output: dims.attentionQuery }));
  }
  specs.push(operatorSpec(`${prefix}.kv_a_proj`, "KV compression projection", "mla_kv_compress", shapeFlow(shapes.hidden, `[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}]`), { input: dims.hidden, output: [-1, -1, normalized.kvLoraRank] }));
  specs.push(operatorSpec(`${prefix}.kv_b_proj`, "KV expansion projection", "linear", shapeFlow(`[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}]`, `${shapes.attentionKey}, ${shapes.attentionValue}`), { input: [-1, -1, normalized.kvLoraRank], output: dims.attentionKey }));
  specs.push(operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope", {
    ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, `${shapes.attentionQuery}, ${shapes.attentionKey}`),
    query_shape: shapes.attentionQuery,
    key_shape: shapes.attentionKey,
  }, { input: dims.attentionQuery, output: dims.attentionQuery }));
  specs.push(operatorSpec(`${prefix}.scores`, "latent attention scores", "matmul", {
    ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, shapes.attentionScores),
    formula: "S = Q K^T / sqrt(d_rope)",
    attention_kind: "mla",
  }, { input: dims.attentionQuery, output: dims.attentionScores }));
  specs.push(operatorSpec(`${prefix}.softmax`, "attention probabilities", "softmax", shapeFlow(shapes.attentionScores, shapes.attentionProbabilities), { input: dims.attentionScores, output: dims.attentionProbabilities }));
  specs.push(operatorSpec(`${prefix}.context`, "weighted value", "matmul", {
    ...shapeFlow(`${shapes.attentionProbabilities}, ${shapes.attentionValue}`, shapes.attentionContext),
    formula: "O = P V",
    attention_kind: "mla",
  }, { input: dims.attentionProbabilities, output: dims.attentionContext }));
  if (normalized.mlaUseOutputGate) {
    specs.push(operatorSpec(`${prefix}.g_proj`, "MLA output gate", "mla_output_gate", shapeFlow(shapes.hidden, shapes.attentionContext), { input: dims.hidden, output: dims.attentionContext }));
  }
  specs.push(operatorSpec(`${prefix}.o_proj`, "output projection", "linear", shapeFlow(shapes.attentionContext, shapes.hidden), { input: dims.attentionContext, output: dims.hidden }));
  return specs;
}

export function qsaAttentionOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const indexerHeads = normalized.indexerNHeads || 0;
  const indexerKVHeads = normalized.indexerKVHeads || 0;
  const indexerDim = normalized.indexerHeadDim || 0;
  const budget = normalized.indexerBudget || 0;
  return [
    operatorSpec(`${prefix}.qkv_proj`, "QSA qkv and output-gate projection", "linear", shapeFlow(shapes.hidden, shapes.attentionQuery), { input: dims.hidden, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.q_norm`, "Q attention norm", "rmsnorm", shapeFlow(shapes.attentionQuery, shapes.attentionQuery), { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.k_norm`, "K attention norm", "rmsnorm", shapeFlow(shapes.attentionKey, shapes.attentionKey), { input: dims.attentionKey, output: dims.attentionKey }),
    operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, `${shapes.attentionQuery}, ${shapes.attentionKey}`),
      query_shape: shapes.attentionQuery,
      key_shape: shapes.attentionKey,
    }, { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.indexer`, "QSA indexer", "qsa_indexer", {
      ...shapeFlow(shapes.hidden, `[batch, sequence, selected=${budget}]`),
      indexer_heads: indexerHeads,
      indexer_kv_heads: indexerKVHeads,
      indexer_head_dim: indexerDim,
      budget,
      compress_ratio: normalized.indexerCompressRatio,
    }, { input: dims.hidden, output: [-1, -1, budget] }),
    operatorSpec(`${prefix}.sparse_attention`, "QSA sparse attention", "qsa_attention", {
      ...shapeFlow(`${shapes.attentionQuery}, selected K/V`, shapes.attentionContext),
      selected_tokens: budget,
      attention_kind: "qsa",
    }, { input: dims.attentionQuery, output: dims.attentionContext }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", shapeFlow(shapes.attentionContext, shapes.hidden), { input: dims.attentionContext, output: dims.hidden }),
  ];
}

export function mlpOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return [
    operatorSpec(`${prefix}.gate_proj`, "gate projection", "linear", shapeFlow(shapes.hidden, shapes.intermediate), { input: dims.hidden, output: dims.intermediate }),
    operatorSpec(`${prefix}.up_proj`, "up projection", "linear", shapeFlow(shapes.hidden, shapes.intermediate), { input: dims.hidden, output: dims.intermediate }),
    operatorSpec(`${prefix}.swiglu`, "SwiGLU activation", "swiglu", {
      ...shapeFlow(`${shapes.intermediate}, ${shapes.intermediate}`, shapes.intermediate),
      gate_shape: shapes.intermediate,
      up_shape: shapes.intermediate,
    }, { input: dims.intermediate, output: dims.intermediate }),
    operatorSpec(`${prefix}.down_proj`, "down projection", "linear", shapeFlow(shapes.intermediate, shapes.hidden), { input: dims.intermediate, output: dims.hidden }),
  ];
}

export function moeOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return [
    operatorSpec(`${prefix}.router`, "router logits", "linear", shapeFlow(shapes.hidden, shapes.routerLogits), { input: dims.hidden, output: dims.routerLogits }),
    operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk", {
      ...shapeFlow(shapes.routerLogits, `${shapes.topExperts}, ${shapes.topExperts}`),
      expert_ids_shape: shapes.topExperts,
      expert_weights_shape: shapes.topExperts,
    }, { input: dims.routerLogits, output: dims.topExperts }),
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.topExperts}`, shapes.expertInput),
      token_shape: shapes.hidden,
      expert_ids_shape: shapes.topExperts,
    }, { input: dims.hidden, output: dims.expertInput }),
    operatorSpec(`${prefix}.expert_mlp`, "expert MLP", "swiglu", {
      ...shapeFlow(shapes.expertInput, shapes.expertInput),
      intermediate_shape: shapes.moeIntermediate,
    }, { input: dims.expertInput, output: dims.expertInput }),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine", {
      ...shapeFlow(`${shapes.expertInput}, ${shapes.topExperts}`, shapes.hidden),
      expert_output_shape: shapes.expertInput,
      expert_weights_shape: shapes.topExperts,
    }, { input: dims.expertInput, output: dims.hidden }),
  ];
}

export function kimiK3MoeOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const latent = normalized.routedExpertHiddenSize;
  const latentShape = `[tokens_per_expert, routed expert hidden size=${latent}]`;
  const latentDims = [-1, latent];
  return [
    operatorSpec(`${prefix}.router`, "router logits", "linear", shapeFlow(shapes.hidden, shapes.routerLogits), { input: dims.hidden, output: dims.routerLogits }),
    operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk", {
      ...shapeFlow(shapes.routerLogits, `${shapes.topExperts}, ${shapes.topExperts}`),
      expert_ids_shape: shapes.topExperts,
      expert_weights_shape: shapes.topExperts,
    }, { input: dims.routerLogits, output: dims.topExperts }),
    operatorSpec(`${prefix}.routed_expert_down_proj`, "routed expert latent down projection", "linear", {
      ...shapeFlow(shapes.hidden, latentShape),
      latent_size: latent,
      semantic_role: "latent_moe_compress",
    }, { input: dims.hidden, output: [-1, latent] }),
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch", {
      ...shapeFlow(`${latentShape}, ${shapes.topExperts}`, latentShape),
      token_shape: latentShape,
      expert_ids_shape: shapes.topExperts,
    }, { input: latentDims, output: latentDims }),
    operatorSpec(`${prefix}.expert_mlp`, "latent expert MLP", "swiglu", {
      ...shapeFlow(latentShape, latentShape),
      intermediate_shape: shapes.moeIntermediate,
      latent_size: latent,
    }, { input: latentDims, output: latentDims }),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine", {
      ...shapeFlow(`${latentShape}, ${shapes.topExperts}`, latentShape),
      expert_output_shape: latentShape,
      expert_weights_shape: shapes.topExperts,
    }, { input: latentDims, output: latentDims }),
    operatorSpec(`${prefix}.routed_expert_norm`, "routed expert latent RMSNorm", "rmsnorm", {
      ...shapeFlow(latentShape, latentShape),
      semantic_role: "latent_moe_reduce_norm",
    }, { input: latentDims, output: latentDims }),
    operatorSpec(`${prefix}.routed_expert_up_proj`, "routed expert latent up projection", "linear", {
      ...shapeFlow(latentShape, shapes.hidden),
      latent_size: latent,
      semantic_role: "latent_moe_expand",
    }, { input: latentDims, output: dims.hidden }),
    operatorSpec(`${prefix}.shared_expert_add`, "shared expert branch add", "moe_add", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.hidden}`, shapes.hidden),
      shared_experts: normalized.sharedExperts,
    }, { input: dims.hidden, output: dims.hidden }),
  ];
}
