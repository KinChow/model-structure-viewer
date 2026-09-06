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
  if (normalized.linearAttentionMode === "qwen3_5") {
    return canonicalKdaOperatorSpecs(prefix, normalized, "qwen3_5");
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
  const qwen = modelKind === "qwen4_exp" || modelKind === "qwen3_5";
  const qkvFlat = qwen ? 2 * keyProjection + 2 * valueProjection : 2 * keyProjection + valueProjection;
  const qkvConvFlat = 2 * keyProjection + valueProjection;
  const qkvShape = qwen
    ? `[batch, sequence, Q/K=${keyHeads}x${keyDim}, V=${valueHeads}x${valueDim}]`
    : `[batch, sequence, linear heads=${keyHeads}, head dimension=${keyDim}]`;
  const betaShape = `[batch, sequence, value heads=${valueHeads}]`;
  const gateShape = qwen
    ? `[batch, sequence, value heads=${valueHeads}, value dimension=${valueDim}]`
    : qkvShape;
  const stateShape = `[batch, value heads=${valueHeads}, state value dimension=${valueDim}, state key dimension=${keyDim}]`;
  const qkvDims = qwen ? [-1, -1, qkvFlat] : [-1, -1, keyHeads, keyDim];
  const qkvConvDims = qwen ? [-1, -1, qkvConvFlat] : qkvDims;
  const outputDims = qwen ? [-1, -1, valueHeads, valueDim] : qkvDims;
  const betaDims = [-1, -1, valueHeads];
  const fullRank = modelKind === "kimi_k3";
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
  const specs = [
    operatorSpec(`${prefix}.qkv_projection`, "QKV projection", "linear", {
      ...shapeFlow(shapes.hidden, qwen ? `[batch, sequence, fused qkvz=${qkvFlat}]` : qkvShape),
      semantic_role: "q_k_v_projection",
      implementation,
      projection_size: qwen ? { qk: keyProjection, v: valueProjection, z: valueProjection } : keyProjection,
      fused_projection_layout: projectionLayout,
    }, { input: dims.hidden, output: qkvDims }),
    ...(qwen ? [operatorSpec(`${prefix}.qkvz_split`, "qkvz split", "qwen_qkvz_split", {
      ...shapeFlow(`[batch, sequence, fused qkvz=${qkvFlat}]`, `${qkvShape}, ${qkvShape}, ${qkvShape}, ${gateShape}`),
      split_sizes: [keyProjection, keyProjection, valueProjection, valueProjection],
      implementation: ["vLLM.QwenGatedDeltaNetAttention.fix_query_key_value_ordering", "SGLang.Qwen3_5GatedDeltaNet.fix_query_key_value_ordering"],
    }, { input: qkvDims, output: qkvConvDims })] : []),
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
      channel_layout: qwen ? { q: keyProjection, k: keyProjection, v: valueProjection, z: valueProjection } : undefined,
    }, { input: qkvConvDims, output: qkvConvDims }),
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
      activation: normalized.outputGateType || "sigmoid",
    }, { input: outputDims, output: outputDims }),
    operatorSpec(`${prefix}.out_proj`, "output projection", "linear", {
      ...shapeFlow(gateShape, shapes.hidden),
      semantic_role: "attention_output_projection",
    }, { input: outputDims, output: dims.hidden }),
  ];
  return specs;
}

export function qwen35FullAttentionOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const qProjection = (normalized.attentionHeads || 0) * (normalized.headDim || 0);
  const kvProjection = (normalized.kvHeads || normalized.attentionHeads || 0) * (normalized.headDim || 0);
  const fusedWidth = 2 * qProjection + 2 * kvProjection;
  const fusedShape = `[batch, sequence, fused qkv+gate=${fusedWidth}]`;
  const qShape = shapes.attentionQuery;
  const kShape = shapes.attentionKey;
  const vShape = shapes.attentionValue;
  const gateShape = qShape;
  const qkvDims = [-1, -1, fusedWidth];
  return [
    operatorSpec(`${prefix}.qkv_gate_proj`, "fused QKV + attention gate projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: ["q", "gate", "k", "v"],
      implementation: ["vLLM.Qwen3NextAttention.qkv_proj", "SGLang.Qwen3_5Attention.qkv_proj"],
    }, { input: dims.hidden, output: qkvDims }),
    operatorSpec(`${prefix}.qkv_gate_split`, "QKV + gate split", "split", {
      ...shapeFlow(fusedShape, `${qShape}, ${gateShape}, ${kShape}, ${vShape}`),
      split_sizes: [qProjection, qProjection, kvProjection, kvProjection],
    }, { input: qkvDims, output: [-1, -1, qProjection] }),
    operatorSpec(`${prefix}.q_norm`, "Q attention Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(qShape, qShape), { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.k_norm`, "K attention Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(kShape, kShape), { input: dims.attentionKey, output: dims.attentionKey }),
    operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope", {
      ...shapeFlow(`${qShape}, ${kShape}`, `${qShape}, ${kShape}`),
      partial_rotary_factor: normalized.partialRotaryFactor,
    }, { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.scores`, "attention scores", "matmul", {
      ...shapeFlow(`${qShape}, ${kShape}`, shapes.attentionScores),
      formula: "S = Q K^T / sqrt(d)",
      attention_kind: "qwen35_full",
    }, { input: dims.attentionQuery, output: dims.attentionScores }),
    operatorSpec(`${prefix}.softmax`, "attention probabilities", "softmax", shapeFlow(shapes.attentionScores, shapes.attentionProbabilities), { input: dims.attentionScores, output: dims.attentionProbabilities }),
    operatorSpec(`${prefix}.context`, "weighted value", "matmul", {
      ...shapeFlow(`${shapes.attentionProbabilities}, ${vShape}`, shapes.attentionContext),
      formula: "O = P V",
      attention_kind: "qwen35_full",
    }, { input: dims.attentionProbabilities, output: dims.attentionContext }),
    operatorSpec(`${prefix}.output_gate`, "attention output gate", "attention_output_gate", {
      ...shapeFlow(`${shapes.attentionContext}, ${gateShape}`, shapes.attentionContext),
      activation: normalized.attentionOutputGate ? "sigmoid" : "none",
      implementation: ["vLLM.fused_sigmoid_mul", "SGLang.fused_sigmoid_mul"],
    }, { input: dims.attentionContext, output: dims.attentionContext }),
    operatorSpec(`${prefix}.o_proj`, "output projection", "linear", shapeFlow(shapes.attentionContext, shapes.hidden), { input: dims.attentionContext, output: dims.hidden }),
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
    specs.push(operatorSpec(`${prefix}.q_a_norm`, "query latent RMSNorm", "rmsnorm", shapeFlow(`[batch, sequence, q latent=${normalized.qLoraRank}]`, `[batch, sequence, q latent=${normalized.qLoraRank}]`), { input: [-1, -1, normalized.qLoraRank], output: [-1, -1, normalized.qLoraRank] }));
    specs.push(operatorSpec(`${prefix}.q_b_proj`, "query up projection", "linear", shapeFlow(`[batch, sequence, q latent=${normalized.qLoraRank}]`, shapes.attentionQuery), { input: [-1, -1, normalized.qLoraRank], output: dims.attentionQuery }));
  } else {
    specs.push(operatorSpec(`${prefix}.q_proj`, "q projection", "linear", shapeFlow(shapes.hidden, shapes.attentionQuery), { input: dims.hidden, output: dims.attentionQuery }));
  }
  specs.push(operatorSpec(`${prefix}.kv_a_proj`, "KV compression projection", "mla_kv_compress", shapeFlow(shapes.hidden, `[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"} + rope=${normalized.qkRopeHeadDim ?? "unknown"}]`), { input: dims.hidden, output: [-1, -1, (normalized.kvLoraRank || 0) + (normalized.qkRopeHeadDim || 0)] }));
  specs.push(operatorSpec(`${prefix}.kv_split`, "KV latent and rope split", "mla_kv_split", {
    ...shapeFlow(`[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"} + rope=${normalized.qkRopeHeadDim ?? "unknown"}]`, `[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}], [batch, sequence, rope=${normalized.qkRopeHeadDim ?? "unknown"}]`),
    split_sizes: [normalized.kvLoraRank, normalized.qkRopeHeadDim],
  }, { input: [-1, -1, (normalized.kvLoraRank || 0) + (normalized.qkRopeHeadDim || 0)], output: [-1, -1, normalized.kvLoraRank] }));
  specs.push(operatorSpec(`${prefix}.kv_a_norm`, "KV latent RMSNorm", "rmsnorm", shapeFlow(`[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}]`, `[batch, sequence, kv latent=${normalized.kvLoraRank ?? "unknown"}]`), { input: [-1, -1, normalized.kvLoraRank], output: [-1, -1, normalized.kvLoraRank] }));
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

// DeepSeek V4 的 vLLM/SGLang 实现使用不同 fused kernel，但语义都是同一条 MLA 链。
// 这里保留一份 canonical 节点，通过 implementation 与 compress_ratio 描述实现差异。
export function deepseekV4AttentionOperatorSpecs(prefix, normalized, layerIndex = 0) {
  const dims = tensorDims(normalized);
  const ratio = normalized.compressRatios?.[layerIndex] ?? 0;
  const qRank = normalized.qLoraRank;
  const headDim = normalized.headDim;
  const groups = normalized.oGroups;
  const outputRank = normalized.oLoraRank;
  const indexHeads = normalized.indexerNHeads;
  const indexDim = normalized.indexerHeadDim;
  const budget = normalized.indexerBudget;
  const qLatent = `[batch, sequence, q latent=${qRank ?? "unknown"}]`;
  const kvLatent = `[batch, sequence, kv latent=${headDim ?? "unknown"}]`;
  const query = `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim ?? "unknown"}]`;
  const outputLatent = `[batch, sequence, output groups=${groups ?? "unknown"}, output rank=${outputRank ?? "unknown"}]`;
  const specs = [
    operatorSpec(`${prefix}.fused_wqa_wkv`, "fused q/kv projection", "linear", {
      ...shapeFlow(shapesForHidden(normalized), `${qLatent}, ${kvLatent}`),
      projection_layout: ["q_lora", "kv"],
      q_lora_rank: qRank,
      kv_head_dim: headDim,
      implementation: ["vLLM.fused_wqa_wkv", "SGLang.wqkv_a or wq_a+wkv"],
    }, { input: dims.hidden, output: [-1, -1, (qRank || 0) + (headDim || 0)] }),
    operatorSpec(`${prefix}.qkv_split`, "q/kv latent split", "split", {
      ...shapeFlow(`${qLatent}, ${kvLatent}`, `${qLatent}, ${kvLatent}`),
      split_sizes: [qRank, headDim],
      implementation: ["vLLM.split_qkv_and_norm", "SGLang._compute_q_a/_compute_kv"],
    }, { input: [-1, -1, (qRank || 0) + (headDim || 0)], output: [-1, -1, qRank || 0] }),
    operatorSpec(`${prefix}.q_norm`, "query latent RMSNorm", "rmsnorm", shapeFlow(qLatent, qLatent), { input: [-1, -1, qRank], output: [-1, -1, qRank] }),
    operatorSpec(`${prefix}.kv_norm`, "KV latent RMSNorm", "rmsnorm", shapeFlow(kvLatent, kvLatent), { input: [-1, -1, headDim], output: [-1, -1, headDim] }),
    operatorSpec(`${prefix}.q_proj`, "query expansion projection", "linear", {
      ...shapeFlow(qLatent, query),
      projection_role: "wq_b",
      implementation: ["vLLM.wq_b", "SGLang.wq_b"],
    }, { input: [-1, -1, qRank], output: [-1, -1, normalized.attentionHeads, headDim] }),
    operatorSpec(`${prefix}.rope`, "query/KV rotary position embedding", "rope", {
      ...shapeFlow(`${query}, ${kvLatent}`, `${query}, ${kvLatent}`),
      qk_rope_head_dim: normalized.qkRopeHeadDim,
      compress_ratio: ratio,
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }),
  ];

  if (ratio > 1) {
    specs.push(operatorSpec(`${prefix}.compressor`, "compressed KV/state compressor", "mla_kv_compress", {
      ...shapeFlow(shapesForHidden(normalized), `[compressed sequence=ceil(sequence/${ratio}), state dimension]`),
      compress_ratio: ratio,
      implementation: ["vLLM.DeepseekCompressor", "SGLang.Compressor"],
      cache_role: "compressed_kv_and_score_state",
    }, { input: dims.hidden, output: [-1, -1, 2 * (ratio === 4 ? 2 : 1) * headDim] }));
  }

  if (ratio === 4) {
    specs.push(operatorSpec(`${prefix}.indexer.weights_proj`, "indexer weight projection", "linear", {
      ...shapeFlow(shapesForHidden(normalized), `[batch, sequence, index heads=${indexHeads}]`),
      implementation: ["vLLM.DeepseekV4Indexer.weights_proj", "SGLang.C4Indexer"],
    }, { input: dims.hidden, output: [-1, -1, indexHeads] }));
    specs.push(operatorSpec(`${prefix}.indexer.q_proj`, "indexer query projection", "linear", {
      ...shapeFlow(qLatent, `[batch, sequence, index heads=${indexHeads}, index head dimension=${indexDim}]`),
      implementation: ["vLLM.DeepseekV4Indexer.wq_b", "SGLang.C4Indexer"],
    }, { input: [-1, -1, qRank], output: [-1, -1, indexHeads, indexDim] }));
    specs.push(operatorSpec(`${prefix}.indexer`, "C4 sparse indexer", "qsa_indexer", {
      ...shapeFlow(shapesForHidden(normalized), `[batch, sequence, selected=${budget}]`),
      indexer_heads: indexHeads,
      indexer_head_dim: indexDim,
      budget,
      compress_ratio: ratio,
      implementation: ["vLLM.SparseAttnIndexer", "SGLang.C4Indexer"],
    }, { input: dims.hidden, output: [-1, -1, budget] }));
    specs.push(operatorSpec(`${prefix}.attention`, "C4 sparse MLA attention", "qsa_attention", {
      ...shapeFlow(`${query}, selected compressed KV`, `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`),
      selected_tokens: budget,
      compress_ratio: ratio,
      attention_kind: "dsv4_sparse_mla",
      implementation: ["vLLM.DeepseekV4FlashMLAAttention", "SGLang.RadixAttention + DSV4 backend"],
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }));
  } else if (ratio === 128) {
    specs.push(operatorSpec(`${prefix}.attention`, "compressed MLA attention", "dsv4_compressed_attention", {
      ...shapeFlow(`${query}, compressed KV`, `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`),
      compress_ratio: ratio,
      attention_kind: "dsv4_compressed_mla",
      implementation: ["vLLM.DeepseekV4FlashMLAAttention", "SGLang.MQALayer"],
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }));
  } else {
    specs.push(operatorSpec(`${prefix}.attention`, "sliding-window MQA", "dsv4_swa_attention", {
      ...shapeFlow(`${query}, KV window`, `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`),
      sliding_window: normalized.slidingWindow,
      compress_ratio: ratio,
      attention_kind: "dsv4_swa_mqa",
      implementation: ["vLLM.DeepseekV4SWACache", "SGLang.RadixAttention"],
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }));
  }

  specs.push(
    operatorSpec(`${prefix}.inverse_rope`, "inverse output rotary transform", "rope", {
      ...shapeFlow(`[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`, `[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`),
      phase: "output_inverse_rope",
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, normalized.attentionHeads, headDim] }),
    operatorSpec(`${prefix}.wo_a`, "output low-rank projection", "linear", {
      ...shapeFlow(`[batch, sequence, attention heads=${normalized.attentionHeads}, head dimension=${headDim}]`, outputLatent),
      projection_role: "wo_a",
      output_groups: groups,
      output_rank: outputRank,
      implementation: ["vLLM.wo_a", "SGLang.wo_a"],
    }, { input: [-1, -1, normalized.attentionHeads, headDim], output: [-1, -1, groups, outputRank] }),
    operatorSpec(`${prefix}.wo_b`, "output hidden projection", "linear", {
      ...shapeFlow(outputLatent, shapesForHidden(normalized)),
      projection_role: "wo_b",
      implementation: ["vLLM.wo_b", "SGLang.wo_b"],
    }, { input: [-1, -1, groups, outputRank], output: dims.hidden }),
  );
  return specs;
}

function shapesForHidden(normalized) {
  return `[batch, sequence, hidden size=${normalized.hiddenSize ?? "unknown"}]`;
}

export function qsaAttentionOperatorSpecs(prefix, normalized, layerIndex = 0) {
  if (["deepseek_v32", "glm_moe_dsa"].includes(normalized.modelType)) {
    return dsaAttentionOperatorSpecs(prefix, normalized, layerIndex);
  }
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

function minimaxAttentionCommon(prefix, normalized, sparse, layerIndex = 0) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const heads = normalized.attentionHeads || 0;
  const kvHeads = normalized.kvHeads || heads;
  const headDim = normalized.headDim || 0;
  const qProjection = heads * headDim;
  const kvProjection = kvHeads * headDim;
  const indexHeads = normalized.sparseIndexHeads || kvHeads;
  const indexDim = normalized.sparseIndexDim || headDim;
  const indexProjection = indexHeads * indexDim;
  const disableIndexValue = normalized.sparseDisableIndexValue?.[layerIndex] ?? true;
  const indexValueProjection = disableIndexValue ? 0 : indexProjection;
  const fusedWidth = qProjection + 2 * kvProjection + (sparse ? 2 * indexProjection + indexValueProjection : 0);
  const fusedShape = `[batch, sequence, fused main QKV + index QKV=${fusedWidth}]`;
  const indexShape = `[batch, sequence, index heads=${indexHeads}, index head dimension=${indexDim}]`;
  const specs = [
    operatorSpec(`${prefix}.qkv_index_proj`, sparse ? "fused QKV + index projection" : "QKV projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: sparse ? ["q", "k", "v", "index_q", "index_k", ...(disableIndexValue ? [] : ["index_v"])] : ["q", "k", "v"],
      implementation: sparse
        ? ["vLLM.MinimaxM3QKVParallelLinearWithIndexer", "SGLang._FusedQKVIndexProj"]
        : ["vLLM.QKVParallelLinear", "SGLang.qkv_proj"],
      disable_index_value: disableIndexValue,
    }, { input: dims.hidden, output: [-1, -1, fusedWidth] }),
    operatorSpec(`${prefix}.qkv_index_split`, sparse ? "main/index QKV split" : "QKV split", "split", {
      ...shapeFlow(fusedShape, sparse ? `${shapes.attentionQuery}, ${shapes.attentionKey}, ${shapes.attentionValue}, ${indexShape}, ${indexShape}` : `${shapes.attentionQuery}, ${shapes.attentionKey}, ${shapes.attentionValue}`),
      split_sizes: sparse ? [qProjection, kvProjection, kvProjection, indexProjection, indexProjection, ...(disableIndexValue ? [] : [indexValueProjection])] : [qProjection, kvProjection, kvProjection],
    }, { input: [-1, -1, fusedWidth], output: [-1, -1, qProjection] }),
    operatorSpec(`${prefix}.q_norm`, "Q Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(shapes.attentionQuery, shapes.attentionQuery), { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.k_norm`, "K Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(shapes.attentionKey, shapes.attentionKey), { input: dims.attentionKey, output: dims.attentionKey }),
    operatorSpec(`${prefix}.rope`, "partial rotary position embedding", "rope", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, `${shapes.attentionQuery}, ${shapes.attentionKey}`),
      partial_rotary_factor: normalized.partialRotaryFactor,
      implementation: ["vLLM.MiniMaxM3Attention.rotary_emb", "SGLang.MiniMaxM3Attention.rotary_emb"],
    }, { input: dims.attentionQuery, output: dims.attentionQuery }),
  ];
  if (sparse) {
    specs.push(
      operatorSpec(`${prefix}.index_q_norm`, "index Q Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(indexShape, indexShape), { input: [-1, -1, indexHeads, indexDim], output: [-1, -1, indexHeads, indexDim] }),
      operatorSpec(`${prefix}.index_k_norm`, "index K Gemma RMSNorm", "gemma_rmsnorm", shapeFlow(indexShape, indexShape), { input: [-1, -1, indexHeads, indexDim], output: [-1, -1, indexHeads, indexDim] }),
      operatorSpec(`${prefix}.index_rope`, "index partial rotary position embedding", "rope", {
        ...shapeFlow(`${indexShape}, ${indexShape}`, `${indexShape}, ${indexShape}`),
        partial_rotary_factor: normalized.partialRotaryFactor,
      }, { input: [-1, -1, indexHeads, indexDim], output: [-1, -1, indexHeads, indexDim] }),
      operatorSpec(`${prefix}.indexer`, "MiniMax M3 block indexer", "minimax_sparse_indexer", {
        ...shapeFlow(`${indexShape}, ${indexShape}`, `[batch, sequence, selected blocks=${normalized.sparseTopkBlocks}]`),
        index_heads: indexHeads,
        index_head_dim: indexDim,
        topk_blocks: normalized.sparseTopkBlocks,
        block_size: normalized.sparseBlockSize,
        init_blocks: normalized.sparseInitBlock,
        local_blocks: normalized.sparseLocalBlock,
        score_type: normalized.sparseScoreType || "max",
        disable_index_value: disableIndexValue,
        implementation: ["vLLM.MiniMaxM3Indexer", "SGLang.Minimax sparse indexer"],
      }, { input: [-1, -1, indexHeads, indexDim], output: [-1, -1, normalized.sparseTopkBlocks] }),
      operatorSpec(`${prefix}.sparse_attention`, "MiniMax M3 block-sparse GQA", "minimax_sparse_attention", {
        ...shapeFlow(`${shapes.attentionQuery}, selected KV blocks`, shapes.attentionContext),
        topk_blocks: normalized.sparseTopkBlocks,
        block_size: normalized.sparseBlockSize,
        local_blocks: normalized.sparseLocalBlock,
        init_blocks: normalized.sparseInitBlock,
        disable_index_value: disableIndexValue,
        attention_kind: "minimax_m3_sparse_gqa",
        implementation: ["vLLM.MiniMaxM3SparseImpl", "SGLang.minimax_sparse_backend"],
      }, { input: dims.attentionQuery, output: dims.attentionContext }),
    );
  } else {
    specs.push(
      operatorSpec(`${prefix}.scores`, "attention scores", "matmul", {
        ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, shapes.attentionScores),
        formula: "S = Q K^T / sqrt(d)",
      }, { input: dims.attentionQuery, output: dims.attentionScores }),
      operatorSpec(`${prefix}.softmax`, "attention probabilities", "softmax", shapeFlow(shapes.attentionScores, shapes.attentionProbabilities), { input: dims.attentionScores, output: dims.attentionProbabilities }),
      operatorSpec(`${prefix}.context`, "weighted value", "matmul", {
      ...shapeFlow(`${shapes.attentionProbabilities}, ${shapes.attentionValue}`, shapes.attentionContext),
      formula: "O = P V",
      }, { input: dims.attentionProbabilities, output: dims.attentionContext }),
    );
  }
  specs.push(operatorSpec(`${prefix}.o_proj`, "output projection", "linear", shapeFlow(shapes.attentionContext, shapes.hidden), { input: dims.attentionContext, output: dims.hidden }));
  return specs;
}

export function minimaxDenseAttentionOperatorSpecs(prefix, normalized) {
  return minimaxAttentionCommon(prefix, normalized, false);
}

export function minimaxSparseAttentionOperatorSpecs(prefix, normalized, layerIndex = 0) {
  return minimaxAttentionCommon(prefix, normalized, true, layerIndex);
}

export function minimaxM2AttentionOperatorSpecs(prefix, normalized, modelVariant = "minimax_m2") {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const qProjection = (normalized.attentionHeads || 0) * (normalized.headDim || 0);
  const kvProjection = (normalized.kvHeads || normalized.attentionHeads || 0) * (normalized.headDim || 0);
  const fusedWidth = qProjection + 2 * kvProjection;
  const fusedShape = `[batch, sequence, fused qkv=${fusedWidth}]`;
  return [
    operatorSpec(`${prefix}.qkv_proj`, "fused QKV projection", "linear", {
      ...shapeFlow(shapes.hidden, fusedShape),
      projection_layout: ["q", "k", "v"],
      bias: normalized.attentionBias,
      implementation: modelVariant === "glm4_moe"
        ? ["vLLM.Glm4MoeAttention.qkv_proj", "SGLang.Glm4MoeAttention.qkv_proj"]
        : ["vLLM.MiniMaxM2Attention.qkv_proj", "SGLang.MiniMaxM2Attention.qkv_proj"],
    }, { input: dims.hidden, output: [-1, -1, fusedWidth] }),
    operatorSpec(`${prefix}.qkv_split`, "QKV split", "attention_qkv_split", {
      ...shapeFlow(fusedShape, `${shapes.attentionQuery}, ${shapes.attentionKey}, ${shapes.attentionValue}`),
      split_sizes: [qProjection, kvProjection, kvProjection],
    }, { input: [-1, -1, fusedWidth], output: [-1, -1, qProjection] }),
    operatorSpec(`${prefix}.q_norm`, "Q RMSNorm", "rmsnorm", shapeFlow(shapes.attentionQuery, shapes.attentionQuery), {
      input: dims.attentionQuery,
      output: dims.attentionQuery,
      norm_type: normalized.qkNormType || "per_layer",
      implementation: modelVariant === "glm4_moe"
        ? ["vLLM.Glm4MoeAttention.q_norm", "SGLang.Glm4MoeAttention.q_norm"]
        : ["vLLM.MiniMaxText01RMSNormTP", "SGLang.MiniMaxM2RMSNormTP"],
    }),
    operatorSpec(`${prefix}.k_norm`, "K RMSNorm", "rmsnorm", shapeFlow(shapes.attentionKey, shapes.attentionKey), {
      input: dims.attentionKey,
      output: dims.attentionKey,
      norm_type: normalized.qkNormType || "per_layer",
      implementation: modelVariant === "glm4_moe"
        ? ["vLLM.Glm4MoeAttention.k_norm", "SGLang.Glm4MoeAttention.k_norm"]
        : ["vLLM.MiniMaxText01RMSNormTP", "SGLang.MiniMaxM2RMSNormTP"],
    }),
    operatorSpec(`${prefix}.rope`, "partial rotary position embedding", "rope", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, `${shapes.attentionQuery}, ${shapes.attentionKey}`),
      rotary_dim: normalized.rotaryDim,
      partial_rotary_factor: normalized.partialRotaryFactor,
    }, { input: dims.attentionQuery, output: dims.attentionQuery }),
    operatorSpec(`${prefix}.scores`, "attention scores", "matmul", {
      ...shapeFlow(`${shapes.attentionQuery}, ${shapes.attentionKey}`, shapes.attentionScores),
      formula: "S = Q K^T / sqrt(d)",
    }, { input: dims.attentionQuery, output: dims.attentionScores }),
    operatorSpec(`${prefix}.softmax`, "attention probabilities", "softmax", shapeFlow(shapes.attentionScores, shapes.attentionProbabilities), { input: dims.attentionScores, output: dims.attentionProbabilities }),
    operatorSpec(`${prefix}.context`, "weighted value", "matmul", {
      ...shapeFlow(`${shapes.attentionProbabilities}, ${shapes.attentionValue}`, shapes.attentionContext),
      formula: "O = P V",
    }, { input: dims.attentionProbabilities, output: dims.attentionContext }),
    operatorSpec(`${prefix}.o_proj`, "output projection", "linear", shapeFlow(shapes.attentionContext, shapes.hidden), { input: dims.attentionContext, output: dims.hidden }),
  ];
}

// DeepSeek V3.2/GLM DSA 共用一份 MLA + indexer 语义；vLLM/SGLang 的融合方式只记录在 implementation。
function dsaAttentionOperatorSpecs(prefix, normalized, layerIndex) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const heads = normalized.attentionHeads || 0;
  const qRank = normalized.qLoraRank || 0;
  const kvRank = normalized.kvLoraRank || 0;
  const qkNope = normalized.qkNopeHeadDim || 0;
  const ropeDim = normalized.qkRopeHeadDim || 0;
  const qkDim = qkNope + ropeDim;
  const valueDim = normalized.valueHeadDim || normalized.headDim || 0;
  const indexHeads = normalized.indexerNHeads || 0;
  const indexDim = normalized.indexerHeadDim || 0;
  const budget = normalized.indexerBudget || 0;
  const indexerMode = normalized.indexerSchedule?.[layerIndex] || "compute";
  const qLatentShape = `[batch, sequence, q latent=${qRank}]`;
  const kvLatentShape = `[batch, sequence, kv latent=${kvRank}, rope=${ropeDim}]`;
  const qShape = `[batch, sequence, attention heads=${heads}, head dimension=${qkDim}]`;
  const kShape = `[batch, sequence, attention heads=${heads}, head dimension=${qkDim}]`;
  const vShape = `[batch, sequence, attention heads=${heads}, value head dimension=${valueDim}]`;
  return [
    operatorSpec(`${prefix}.q_a_proj`, "query down projection", "mla_query_compress", {
      ...shapeFlow(shapes.hidden, qLatentShape),
      implementation: ["vLLM.q_a_proj", "SGLang.q_a_proj"],
    }, { input: dims.hidden, output: [-1, -1, qRank] }),
    operatorSpec(`${prefix}.q_a_norm`, "query latent RMSNorm", "rmsnorm", shapeFlow(qLatentShape, qLatentShape), { input: [-1, -1, qRank], output: [-1, -1, qRank] }),
    operatorSpec(`${prefix}.q_b_proj`, "query up projection", "linear", {
      ...shapeFlow(qLatentShape, qShape),
      implementation: ["vLLM.q_b_proj", "SGLang.q_b_proj"],
    }, { input: [-1, -1, qRank], output: [-1, -1, heads, qkDim] }),
    operatorSpec(`${prefix}.kv_a_proj`, "KV compression projection", "mla_kv_compress", {
      ...shapeFlow(shapes.hidden, kvLatentShape),
      implementation: ["vLLM.kv_a_proj_with_mqa", "SGLang.kv_a_proj_with_mqa"],
      kv_lora_rank: kvRank,
      qk_rope_head_dim: ropeDim,
    }, { input: dims.hidden, output: [-1, -1, kvRank + ropeDim] }),
    operatorSpec(`${prefix}.kv_split`, "KV latent and rope split", "mla_kv_split", {
      ...shapeFlow(kvLatentShape, `[batch, sequence, kv latent=${kvRank}], [batch, sequence, rope=${ropeDim}]`),
      split_sizes: [kvRank, ropeDim],
    }, { input: [-1, -1, kvRank + ropeDim], output: [-1, -1, kvRank] }),
    operatorSpec(`${prefix}.kv_a_norm`, "KV latent RMSNorm", "rmsnorm", shapeFlow(`[batch, sequence, kv latent=${kvRank}]`, `[batch, sequence, kv latent=${kvRank}]`), { input: [-1, -1, kvRank], output: [-1, -1, kvRank] }),
    operatorSpec(`${prefix}.kv_b_proj`, "KV expansion projection", "linear", {
      ...shapeFlow(`[batch, sequence, kv latent=${kvRank}]`, `${kShape}, ${vShape}`),
      implementation: ["vLLM.kv_b_proj", "SGLang.kv_b_proj"],
      qk_nope_head_dim: qkNope,
      value_head_dim: valueDim,
    }, { input: [-1, -1, kvRank], output: [-1, -1, heads, qkNope + valueDim] }),
    operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope", {
      ...shapeFlow(`${qShape}, ${kShape}`, `${qShape}, ${kShape}`),
      qk_rope_head_dim: ropeDim,
      implementation: ["vLLM.DeepseekV32 rotary_emb", "SGLang.Deepseek rotary_emb"],
    }, { input: [-1, -1, heads, qkDim], output: [-1, -1, heads, qkDim] }),
    operatorSpec(`${prefix}.indexer.q_proj`, "indexer query projection", "linear", {
      ...shapeFlow(qLatentShape, `[batch, sequence, index heads=${indexHeads}, index head dimension=${indexDim}]`),
      implementation: ["vLLM.Indexer.wq_b", "SGLang.Indexer.wq_b"],
    }, { input: [-1, -1, qRank], output: [-1, -1, indexHeads, indexDim] }),
    operatorSpec(`${prefix}.indexer.wk_weights_proj`, "indexer key and weight projection", "linear", {
      ...shapeFlow(shapes.hidden, `[batch, sequence, index head dimension=${indexDim}] + [batch, sequence, index heads=${indexHeads}]`),
      projection_layout: ["wk", "weights"],
      implementation: ["vLLM.Indexer.wk_weights_proj", "SGLang.Indexer.wk_weights_proj"],
    }, { input: dims.hidden, output: [-1, -1, indexDim + indexHeads] }),
    operatorSpec(`${prefix}.indexer.k_norm`, "indexer key RMSNorm", "rmsnorm", {
      ...shapeFlow(`[batch, sequence, index head dimension=${indexDim}]`, `[batch, sequence, index head dimension=${indexDim}]`),
      implementation: ["vLLM.Indexer.k_norm", "SGLang.Indexer.k_norm"],
    }, { input: [-1, -1, indexDim], output: [-1, -1, indexDim] }),
    operatorSpec(`${prefix}.indexer`, "DSA indexer", "qsa_indexer", {
      ...shapeFlow(shapes.hidden, `[batch, sequence, selected=${budget}]`),
      indexer_heads: indexHeads,
      indexer_head_dim: indexDim,
      budget,
      indexer_mode: indexerMode,
      reuse_previous_indices: indexerMode === "reuse",
      implementation: ["vLLM.SparseAttnIndexer", "SGLang.dsa_indexer"],
    }, { input: dims.hidden, output: [-1, -1, budget] }),
    operatorSpec(`${prefix}.sparse_attention`, "DSA sparse MLA attention", "qsa_attention", {
      ...shapeFlow(`${qShape}, selected ${kShape}, selected ${vShape}`, `[batch, sequence, attention heads=${heads}, value head dimension=${valueDim}]`),
      selected_tokens: budget,
      attention_kind: "dsa_sparse_mla",
      indexer_mode: indexerMode,
      implementation: ["vLLM.DeepseekV32MLAAttention", "SGLang.RadixAttention + DSA backend"],
    }, { input: [-1, -1, heads, qkDim], output: [-1, -1, heads, valueDim] }),
    operatorSpec(`${prefix}.o_proj`, "output projection", "linear", {
      ...shapeFlow(`[batch, sequence, attention heads=${heads}, value head dimension=${valueDim}]`, shapes.hidden),
      implementation: ["vLLM.o_proj", "SGLang.o_proj"],
    }, { input: [-1, -1, heads, valueDim], output: dims.hidden }),
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
      activation: normalized.modelType === "minimax_m3_vl" ? "swigluoai" : undefined,
      swiglu_alpha: normalized.swigluAlpha,
      swiglu_beta: normalized.swigluBeta,
      swiglu_limit: normalized.swigluLimit,
    }, { input: dims.intermediate, output: dims.intermediate }),
    operatorSpec(`${prefix}.down_proj`, "down projection", "linear", shapeFlow(shapes.intermediate, shapes.hidden), { input: dims.intermediate, output: dims.hidden }),
  ];
}

export function moeOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const isMiniMaxRouter = ["minimax_m2", "minimax_m3_vl", "glm4_moe"].includes(normalized.modelType);
  return [
    operatorSpec(`${prefix}.router`, "router logits", "linear", {
      ...shapeFlow(shapes.hidden, shapes.routerLogits),
      scoring_func: isMiniMaxRouter ? "sigmoid" : undefined,
      routing_bias: isMiniMaxRouter ? true : undefined,
      implementation: isMiniMaxRouter ? ["vLLM.GateLinear fp32 router", "SGLang.GateLinear fp32 router"] : undefined,
    }, { input: dims.hidden, output: dims.routerLogits }),
    operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk", {
      ...shapeFlow(shapes.routerLogits, `${shapes.topExperts}, ${shapes.topExperts}`),
      expert_ids_shape: shapes.topExperts,
      expert_weights_shape: shapes.topExperts,
      scoring_func: isMiniMaxRouter ? "sigmoid" : undefined,
    }, { input: dims.routerLogits, output: dims.topExperts }),
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.topExperts}`, shapes.expertInput),
      token_shape: shapes.hidden,
      expert_ids_shape: shapes.topExperts,
    }, { input: dims.hidden, output: dims.expertInput }),
    operatorSpec(`${prefix}.expert_mlp`, "expert MLP", "swiglu", {
      ...shapeFlow(shapes.expertInput, shapes.expertInput),
      intermediate_shape: shapes.moeIntermediate,
      activation: normalized.modelType === "minimax_m3_vl" ? "swigluoai_uninterleave" : undefined,
      swiglu_alpha: normalized.swigluAlpha,
      swiglu_beta: normalized.swigluBeta,
      swiglu_limit: normalized.swigluLimit,
    }, { input: dims.expertInput, output: dims.expertInput }),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine", {
      ...shapeFlow(`${shapes.expertInput}, ${shapes.topExperts}`, shapes.hidden),
      expert_output_shape: shapes.expertInput,
      expert_weights_shape: shapes.topExperts,
    }, { input: dims.expertInput, output: dims.hidden }),
  ];
}

// DeepSeek V4 的 hash 层和普通 MoE 共用同一 routed/shared expert 语义；差异只在路由节点。
export function deepseekV4MoeOperatorSpecs(prefix, normalized, isHashMoe = false) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const specs = isHashMoe
    ? [operatorSpec(`${prefix}.hash_router`, "input-id hash expert routing", "dsv4_hash_route", {
      ...shapeFlow("[batch, sequence] input_ids", shapes.topExperts),
      num_hash_layers: normalized.numHashLayers,
      hash_table_shape: `[vocab size=${normalized.vocabSize}, experts per token=${normalized.expertsPerToken}]`,
      implementation: ["vLLM.gate.tid2eid + fused_topk_bias", "SGLang DeepSeek V4 hash routing"],
    }, { input: [-1, -1], output: dims.topExperts })]
    : [
      operatorSpec(`${prefix}.router`, "router logits", "linear", {
        ...shapeFlow(shapes.hidden, shapes.routerLogits),
        scoring_func: "sqrtsoftplus",
        routed_scaling_factor: normalized.routedScalingFactor,
        implementation: ["vLLM.GateLinear + fused_topk_bias", "SGLang fused_moe"],
      }, { input: dims.hidden, output: dims.routerLogits }),
      operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk", {
        ...shapeFlow(shapes.routerLogits, `${shapes.topExperts}, ${shapes.topExperts}`),
        expert_ids_shape: shapes.topExperts,
        expert_weights_shape: shapes.topExperts,
        scoring_func: "sqrtsoftplus",
        renormalize: normalized.normTopkProb,
      }, { input: dims.routerLogits, output: dims.topExperts }),
    ];
  specs.push(
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.topExperts}`, shapes.expertInput),
      token_shape: shapes.hidden,
      expert_ids_shape: shapes.topExperts,
      implementation: ["vLLM.FusedMoE", "SGLang fused_moe"],
    }, { input: dims.hidden, output: dims.expertInput }),
    operatorSpec(`${prefix}.expert_mlp`, "expert SwiGLU", "swiglu", {
      ...shapeFlow(shapes.expertInput, shapes.expertInput),
      intermediate_shape: shapes.moeIntermediate,
      swiglu_limit: normalized.swigluLimit,
      implementation: ["vLLM.DeepseekV4MegaMoEExperts", "SGLang fused_moe"],
    }, { input: dims.expertInput, output: dims.expertInput }),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine", {
      ...shapeFlow(`${shapes.expertInput}, ${shapes.topExperts}`, shapes.hidden),
      expert_output_shape: shapes.expertInput,
      expert_weights_shape: shapes.topExperts,
      routed_scaling_factor: normalized.routedScalingFactor,
    }, { input: dims.expertInput, output: dims.hidden }),
  );
  return specs;
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
