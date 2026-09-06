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
  if (normalized.linearAttentionMode === "kimi") {
    const kdaFormula = "kimi_kda";
    const gateFormula = "kimi_kda_output_gate";
    return kdaLinearAttentionOperatorSpecs(prefix, normalized, kdaFormula, gateFormula);
  }
  if (normalized.linearAttentionMode === "glm5_next") {
    return glm5NextLinearAttentionOperatorSpecs(prefix, normalized);
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
