import { formulaForOperator } from "../../formulas/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

function cleanAttributes(attributes) {
  return Object.fromEntries(
    Object.entries(attributes).filter(([, value]) => value !== undefined && value !== null),
  );
}

export function operatorSpec(id, name, operatorId, attributes = {}) {
  const formula = formulaForOperator(operatorId);
  return {
    kind: "operator",
    id,
    name,
    operatorId,
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
  return [
    operatorSpec(`${prefix}.q_proj`, "q projection", "linear", shapeFlow(shapes.hidden, shapes.attentionQuery)),
    operatorSpec(`${prefix}.k_proj`, "k projection", "linear", shapeFlow(shapes.hidden, shapes.attentionKey)),
    operatorSpec(`${prefix}.v_proj`, "v projection", "linear", shapeFlow(shapes.hidden, shapes.attentionValue)),
    operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope", {
      query_shape: shapes.attentionQuery,
      key_shape: shapes.attentionKey,
      output_shape: `${shapes.attentionQuery}, ${shapes.attentionKey}`,
    }),
    operatorSpec(`${prefix}.scores`, "attention scores", "matmul", {
      attention_kind: attentionKind,
      query_shape: shapes.attentionQuery,
      key_shape: shapes.attentionKey,
      output_shape: shapes.attentionScores,
    }),
    operatorSpec(
      `${prefix}.softmax`,
      "attention probabilities",
      "softmax",
      shapeFlow(shapes.attentionScores, shapes.attentionProbabilities),
    ),
    operatorSpec(`${prefix}.context`, "weighted value", "matmul", {
      probabilities_shape: shapes.attentionProbabilities,
      value_shape: shapes.attentionValue,
      output_shape: shapes.attentionContext,
    }),
    operatorSpec(`${prefix}.o_proj`, "output projection", "linear", shapeFlow(shapes.attentionContext, shapes.hidden)),
  ];
}

export function mlpOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  return [
    operatorSpec(`${prefix}.gate_proj`, "gate projection", "linear", shapeFlow(shapes.hidden, shapes.intermediate)),
    operatorSpec(`${prefix}.up_proj`, "up projection", "linear", shapeFlow(shapes.hidden, shapes.intermediate)),
    operatorSpec(`${prefix}.swiglu`, "SwiGLU activation", "swiglu", {
      gate_shape: shapes.intermediate,
      up_shape: shapes.intermediate,
      output_shape: shapes.intermediate,
    }),
    operatorSpec(`${prefix}.down_proj`, "down projection", "linear", shapeFlow(shapes.intermediate, shapes.hidden)),
  ];
}

export function moeOperatorSpecs(prefix, normalized) {
  const shapes = tensorShapes(normalized);
  return [
    operatorSpec(`${prefix}.router`, "router logits", "linear", shapeFlow(shapes.hidden, shapes.routerLogits)),
    operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk", {
      input_shape: shapes.routerLogits,
      expert_ids_shape: shapes.topExperts,
      expert_weights_shape: shapes.topExperts,
    }),
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch", {
      token_shape: shapes.hidden,
      expert_ids_shape: shapes.topExperts,
      output_shape: shapes.expertInput,
    }),
    operatorSpec(`${prefix}.expert_mlp`, "expert MLP", "swiglu", {
      input_shape: shapes.expertInput,
      intermediate_shape: shapes.moeIntermediate,
      output_shape: shapes.expertInput,
    }),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine", {
      expert_output_shape: shapes.expertInput,
      expert_weights_shape: shapes.topExperts,
      output_shape: shapes.hidden,
    }),
  ];
}
