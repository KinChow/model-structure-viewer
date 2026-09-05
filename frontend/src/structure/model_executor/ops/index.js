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
