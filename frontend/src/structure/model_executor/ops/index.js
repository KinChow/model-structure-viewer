import { formulaForOperator } from "../../formulas/index.js";

export function operatorSpec(id, name, operatorId, attributes = {}) {
  const formula = formulaForOperator(operatorId);
  return {
    kind: "operator",
    id,
    name,
    operatorId,
    attributes: {
      formula_id: operatorId,
      formula: formula?.formula,
      explanation: formula?.explanation,
      inputs: formula?.inputs,
      outputs: formula?.outputs,
      ...attributes,
    },
  };
}

export function attentionOperatorSpecs(prefix, attentionKind) {
  return [
    operatorSpec(`${prefix}.q_proj`, "q projection", "linear"),
    operatorSpec(`${prefix}.k_proj`, "k projection", "linear"),
    operatorSpec(`${prefix}.v_proj`, "v projection", "linear"),
    operatorSpec(`${prefix}.rope`, "rotary position embedding", "rope"),
    operatorSpec(`${prefix}.scores`, "attention scores", "matmul", { attention_kind: attentionKind }),
    operatorSpec(`${prefix}.softmax`, "attention probabilities", "softmax"),
    operatorSpec(`${prefix}.context`, "weighted value", "matmul"),
    operatorSpec(`${prefix}.o_proj`, "output projection", "linear"),
  ];
}

export function mlpOperatorSpecs(prefix) {
  return [
    operatorSpec(`${prefix}.gate_proj`, "gate projection", "linear"),
    operatorSpec(`${prefix}.up_proj`, "up projection", "linear"),
    operatorSpec(`${prefix}.swiglu`, "SwiGLU activation", "swiglu"),
    operatorSpec(`${prefix}.down_proj`, "down projection", "linear"),
  ];
}

export function moeOperatorSpecs(prefix) {
  return [
    operatorSpec(`${prefix}.router`, "router logits", "linear"),
    operatorSpec(`${prefix}.topk`, "top-k expert routing", "topk"),
    operatorSpec(`${prefix}.dispatch`, "expert dispatch", "moe_dispatch"),
    operatorSpec(`${prefix}.expert_mlp`, "expert MLP", "swiglu"),
    operatorSpec(`${prefix}.combine`, "expert combine", "moe_combine"),
  ];
}
