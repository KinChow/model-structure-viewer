const FORMULAS = {
  linear: {
    title: "Linear",
    formula: "Y = XW^T + b",
    explanation: "线性投影，用于生成 q/k/v、MLP 中间状态或输出投影。",
    inputs: ["X", "W", "b"],
    outputs: ["Y"],
  },
  matmul: {
    title: "MatMul",
    formula: "Y = A B",
    explanation: "矩阵乘法，用于 attention score 或加权 value 聚合。",
    inputs: ["A", "B"],
    outputs: ["Y"],
  },
  softmax: {
    title: "Softmax",
    formula: "softmax(x_i) = exp(x_i) / sum_j exp(x_j)",
    explanation: "将 attention score 转成概率分布。",
    inputs: ["scores"],
    outputs: ["probabilities"],
  },
  rope: {
    title: "RoPE",
    formula: "q', k' = rotate(q, k, position)",
    explanation: "对 q/k 注入旋转位置编码。",
    inputs: ["q", "k", "position"],
    outputs: ["q'", "k'"],
  },
  rmsnorm: {
    title: "RMSNorm",
    formula: "y = x / sqrt(mean(x^2) + eps) * weight",
    explanation: "按均方根归一化，不减去均值。",
    inputs: ["x", "weight", "eps"],
    outputs: ["y"],
  },
  swiglu: {
    title: "SwiGLU",
    formula: "y = SiLU(xW_gate) * (xW_up)",
    explanation: "门控前馈激活，常用于 LLM 的 MLP。",
    inputs: ["x", "W_gate", "W_up"],
    outputs: ["y"],
  },
  topk: {
    title: "TopK Routing",
    formula: "experts = topk(router_logits, k)",
    explanation: "为 token 选择得分最高的专家。",
    inputs: ["router_logits", "k"],
    outputs: ["expert_ids", "expert_weights"],
  },
  moe_dispatch: {
    title: "MoE Dispatch",
    formula: "x_e = dispatch(x, expert_ids)",
    explanation: "按路由结果把 token 分发给专家。",
    inputs: ["x", "expert_ids"],
    outputs: ["expert_inputs"],
  },
  moe_combine: {
    title: "MoE Combine",
    formula: "y = sum_e weight_e * expert_e(x_e)",
    explanation: "按路由权重合并专家输出。",
    inputs: ["expert_outputs", "expert_weights"],
    outputs: ["y"],
  },
};

export function formulaForOperator(operatorId) {
  return FORMULAS[operatorId] || null;
}
