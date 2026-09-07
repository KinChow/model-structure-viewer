// W3-B2 验收：role 连接键绑定 + §4.6 可逆校验 + ambiguous=0。
import assert from "node:assert/strict";
import test from "node:test";
import { buildSkeleton } from "../skeleton.js";
import { bindTruthToGraph, skeletonTruthGraph } from "../graphTruth.js";
import { suffixesForRole } from "../../archs/index.js";

const T = (name, dtype = "BF16", shape) => ({ name, dtype, shape });

// materializeStructureGraph 的极简替身：只保留绑定所需字段（role/canonical_id）。
function graphFromSpecs(nodes) {
  return {
    version: 2,
    schema_version: 2,
    root_id: "root",
    nodes: nodes.map((node, index) => ({
      id: node.id,
      canonical_id: node.id,
      module_id: node.id,
      parent_id: node.parentId ?? "root",
      order: index,
      name: node.id,
      type: node.type ?? "operator",
      role: node.role ?? null,
      repeat: null,
      attributes: {},
      source_fields: [],
      confidence: "high",
      params: null,
      weight_shapes: null,
      dtype: null,
      input_shape: null,
      output_shape: null,
      value_source: null,
      tensor_names: null,
    })),
    edges: [],
  };
}

function truthGraphFrom(tensorNames, modelType) {
  const tensors = tensorNames.map((name) => T(name, "BF16", [4, 4]));
  return skeletonTruthGraph(buildSkeleton(tensors));
}

test("role 绑定：标准 dense（投影+norm+embed/lm_head）全命中且 ambiguous=0", () => {
  const template = graphFromSpecs([
    { id: "embed_tokens", type: "embedding", role: "token_embd" },
    { id: "decoder.0.self_attn.q_proj", role: "attn_q" },
    { id: "decoder.0.self_attn.k_proj", role: "attn_k" },
    { id: "decoder.0.self_attn.v_proj", role: "attn_v" },
    { id: "decoder.0.self_attn.o_proj", role: "attn_out" },
    { id: "decoder.0.input_layernorm", type: "normalization", role: "attn_norm" },
    { id: "decoder.0.mlp.gate_proj", role: "ffn_gate" },
    { id: "decoder.0.mlp.down_proj", role: "ffn_down" },
    { id: "decoder.1.self_attn.q_proj", role: "attn_q" },
    { id: "norm", type: "normalization", role: "output_norm" },
    { id: "lm_head", type: "output", role: "output" },
  ]);
  const truth = truthGraphFrom([
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
  ]);
  const { graph, diagnostics } = bindTruthToGraph(template, truth, { modelType: "qwen3" });
  assert.deepEqual(diagnostics.graph_ambiguous_truth_matches, []);
  const bound = graph.nodes.filter((node) => node.value_source === "checkpoint");
  assert.equal(bound.length, 11, "11 个模板节点应全部绑定");
  // 层号必须参与连接：layer0 的 q 不得绑到 layer1 的真值
  const q0 = bound.find((node) => node.canonical_id === "decoder.0.self_attn.q_proj");
  assert.equal(q0.tensor_names[0], "model.layers.0.self_attn.q_proj.weight");
  const q1 = bound.find((node) => node.canonical_id === "decoder.1.self_attn.q_proj");
  assert.equal(q1.tensor_names[0], "model.layers.1.self_attn.q_proj.weight");
});

test("role 绑定：MoE shared expert 走 shexp 作用域，与 routed 专家不混淆", () => {
  const template = graphFromSpecs([
    { id: "decoder.3.moe.router", role: "ffn_gate_inp" },
    { id: "decoder.3.moe.shared_experts.gate_proj", role: "ffn_gate_shexp" },
    { id: "decoder.3.moe.shared_experts.down_proj", role: "ffn_down_shexp" },
  ]);
  const truth = truthGraphFrom([
    "model.layers.3.moe.router.weight",
    "model.layers.3.moe.shared_experts.gate_proj.weight",
    "model.layers.3.moe.shared_experts.down_proj.weight",
    "model.layers.3.moe.experts.0.gate_proj.weight",
    "model.layers.3.moe.experts.1.gate_proj.weight",
  ]);
  const { graph, diagnostics } = bindTruthToGraph(template, truth, { modelType: "deepseek_v3" });
  assert.deepEqual(diagnostics.graph_ambiguous_truth_matches, [], "逐专家模块不得进入 role 连接");
  const bound = graph.nodes.filter((node) => node.value_source === "checkpoint");
  assert.equal(bound.length, 3);
  // 逐专家张量既不绑定也不消失 → 记入 gaps
  assert.ok(diagnostics.graph_truth_gaps.some((id) => id.includes("experts")));
});

test("role 绑定：fused qkv 与融合模板节点按 role 命中", () => {
  const template = graphFromSpecs([
    { id: "decoder.0.self_attn.qkv_gate_proj", role: "attn_qkv" },
  ]);
  const truth = truthGraphFrom(["model.layers.0.self_attn.qkv_gate_proj.weight"]);
  const { graph, diagnostics } = bindTruthToGraph(template, truth, { modelType: "qwen3_5" });
  assert.deepEqual(diagnostics.graph_ambiguous_truth_matches, []);
  assert.equal(graph.nodes.filter((node) => node.value_source === "checkpoint").length, 1);
});

test("§4.6 可逆校验：每个绑定都满足 真值后缀 ∈ role 的逆像", () => {
  const template = graphFromSpecs([
    { id: "decoder.0.self_attn.q_proj", role: "attn_q" },
    { id: "decoder.0.self_attn.o_proj", role: "attn_out" },
    { id: "decoder.0.mlp.gate_proj", role: "ffn_gate" },
  ]);
  const truthNames = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
  ];
  const truth = truthGraphFrom(truthNames);
  const { graph, diagnostics } = bindTruthToGraph(template, truth, { modelType: "qwen3" });
  assert.deepEqual(diagnostics.graph_ambiguous_truth_matches, []);
  for (const node of graph.nodes.filter((n) => n.value_source === "checkpoint")) {
    const tensor = node.tensor_names[0];
    const suffix = tensor.split(".").at(-2);
    const legal = suffixesForRole(node.role);
    assert.ok(legal.includes(suffix), `${node.canonical_id}(${node.role}) 绑到了非法后缀 ${suffix}，合法：${legal}`);
  }
});
