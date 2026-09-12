// checkpoint 按模块路径绑定：剥 HF 根包装 model. / language_model. 后相等匹配。
import assert from "node:assert/strict";
import test from "node:test";
import { buildSkeleton } from "../skeleton.js";
import { bindTruthToGraph, canonicalModulePath, skeletonTruthGraph } from "../graphTruth.js";

const T = (name, dtype = "BF16", shape) => ({ name, dtype, shape });

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

function truthGraphFrom(tensorNames) {
  const tensors = tensorNames.map((name) => T(name, "BF16", [4, 4]));
  return skeletonTruthGraph(buildSkeleton(tensors));
}

test("canonicalModulePath 只剥 HF 根包装，不改 layers/visual", () => {
  assert.equal(canonicalModulePath("model.layers.0.self_attn.q_proj"), "layers.0.self_attn.q_proj");
  assert.equal(canonicalModulePath("language_model.norm"), "norm");
  assert.equal(canonicalModulePath("model.visual.patch_embed"), "visual.patch_embed");
  assert.equal(canonicalModulePath("embed_tokens"), "embed_tokens");
});

test("路径绑定：layers 图 id 与 checkpoint layers 剥包装后相等", () => {
  const template = graphFromSpecs([
    { id: "embed_tokens", type: "embedding" },
    { id: "layers.0.self_attn.q_proj" },
    { id: "layers.0.mlp.gate_proj" },
    { id: "layers.1.self_attn.q_proj" },
    { id: "norm", type: "normalization" },
    { id: "lm_head", type: "output" },
  ]);
  const truth = truthGraphFrom([
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
  ]);
  const { graph, diagnostics } = bindTruthToGraph(template, truth);
  assert.deepEqual(diagnostics.graph_ambiguous_truth_matches, []);
  const bound = graph.nodes.filter((node) => node.value_source === "checkpoint");
  assert.equal(bound.length, 6);
  const q0 = bound.find((node) => node.canonical_id === "layers.0.self_attn.q_proj");
  assert.equal(q0.tensor_names[0], "model.layers.0.self_attn.q_proj.weight");
  const q1 = bound.find((node) => node.canonical_id === "layers.1.self_attn.q_proj");
  assert.equal(q1.tensor_names[0], "model.layers.1.self_attn.q_proj.weight");
});

test("路径绑定：shared_experts 与 routed experts 靠路径区分", () => {
  const template = graphFromSpecs([
    { id: "layers.3.mlp.router" },
    { id: "layers.3.mlp.shared_experts.gate_proj" },
    { id: "layers.3.mlp.shared_experts.down_proj" },
  ]);
  const truth = truthGraphFrom([
    "model.layers.3.mlp.router.weight",
    "model.layers.3.mlp.shared_experts.gate_proj.weight",
    "model.layers.3.mlp.shared_experts.down_proj.weight",
    "model.layers.3.mlp.experts.0.gate_proj.weight",
    "model.layers.3.mlp.experts.1.gate_proj.weight",
  ]);
  const { graph, diagnostics } = bindTruthToGraph(template, truth);
  assert.deepEqual(diagnostics.graph_ambiguous_truth_matches, []);
  const bound = graph.nodes.filter((node) => node.value_source === "checkpoint");
  assert.equal(bound.length, 3);
  assert.ok(diagnostics.graph_truth_gaps.some((id) => id.includes("experts")));
});

test("路径绑定：非候选的不同名不绑", () => {
  const template = graphFromSpecs([{ id: "layers.0.self_attn.q_proj" }]);
  const truth = truthGraphFrom(["model.blocks.0.self_attn.q_proj.weight"]);
  const { graph, diagnostics } = bindTruthToGraph(template, truth);
  assert.equal(graph.nodes.filter((node) => node.value_source === "checkpoint").length, 0);
  assert.equal(diagnostics.graph_truth_gaps.length, 1);
});
