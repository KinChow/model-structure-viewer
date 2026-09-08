// M11-P0-6 接缝测试（consumer-driven contract）：消费者 diagnosticsModel
// 必须读得到生产者 enrichGraphWithTruth 真实出口的每一个它声称消费的键。
// 此前两侧各有测试（roleBinding 测内部键、ui.test 手捏 fixture），接缝
// 键名不匹配（ambiguous_truth_matches vs graph_ambiguous_truth_matches）
// 导致歧义面板生产永不触发——教训已记 MAINTENANCE 变更纪律第 7 条。
import assert from "node:assert/strict";
import test from "node:test";
import { buildSkeleton } from "../skeleton.js";
import { enrichGraphWithTruth } from "../graphTruth.js";
import { diagnosticsModel } from "../../../cost/ui.js";

const T = (name, dtype = "BF16", shape = [4, 4]) => ({ name, dtype, shape });

function graphFromSpecs(nodes) {
  return {
    version: 2,
    schema_version: 2,
    root_id: "root",
    nodes: nodes.map((node, index) => ({
      id: node.id,
      canonical_id: node.id,
      module_id: node.id,
      parent_id: "root",
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

test("接缝：template+truth 生产出口的键被 diagnosticsModel 完整消费", () => {
  const graph = graphFromSpecs([
    { id: "decoder.0.self_attn.q_proj", role: "attn_q" },
    { id: "norm", type: "normalization", role: "output_norm" },
  ]);
  const truth = { tensors: [T("model.layers.0.self_attn.q_proj.weight"), T("model.norm.weight")], parameterTotal: 32 };
  const enriched = enrichGraphWithTruth(graph, truth, {
    hasTemplate: true,
    modelName: "SeamModel",
    canonicalArchitecture: "gqa-decoder",
    modelType: "seam_test",
  });
  assert.equal(enriched.diagnostics.strategy, "template+truth");
  // 生产者真实出口的键名（graphTruth.js 出口，无 graph_ 前缀）
  assert.ok("ambiguous_truth_matches" in enriched.diagnostics, "producer must emit ambiguous_truth_matches");
  assert.ok("template_gaps" in enriched.diagnostics, "producer must emit template_gaps");

  // 消费侧契约：生产键直接可读（歧义字段以生产者形状填入一条验证非空通路）
  const withAmbiguity = {
    ...enriched.diagnostics,
    ambiguous_truth_matches: [{ template: "decoder.0.self_attn.q_proj", candidates: ["a", "b"] }],
  };
  const model = diagnosticsModel(withAmbiguity, { english: false });
  assert.equal(model.ambiguousCount, 1);
  assert.equal(model.ambiguous[0].template, "decoder.0.self_attn.q_proj");
  assert.equal(model.gapCount, enriched.diagnostics.template_gaps.length);
});

test("接缝：skeleton-truth 策略走 banner 通路", () => {
  const graph = graphFromSpecs([{ id: "decoder.0.self_attn.q_proj", role: "attn_q" }]);
  const truth = { tensors: [T("model.layers.0.self_attn.q_proj.weight")], parameterTotal: 16 };
  const enriched = enrichGraphWithTruth(graph, truth, {
    hasTemplate: false,
    modelName: "SeamModel",
    canonicalArchitecture: "gqa-decoder",
    modelType: "seam_test",
  });
  const model = diagnosticsModel(enriched.diagnostics, { english: false });
  assert.equal(model.strategy, "skeleton-truth");
  assert.ok(model.banner, "skeleton-truth must surface the not-adapted banner");
});
