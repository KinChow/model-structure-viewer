import assert from "node:assert/strict";
import test from "node:test";
import { costSummaryModel, diagnosticsModel } from "../ui.js";

test("costSummaryModel：五类时间命名与未知计费计数", () => {
  const model = costSummaryModel(
    { weightSource: "checkpoint", unknownComputePaths: ["a", "b"] },
    { bound: "memory", times: { matrix: 0.1, vector: 0, sfu: null, memory: 2, comm: null } },
    { english: false },
  );
  assert.equal(model.boundLabel, "访存");
  assert.equal(model.unknownComputeCount, 2);
  assert.equal(model.weightSourceLabel, "checkpoint 真值");
  const sfu = model.times.find((t) => t.unit === "sfu");
  assert.equal(sfu.known, false);
  assert.equal(sfu.label, "SFU");
});

test("costSummaryModel：英文文案与 unknown bound", () => {
  const model = costSummaryModel({}, null, { english: true });
  assert.equal(model.boundLabel, "unknown");
  assert.equal(model.weightSourceLabel, null);
  assert.deepEqual(model.times, []);
});

test("diagnosticsModel：skeleton-truth 触发未适配 banner，歧义计数", () => {
  const model = diagnosticsModel({
    strategy: "skeleton-truth",
    graph_truth_gaps: ["model.layers.0.moe.experts"],
    graph_ambiguous_truth_matches: [{ template: "x", candidates: ["a", "b"] }],
    total_tensors: 10,
    graph_bound_tensors: 8,
  }, { english: false });
  assert.equal(model.banner.skeleton, true);
  assert.equal(model.gapCount, 1);
  assert.equal(model.ambiguousCount, 1);
});

test("diagnosticsModel：template+truth 无 banner；mergeSemantics 旧键兼容", () => {
  const adapted = diagnosticsModel({ strategy: "template+truth", graph_truth_gaps: [], graph_ambiguous_truth_matches: [] });
  assert.equal(adapted.banner, null);
  const legacy = diagnosticsModel({ strategy: "template+truth", template_gaps: ["g1", "g2"] });
  assert.equal(legacy.gapCount, 2);
});
