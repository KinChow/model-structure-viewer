import assert from "node:assert/strict";
import test from "node:test";
import { actionsByFormulaGroup, costByFormulaGroup, costSummaryModel, diagnosticsModel } from "../ui.js";

test("costSummaryModel：五类时间命名与未知计费计数", () => {
  const model = costSummaryModel(
    { weightSource: "checkpoint", unknownComputePaths: ["a", "b"] },
    { bound: "memory", times: { matrix: 0.1, vector: 0, sfu: null, memory: 2, comm: null } },
    { english: false },
  );
  assert.equal(model.boundLabel, "访存");
  assert.equal(model.unknownComputeCount, 2);
  assert.equal(model.boundCategoryLabel, "访存受限");
  assert.equal(model.weightSourceLabel, "checkpoint 真值");
  const sfu = model.times.find((t) => t.unit === "sfu");
  assert.equal(sfu.known, false);
  assert.equal(sfu.label, "SFU");
});

test("costByFormulaGroup：按 FORMULAS.group 归并 compute_macs（含 repeat 乘子）并降序", () => {
  const cost = {
    computeComplete: true,
    nodes: [
      { compute_macs: 100, node: { attributes: { operator_id: "linear" } } }, // gemm
      { compute_macs: 300, node: { attributes: { operator_id: "sdpa_attention" } } }, // attention
      { compute_macs: 50, node: { attributes: { operator_id: "fused_moe_mlp" } } }, // moe
      { compute_macs: 0, node: { attributes: { operator_id: "rope" } } }, // 零 macs 跳过
      { compute_macs: null, node: { attributes: { operator_id: "linear" } } }, // 未知跳过
    ],
  };
  const groups = costByFormulaGroup(cost);
  assert.deepEqual(groups.map((entry) => entry.group), ["attention", "gemm", "moe"]);
  assert.equal(groups[0].macs, 300);
  assert.ok(Math.abs(groups[0].pct - 300 / 450) < 1e-9);
});

test("costByFormulaGroup：compute 未完整时返回空表（不猜构成）", () => {
  assert.deepEqual(costByFormulaGroup({ computeComplete: false, nodes: [{ compute_macs: 1, node: { attributes: { operator_id: "linear" } } }] }), []);
});

test("actionsByFormulaGroup：按 group 聚合完整动作向量并按 matrix 降序", () => {
  const cost = {
    computeComplete: true,
    nodes: [
      { actions: { matrix: 100, vector: 2, sfu: 0, bytes: { weights: 10, actIn: 1, actOut: 1, kvRead: 0, indexRead: 0 } }, node: { attributes: { operator_id: "linear" } } }, // gemm
      { actions: { matrix: 40, vector: 0, sfu: 0, bytes: { weights: 4, actIn: 0, actOut: 0, kvRead: 0, indexRead: 0 } }, node: { attributes: { operator_id: "linear" } } }, // gemm 同域累加
      { actions: { matrix: 300, vector: 5, sfu: 7, bytes: { weights: 0, actIn: 2, actOut: 2, kvRead: 3, indexRead: 0 } }, node: { attributes: { operator_id: "sdpa_attention" } } }, // attention
      { actions: null, node: { attributes: { operator_id: "linear" } } }, // 父/覆盖节点 actions=null 跳过（不重复计）
    ],
  };
  const groups = actionsByFormulaGroup(cost);
  assert.deepEqual(groups.map((entry) => entry.group), ["attention", "gemm"]);
  assert.equal(groups[1].actions.matrix, 140); // 两条 gemm 叶累加
  assert.equal(groups[1].actions.vector, 2);
  assert.equal(groups[1].actions.bytes.weights, 14);
  assert.equal(groups[0].actions.matrix, 300);
  assert.equal(groups[0].actions.bytes.kvRead, 3);
  assert.equal(groups[0].actions.commBytes, 0); // 首版通信不按 stage 归属
});

test("actionsByFormulaGroup：compute 未完整时返回空表", () => {
  assert.deepEqual(actionsByFormulaGroup({ computeComplete: false, nodes: [{ actions: { matrix: 1, bytes: {} }, node: { attributes: { operator_id: "linear" } } }] }), []);
});

test("costSummaryModel：英文文案与 unknown bound", () => {
  const model = costSummaryModel({}, null, { english: true });
  assert.equal(model.boundLabel, "unknown");
  assert.equal(model.boundCategoryLabel, "unknown");
  assert.equal(model.weightSourceLabel, null);
  assert.deepEqual(model.times, []);
});

test("costSummaryModel：结论条友好瓶颈归类（matrix/sfu→Compute-bound）", () => {
  const zh = costSummaryModel({}, { bound: "matrix", times: { matrix: 1 } }, { english: false });
  assert.equal(zh.boundLabel, "矩阵");
  assert.equal(zh.boundCategoryLabel, "算力受限");
  const en = costSummaryModel({}, { bound: "matrix", times: { matrix: 1 } }, { english: true });
  assert.equal(en.boundLabel, "matrix");
  assert.equal(en.boundCategoryLabel, "Compute-bound");
  const enSfu = costSummaryModel({}, { bound: "sfu", times: { sfu: 1 } }, { english: true });
  assert.equal(enSfu.boundCategoryLabel, "Compute-bound");
  const enComm = costSummaryModel({}, { bound: "comm", times: { comm: 1 } }, { english: true });
  assert.equal(enComm.boundCategoryLabel, "Communication-bound");
});

test("diagnosticsModel：skeleton-truth 触发未适配 banner，歧义计数", () => {
  const model = diagnosticsModel({
    strategy: "skeleton-truth",
    graph_truth_gaps: ["model.layers.0.mlp.experts"],
    graph_ambiguous_truth_matches: [{ template: "x", candidates: ["a", "b"] }],
    total_tensors: 10,
    graph_bound_tensors: 8,
  }, { english: false });
  assert.equal(model.banner.skeleton, true);
  assert.equal(model.gapCount, 1);
  assert.equal(model.ambiguousCount, 1);
});

test("diagnosticsModel：template+header-truth 视为已适配，无未适配 banner", async () => {
  const { diagnosticsModel } = await import("../ui.js");
  const model = diagnosticsModel({ strategy: "template+header-truth", parameter_total: 12345 });
  assert.equal(model.banner, null);
});

test("diagnosticsModel：template+truth 无 banner；mergeSemantics 旧键兼容", () => {
  const adapted = diagnosticsModel({ strategy: "template+truth", graph_truth_gaps: [], graph_ambiguous_truth_matches: [] });
  assert.equal(adapted.banner, null);
  const legacy = diagnosticsModel({ strategy: "template+truth", template_gaps: ["g1", "g2"] });
  assert.equal(legacy.gapCount, 2);
});

test("diagnosticsModel：unsupported 与 warnings 透传（M11-P0-3）", () => {
  const model = diagnosticsModel({
    strategy: "skeleton-truth",
    unsupported: [{ code: "unsupported-architecture", message: "Supported architectures: Qwen3ForCausalLM" }],
    warnings: [{ code: "missing-layer-count", message: "No text layer count was found in config" }],
  }, { english: false });
  assert.equal(model.unsupportedCount, 1);
  assert.match(model.unsupported[0].message, /Qwen3ForCausalLM/);
  assert.equal(model.warningCount, 1);
  assert.equal(model.warnings[0].code, "missing-layer-count");
  const empty = diagnosticsModel({ strategy: "no-truth" });
  assert.equal(empty.unsupportedCount, 0);
  assert.equal(empty.warningCount, 0);
});

test("missingLabelsModel：已知键翻译 + 动态费率键 + 未知键透传", async () => {
  const { missingLabelsModel } = await import("../ui.js");
  const zh = missingLabelsModel(["matrix", "peak_flops.bf16", "mystery_key"], { english: false });
  assert.equal(zh[0].label, "矩阵 MACs 数量");
  assert.match(zh[1].label, /peak_flops（bf16）规格/);
  assert.equal(zh[2].label, "mystery_key"); // 未知键原样透传，不伪造可读名
  assert.match(missingLabelsModel(["peak_flops.fp8"], { english: true })[0].label, /peak_flops \(fp8\) spec/);
});

test("macsSourcesModel：固定类目顺序 + 零计数省略", async () => {
  const { macsSourcesModel } = await import("../ui.js");
  const rows = macsSourcesModel({ formula: 30, aggregate: 5, unknown: 2 }, { english: false });
  assert.deepEqual(rows.map((r) => r.source), ["formula", "aggregate", "unknown"]);
  assert.equal(rows.every((r) => r.count > 0), true);
});

test("checkpointTruthModel：静默切换披露 + 错误透传", async () => {
  const { checkpointTruthModel } = await import("../ui.js");
  const fallback = checkpointTruthModel({
    checkpoint_truth: "available", config_endpoint: "https://hf.co/x", checkpoint_truth_endpoint: "https://modelscope/x",
  }, { english: false });
  assert.equal(fallback.show, true);
  assert.equal(fallback.tone, "warn");
  assert.match(fallback.headline, /静默切换/);
  const failed = checkpointTruthModel({ checkpoint_truth: "unavailable", checkpoint_truth_error: "boom" }, { english: false });
  assert.equal(failed.show, true);
  assert.equal(failed.tone, "error");
  assert.equal(failed.error, "boom");
  assert.equal(checkpointTruthModel({ checkpoint_truth: "available" }).show, false);
});

test("etaDisclosureModel：显式乐观上界披露", async () => {
  const { etaDisclosureModel, ETA_VECTOR_SFU_DEFAULT } = await import("../ui.js");
  assert.equal(ETA_VECTOR_SFU_DEFAULT, 1);
  assert.match(etaDisclosureModel({ english: false }).short, /η=1\.0/);
  assert.match(etaDisclosureModel({ english: true }).short, /optimistic/);
});
