// N2-2：离线 checkpoint 真值（skeleton-truth.json）的图侧契约。
// truth.skeleton（已折叠 SkeletonNode）与 truth.tensors 两种形态必须产出
// 等价的 truth graph —— 只有 strategy 标签不同（-file 后缀）。
import assert from "node:assert/strict";
import test from "node:test";
import { buildSkeleton } from "../skeleton.js";
import { enrichGraphWithTruth } from "../graphTruth.js";

const TENSORS = [
  { name: "model.layers.0.input_layernorm.weight", dtype: "BF16", shape: [512] },
  { name: "model.layers.1.input_layernorm.weight", dtype: "BF16", shape: [512] },
  { name: "model.layers.0.self_attn.q_proj.weight", dtype: "BF16", shape: [512, 512] },
  { name: "model.layers.1.self_attn.q_proj.weight", dtype: "BF16", shape: [512, 512] },
];

const OPTS = { hasTemplate: false, modelName: "fixture", canonicalArchitecture: "fixture", modelType: "fixture" };

test("truth.skeleton（离线文件形态）与 truth.tensors 产出等价 truth graph", () => {
  const skeleton = buildSkeleton(TENSORS);
  const viaTensors = enrichGraphWithTruth({}, { tensors: TENSORS, parameterTotal: 2 * (512 + 512 * 512) }, OPTS);
  const viaFile = enrichGraphWithTruth({}, { skeleton, tensor_count: TENSORS.length, parameterTotal: 2 * (512 + 512 * 512) }, OPTS);

  assert.equal(viaTensors.diagnostics.strategy, "skeleton-truth");
  assert.equal(viaFile.diagnostics.strategy, "skeleton-truth-file");
  assert.equal(viaFile.diagnostics.total_tensors, TENSORS.length);
  assert.equal(viaFile.diagnostics.parameter_total, 2 * (512 + 512 * 512));
  // 图节点集合逐 id 相等（同一张量集合折叠出的树必然一致）
  const ids = (g) => g.nodes.map((n) => n.id).sort().join("|");
  assert.equal(ids(viaFile.graph), ids(viaTensors.graph));
  // weight_dtypes 逐节点相等（量化容量的 per-tensor 依据就来自这里）
  const dtypes = (g) => JSON.stringify(g.nodes.map((n) => n.attributes?.weight_dtypes || null));
  assert.equal(dtypes(viaFile.graph), dtypes(viaTensors.graph));
});

test("hasTemplate 时 skeleton 形态同样走绑定链（strategy: template+truth-file）", () => {
  const skeleton = buildSkeleton(TENSORS);
  const template = {
    nodes: [{ id: "root", module_id: "model", type: "model", children: [] }],
    edges: [],
  };
  // 无绑定命中也不得崩——gaps 诊断可见（诚实降级）
  const result = enrichGraphWithTruth(template, { skeleton, tensor_count: TENSORS.length }, { ...OPTS, hasTemplate: true });
  assert.equal(result.diagnostics.strategy, "template+truth-file");
});
