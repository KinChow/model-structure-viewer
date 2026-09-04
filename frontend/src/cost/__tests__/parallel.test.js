import assert from "node:assert/strict";
import test from "node:test";
import { kvBytesPerCard, projectNodePlan, projectPlan, stageForLayer, validatePdPlan, validatePlan, weightBytesPerCard } from "../parallel.js";

test("并行计划校验 TP×PP×DP 与 world_size", () => {
  assert.equal(validatePlan({ tp: 2, pp: 2, dp: 2, worldSize: 8 }).ok, true);
  assert.match(validatePlan({ tp: 2, pp: 2, dp: 2, worldSize: 4 }).errors[0], /TP×PP×DP/);
});

test("GQA 的 KV 分片因子为 min(TP, kv_heads)", () => {
  const result = kvBytesPerCard(160, { kvHeads: 4 }, { tp: 8 });
  assert.equal(result.shardFactor, 4);
  assert.equal(result.bytes, 40);
});

test("MLA 的 KV 在 TP 下全量复制", () => {
  const result = kvBytesPerCard(160, { kvHeads: 16, kvLoraRank: 512, qkRopeHeadDim: 64 }, { tp: 8 });
  assert.equal(result.shardFactor, 1);
  assert.equal(result.bytes, 160);
});

test("DP-attention 下每个 rank 持有完整 KV", () => {
  const result = kvBytesPerCard(160, { kvHeads: 8 }, { tp: 4, dp: 2, attnMode: "dp" });
  assert.equal(result.shardFactor, 1);
  assert.equal(result.bytes, 160);
});

test("权重按模块类别选择 TP/EP/复制投影", () => {
  assert.deepEqual(weightBytesPerCard(100, { id: "decoder.0.self_attn.q_proj" }, { tp: 4 }).bytes, 25);
  assert.deepEqual(weightBytesPerCard(100, { id: "decoder.0.mlp.experts.0.up_proj" }, { tp: 4, ep: 2 }).bytes, 50);
  assert.deepEqual(weightBytesPerCard(100, { id: "decoder.0.input_layernorm" }, { tp: 4 }).bytes, 100);
});

test("PP 层归属和逐 stage 投影返回结构", () => {
  assert.equal(stageForLayer(0, 8, 2), 0);
  assert.equal(stageForLayer(7, 8, 2), 1);
  const result = projectPlan({ weightBytes: 800, kvBytes: 160, config: { kvHeads: 8 }, plan: { tp: 2, pp: 2, dp: 1 } });
  assert.equal(result.ok, true);
  assert.equal(result.stages.length, 2);
  assert.equal(result.stages[0].weightBytes, 400);
  assert.equal(result.stages[0].kvBytes, 80);
});

test("PD 双 plan 分别校验并保留两侧配置", () => {
  const result = validatePdPlan({ prefill_plan: { tp: 2, pp: 1, dp: 1 }, decode_plan: { tp: 4, pp: 1, dp: 2 } }, {});
  assert.equal(result.ok, true);
  assert.equal(result.prefillPlan.tp, 2);
  assert.equal(result.decodePlan.tp, 4);
});

test("按节点路径分配 PP stage，首尾模块不平均摊薄", () => {
  const root = { id: "model", children: [
    { id: "embed_tokens", weight_shapes: { weight: [10, 2] }, dtype: "BF16", children: [] },
    { id: "layers.0", repeat: 2, children: [{ id: "layers.0.q_proj", weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] }] },
    { id: "lm_head", weight_shapes: { weight: [10, 2] }, dtype: "BF16", children: [] },
  ] };
  const result = projectNodePlan({ root, config: { layers: 2, kvHeads: 1 }, plan: { tp: 1, pp: 2, dp: 1 }, kvBytes: 0 });
  assert.equal(result.ok, true);
  assert.equal(result.stages[0].weightBytes, 56);
  assert.equal(result.stages[1].weightBytes, 40);
});

test("PP 汇总不重复计算父列表和范围子节点 repeat", () => {
  const root = { id: "decoder", repeat: 4, children: [{ id: "decoder.0", repeat: 4, children: [{ id: "decoder.0.mlp.down_proj", weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] }] }] };
  const result = projectNodePlan({ root, config: { layers: 4, kvHeads: 1 }, plan: { pp: 1 }, kvBytes: 0 });
  assert.equal(result.stages[0].weightBytes, 32);
});
