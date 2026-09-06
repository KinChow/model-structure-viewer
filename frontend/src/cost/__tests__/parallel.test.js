import assert from "node:assert/strict";
import test from "node:test";
import { expertWeightRange, kvBytesPerCard, maxContextForStages, nodeCostPerCard, projectNodePlan, projectPdFit, projectPlan, stageForLayer, stateBytesPerCard, validatePdPlan, validatePlan, weightBytesPerCard } from "../parallel.js";

test("并行计划校验 TP×PP×DP 与 world_size", () => {
  assert.equal(validatePlan({ tp: 2, pp: 2, dp: 2, worldSize: 8 }).ok, true);
  assert.match(validatePlan({ tp: 2, pp: 2, dp: 2, worldSize: 4 }).errors[0], /TP×PP×DP/);
});

test("F5 GQA 的 KV 分片因子为 min(TP, kv_heads)", () => {
  const result = kvBytesPerCard(160, { kvHeads: 4 }, { tp: 8 });
  assert.equal(result.shardFactor, 4);
  assert.equal(result.bytes, 40);
});

test("F6 MLA 的 KV 在 TP 下全量复制", () => {
  const result = kvBytesPerCard(160, { kvHeads: 16, kvLoraRank: 512, qkRopeHeadDim: 64 }, { tp: 8 });
  assert.equal(result.shardFactor, 1);
  assert.equal(result.bytes, 160);
});

test("F7 DP-attention 下每个 rank 持有完整 KV", () => {
  const result = kvBytesPerCard(160, { kvHeads: 8 }, { tp: 4, dp: 2, attnMode: "dp" });
  assert.equal(result.shardFactor, 1);
  assert.equal(result.bytes, 160);
});

test("KDA request state follows attention TP and is replicated under DP-attention", () => {
  assert.equal(stateBytesPerCard(160, {}, { tp: 4 }).bytes, 40);
  assert.equal(stateBytesPerCard(160, {}, { tp: 4, attnMode: "dp" }).bytes, 160);
});

test("权重按模块类别选择 TP/EP/复制投影", () => {
  assert.deepEqual(weightBytesPerCard(100, { id: "decoder.0.self_attn.q_proj" }, { tp: 4 }).bytes, 25);
  assert.deepEqual(weightBytesPerCard(100, { id: "decoder.0.mlp.experts.0.up_proj" }, { tp: 4, ep: 2 }).bytes, 50);
  assert.deepEqual(weightBytesPerCard(100, { id: "decoder.0.moe.expert_mlp" }, { tp: 4, ep: 2 }).bytes, 50);
  assert.deepEqual(weightBytesPerCard(100, { id: "decoder.0.input_layernorm" }, { tp: 4 }).bytes, 100);
});

test("节点 roofline 成本按 TP、EP 或复制规则投影到单卡", () => {
  const cost = { macs: 80, weightBytes: 40, actInBytes: 24, actOutBytes: 16 };
  assert.deepEqual(nodeCostPerCard(cost, { id: "decoder.0.self_attn.q_proj" }, { tp: 4 }), {
    macs: 20, weightBytes: 10, actInBytes: 6, actOutBytes: 4,
    projection: { axis: "tp", divisor: 4 },
  });
  assert.equal(nodeCostPerCard(cost, { id: "decoder.0.mlp.experts.0.up_proj" }, { tp: 4, ep: 2 }).macs, 40);
  assert.equal(nodeCostPerCard(cost, { id: "decoder.0.input_layernorm" }, { tp: 4 }).macs, 80);
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

test("F14 按节点路径分配 PP stage，首尾模块不平均摊薄", () => {
  const root = { id: "model", children: [
    { id: "embed_tokens", weight_shapes: { weight: [10, 2] }, dtype: "BF16", children: [] },
    { id: "layers.0", repeat: 2, children: [{ id: "layers.0.q_proj", weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] }] },
    { id: "lm_head", weight_shapes: { weight: [5, 2] }, dtype: "BF16", children: [] },
  ] };
  const result = projectNodePlan({ root, config: { layers: 2, kvHeads: 1 }, plan: { tp: 1, pp: 2, dp: 1 }, kvBytes: 0 });
  assert.equal(result.ok, true);
  assert.equal(result.stages[0].weightBytes, 48);
  assert.equal(result.stages[1].weightBytes, 28);
});

test("PP 按 stage 层数分配 KV 而不是每个 stage 复制全量", () => {
  const result = projectNodePlan({ root: { id: "model", children: [] }, config: { layers: 5, kvHeads: 1 }, plan: { pp: 2 }, kvBytes: 100 });
  assert.deepEqual(result.stages.map((stage) => stage.kvBytes), [40, 60]);
});

test("PP 按实际 linear-attention 层分配 KDA state", () => {
  const config = { layers: 2, attentionSchedule: ["linear", "gqa"], attentionHeads: 2, headDim: 4, linearKeyHeads: 2, linearValueHeads: 2, linearKeyDim: 4, linearValueDim: 4, linearConvKernelSize: 3 };
  const result = projectNodePlan({ root: { id: "model", children: [] }, config, plan: { tp: 1, pp: 2 }, stateBytes: 128 });
  assert.deepEqual(result.stages.map((stage) => stage.stateBytes), [128, 0]);
});

test("PP 汇总不重复计算父列表和范围子节点 repeat", () => {
  const root = { id: "decoder", repeat: 4, children: [{ id: "decoder.0", repeat: 4, children: [{ id: "decoder.0.mlp.down_proj", weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] }] }] };
  const result = projectNodePlan({ root, config: { layers: 4, kvHeads: 1 }, plan: { pp: 1 }, kvBytes: 0 });
  assert.equal(result.stages[0].weightBytes, 32);
});

test("EP 返回专家权重平均值和最坏值区间", () => {
  const result = expertWeightRange(100, 8, 3);
  assert.equal(result.averageBytes, 100 / 3);
  assert.equal(result.worstBytes, 100 / 8 * 3);
  assert.equal(result.expertsPerRank, 3);
});

test("stage 权重返回 EP 平均/最坏两种投影", () => {
  const root = { id: "model.layers.0", children: [{ id: "model.layers.0.mlp.experts.0", repeat: 8, children: [{ id: "model.layers.0.mlp.experts.0.up_proj", weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] }] }] };
  const result = projectNodePlan({ root, config: { layers: 1, experts: 8, kvHeads: 1 }, plan: { ep: 3 }, kvBytes: 0 });
  assert.equal(result.stages[0].weightAverageBytes, 64 / 3);
  assert.equal(result.stages[0].weightWorstBytes, 24);
});

test("PD fit 分别按两侧芯片容量判定", () => {
  const result = projectPdFit({ weightBytes: 100, kvBytes: 20, config: {}, pdPlan: { prefill_plan: { tp: 1 }, decode_plan: { tp: 2 } }, prefillChip: { id: "p", memory_bytes: 200 }, decodeChip: { id: "d", memory_bytes: 50 } });
  assert.equal(result.ok, true);
  assert.equal(result.prefill.fit, true);
  assert.equal(result.decode.fit, false);
});

test("PD fit accepts independent prefill and decode KV footprints", () => {
  const result = projectPdFit({ weightBytes: 100, prefillKvBytes: 20, decodeKvBytes: 80, config: {}, pdPlan: { prefill_plan: { tp: 1 }, decode_plan: { tp: 1 } }, prefillChip: { memory_bytes: 130 }, decodeChip: { memory_bytes: 130 } });
  assert.equal(result.prefill.stages[0].kvBytes, 20);
  assert.equal(result.decode.stages[0].kvBytes, 80);
  assert.equal(result.prefill.fit, true);
  assert.equal(result.decode.fit, false);
});

test("计划最大上下文由最紧张 stage 决定", () => {
  const value = maxContextForStages([{ weightBytes: 40, kvBytes: 20 }, { weightBytes: 60, kvBytes: 10 }], { capacityBytes: 100, activationBytes: 10, runtimeBytes: 10, sequence: 10 });
  assert.equal(value, 20);
});

test("权重 what-if 比例同步应用到节点级 stage 投影", () => {
  const root = { id: "model", weight_shapes: { weight: [10] }, dtype: "BF16", children: [] };
  const result = projectPlan({ root, weightBytes: 5, config: {}, plan: { tp: 1 }, kvBytes: 0 });
  assert.equal(result.stages[0].weightBytes, 5);
});
