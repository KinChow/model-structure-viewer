import assert from "node:assert/strict";
import test from "node:test";
import { expertAllToAllBytes, nodeCommunicationBytes, pdKvTransferBytes, pipelineP2PBytes, ringAllReduceBytes } from "../comm.js";

test("TP ring all-reduce 每层两次时包含 2×(TP-1)/TP 系数", () => {
  assert.equal(ringAllReduceBytes({ batch: 2, tokens: 3, hidden: 4, bytesPerElement: 2, tp: 4 }), 144);
  assert.equal(ringAllReduceBytes({ hidden: 4, tp: 1 }), 0);
});

test("EP all-to-all 使用 expertsPerToken 而不是专家总数", () => {
  assert.equal(expertAllToAllBytes({ batch: 2, tokens: 3, hidden: 4, expertsPerToken: 2, bytesPerElement: 2 }), 192);
});

test("PP P2P 按相邻 stage 边界计算", () => {
  assert.equal(pipelineP2PBytes({ batch: 1, tokens: 2, hidden: 4, bytesPerElement: 2, pp: 3 }), 32);
});

test("节点路径可识别 TP 与 EP 通信模块", () => {
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.self_attn.o_proj" }, { hiddenSize: 4 }, { tp: 2 }, { batch: 1, tokens: 1, bytesPerElement: 2 }), 8);
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.mlp.experts.0.dispatch" }, { hiddenSize: 4, expertsPerToken: 2 }, { tp: 2 }, { batch: 1, tokens: 1, bytesPerElement: 2 }), 32);
});

test("PD KV 传输量按 decode 侧 GQA 布局计算", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 160, config: { kvHeads: 4 }, pdPlan: {
    prefill_plan: { tp: 1 }, decode_plan: { tp: 8, dp: 1 },
  } });
  assert.equal(result.ok, true);
  assert.equal(result.shardFactor, 4);
  assert.equal(result.perDecodeRankBytes, 40);
});

test("PD KV 传输量按 decode 侧 MLA 布局复制", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 160, config: { kvLoraRank: 512, qkRopeHeadDim: 64 }, pdPlan: {
    prefill_plan: { tp: 1 }, decode_plan: { tp: 8, dp: 1 },
  } });
  assert.equal(result.ok, true);
  assert.equal(result.shardFactor, 1);
  assert.equal(result.perDecodeRankBytes, 160);
  assert.equal(result.aggregateBytes, 1280);
});
