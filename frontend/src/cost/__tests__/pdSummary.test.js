import assert from "node:assert/strict";
import test from "node:test";
import { pdKvTransferBytes } from "../comm.js";

test("PD 摘要使用 decode 侧 TP 计算每 rank 传输量", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 160, config: { kvHeads: 4 }, pdPlan: {
    prefill_plan: { tp: 1 }, decode_plan: { tp: 8 },
  } });
  assert.equal(result.perDecodeRankBytes, 40);
});

test("PD 双侧计划支持 PP/EP/DP 字段", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 160, config: { kvHeads: 4, experts: 8 }, pdPlan: {
    prefill_plan: { tp: 2, pp: 2, ep: 2, dp: 1 }, decode_plan: { tp: 4, pp: 2, ep: 2, dp: 2 },
  } });
  assert.equal(result.ok, true);
  assert.equal(result.decodePlan.pp, 2);
  assert.equal(result.decodePlan.ep, 2);
  assert.equal(result.decodePlan.dp, 2);
});
