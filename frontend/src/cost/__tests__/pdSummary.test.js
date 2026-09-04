import assert from "node:assert/strict";
import test from "node:test";
import { pdKvTransferBytes } from "../comm.js";

test("PD 摘要使用 decode 侧 TP 计算每 rank 传输量", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 160, config: { kvHeads: 4 }, pdPlan: {
    prefill_plan: { tp: 1 }, decode_plan: { tp: 8 },
  } });
  assert.equal(result.perDecodeRankBytes, 40);
});
