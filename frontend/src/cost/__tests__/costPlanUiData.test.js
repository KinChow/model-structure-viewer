import assert from "node:assert/strict";
import test from "node:test";
import { projectPlan } from "../parallel.js";

test("给定计划能得到逐 PP stage 的 fit 输入", () => {
  const result = projectPlan({ weightBytes: 800, kvBytes: 160, config: { kvHeads: 8 }, plan: { tp: 2, pp: 2, dp: 1 } });
  assert.equal(result.ok, true);
  assert.deepEqual(result.stages.map((stage) => [stage.stage, stage.weightBytes, stage.kvBytes]), [[0, 400, 80], [1, 400, 80]]);
});
