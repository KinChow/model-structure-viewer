import assert from "node:assert/strict";
import test from "node:test";
import { boundFlips, COMPARISON_MODE, resolveComparisonScenario } from "./compare.js";

test("芯片对比输出 bound 翻转节点", () => {
  assert.deepEqual(boundFlips({ a: { bound: "memory" }, b: { bound: "compute" } }, { a: { bound: "compute" }, b: { bound: "compute" } }), [{ path: "a", primary: "memory", secondary: "compute" }]);
});

test("同卡不同并行计划也可复用 bound 翻转比较", () => {
  const flips = boundFlips({ a: { bound: "memory" } }, { a: { bound: "comm" } });
  assert.equal(flips[0].secondary, "comm");
});

test("芯片对比只替换芯片并锁定基准计划", () => {
  const primary = { chip: { id: "a" }, plan: { tp: 2, ep: 1 } };
  const candidate = { chip: { id: "b" }, plan: { tp: 8, ep: 4 } };
  const result = resolveComparisonScenario(COMPARISON_MODE.CHIP, primary, candidate);
  assert.equal(result.chip.id, "b");
  assert.deepEqual(result.plan, { tp: 2, ep: 1 });
});

test("方案对比只替换计划并锁定基准芯片", () => {
  const primary = { chip: { id: "a" }, plan: { tp: 2, ep: 1 } };
  const candidate = { chip: { id: "b" }, plan: { tp: 8, ep: 4 } };
  const result = resolveComparisonScenario(COMPARISON_MODE.PLAN, primary, candidate);
  assert.equal(result.chip.id, "a");
  assert.deepEqual(result.plan, { tp: 8, ep: 4 });
  assert.equal(resolveComparisonScenario(COMPARISON_MODE.OFF, primary, candidate), null);
});
