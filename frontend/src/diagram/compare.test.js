import assert from "node:assert/strict";
import test from "node:test";
import { boundFlips } from "./compare.js";

test("双卡对比输出 bound 翻转节点", () => {
  assert.deepEqual(boundFlips({ a: { bound: "memory" }, b: { bound: "compute" } }, { a: { bound: "compute" }, b: { bound: "compute" } }), [{ path: "a", primary: "memory", secondary: "compute" }]);
});

test("同卡不同并行计划也可复用 bound 翻转比较", () => {
  const flips = boundFlips({ a: { bound: "memory" } }, { a: { bound: "comm" } });
  assert.equal(flips[0].secondary, "comm");
});
