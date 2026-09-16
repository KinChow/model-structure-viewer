import assert from "node:assert/strict";
import test from "node:test";
import { resolveNumberCommit } from "./numberField.js";

test("空字符串在非 allowEmpty 时回退到 fallback", () => {
  assert.deepEqual(resolveNumberCommit("", { min: 1, fallback: 1 }), { value: 1 });
});

test("空字符串在 allowEmpty 时返回 undefined", () => {
  assert.deepEqual(resolveNumberCommit("", { min: 1, allowEmpty: true }), { value: undefined });
});

test("低于 min 的值被夹到 min", () => {
  assert.deepEqual(resolveNumberCommit("0", { min: 1, fallback: 1 }), { value: 1 });
});

test("合法值原样返回并按上界夹取", () => {
  assert.deepEqual(resolveNumberCommit("8", { min: 1, fallback: 1 }), { value: 8 });
  assert.deepEqual(resolveNumberCommit("2", { min: 0.1, max: 1, fallback: 0.7 }), { value: 1 });
});

test("非法输入回退到 fallback", () => {
  assert.deepEqual(resolveNumberCommit("abc", { min: 1, fallback: 1 }), { value: 1 });
});

test("小数在区间内保留", () => {
  assert.deepEqual(resolveNumberCommit("0.55", { min: 0.1, max: 1, fallback: 0.7 }), { value: 0.55 });
});
