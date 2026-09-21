// 框架预设 → 线性 recurrent(ssm) state dtype 默认（vLLM≠SGLang 分叉可切换）。
// 铁律：config 显式声明恒胜；neutral 不改任何东西（向后兼容）。
import assert from "node:assert/strict";
import test from "node:test";
import { applyFrameworkProfile, FRAMEWORK_PROFILES } from "../buildStructure.js";

test("vLLM 预设：config 未声明 ssm dtype → 默认 bf16", () => {
  const n = {};
  applyFrameworkProfile(n, "vllm");
  assert.equal(n.mambaSsmDtype, "bfloat16");
});

test("SGLang 预设：config 未声明 → 默认 fp32", () => {
  const n = {};
  applyFrameworkProfile(n, "sglang");
  assert.equal(n.mambaSsmDtype, "float32");
});

test("neutral 预设：不改（保持 config 或缺省）", () => {
  const n = {};
  applyFrameworkProfile(n, "neutral");
  assert.equal(n.mambaSsmDtype, undefined);
});

test("config 显式声明恒胜（预设不覆盖）", () => {
  const n = { mambaSsmDtype: "float32" };
  applyFrameworkProfile(n, "vllm");
  assert.equal(n.mambaSsmDtype, "float32");
});

test("FRAMEWORK_PROFILES 取值集合", () => {
  assert.deepEqual(FRAMEWORK_PROFILES, ["neutral", "sglang", "vllm"]);
});
