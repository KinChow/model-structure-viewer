import assert from "node:assert/strict";
import test from "node:test";
import { memoryBreakdown } from "../memory.js";
import { classifyRoofline } from "../roofline.js";

test("KV bytes 和常数项使用用户可调假设", () => {
  const result = memoryBreakdown({ weightBytes: 0, config: { layers: 1, kvHeads: 1, headDim: 1 }, tokens: 2, kvBytes: 1, activationPeak: 3, runtimeConst: 4 });
  assert.equal(result.kvBytes, 4);
  assert.equal(result.totalBytes, 11);
});

test("roofline 使用用户提供的效率因子", () => {
  const result = classifyRoofline({ macs: 10, weightBytes: 10 }, { peak_flops: { bf16: 100 }, memory_bandwidth: 10 }, { efficiency: { flops: 0.5, hbm: 0.5 } });
  assert.equal(result.times.compute, 0.4);
  assert.equal(result.times.memory, 2);
});
