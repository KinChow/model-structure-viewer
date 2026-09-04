import assert from "node:assert/strict";
import test from "node:test";
import { activationTensorBytes, kvBytesPerToken, memoryBreakdown, tensorElements } from "../memory.js";
import { aggregateCost } from "../aggregate.js";

test("KV cache uses two tensors and KV heads", () => {
  assert.equal(kvBytesPerToken({ layers: 2, kvHeads: 4, headDim: 8 }, 2), 2 * 4 * 8 * 2 * 2);
});

test("MLA uses compressed latent plus rotary component", () => {
  assert.equal(kvBytesPerToken({ layers: 2, kvHeads: 16, headDim: 128, kvLoraRank: 512, qkRopeHeadDim: 64 }, 2), 2 * (512 + 64) * 2);
});

test("memory breakdown exposes five additive components", () => {
  const result = memoryBreakdown({ weightBytes: 10, config: { layers: 1, kvHeads: 1, headDim: 1 }, batch: 1, tokens: 2,
    activationPeak: 3, runtimeConst: 4, commBuffer: 5, kvBytes: 1 });
  assert.deepEqual(result, { weightBytes: 10, kvBytes: 4, kvBytesPerToken: 2, activationBytes: 3, runtimeBytes: 4, commBufferBytes: 5, totalBytes: 26 });
});

test("offline weight fallback multiplies folded layer repeats", () => {
  const root = { weight_shapes: {}, children: [{ repeat: 3, weight_shapes: {}, children: [
    { weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] },
  ] }] };
  const result = aggregateCost({ root, config: { layers: 3, kvHeads: 1, headDim: 1 }, sequence: 1, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 3 * 2 * 2 * 2);
});

test("empty parameterCount falls back to node weights", () => {
  const root = { weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] };
  const result = aggregateCost({ root, config: {}, parameterCount: {}, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 8);
});

test("动态数值 shape 分别解析普通张量和 attention 矩阵", () => {
  assert.equal(tensorElements([-1, -1, 4], { batch: 2, sequence: 3 }), 24);
  assert.equal(tensorElements([-1, -1, -1, -1], { batch: 2, sequence: 3, phase: "prefill", attentionHeads: 2 }), 36);
  assert.equal(tensorElements([-1, -1, -1, -1], { batch: 2, sequence: 3, phase: "decode", attentionHeads: 2 }), 12);
  assert.equal(activationTensorBytes([-1, -1, 4], { batch: 2, sequence: 3 }, 2), 48);
});
