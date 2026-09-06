import assert from "node:assert/strict";
import test from "node:test";
import { activationTensorBytes, kvBytesPerToken, linearStateBytesPerSequence, memoryBreakdown, tensorElements } from "../memory.js";
import { aggregateCost } from "../aggregate.js";

test("F3 KV cache 使用 K/V 两份张量和 KV heads", () => {
  assert.equal(kvBytesPerToken({ layers: 2, kvHeads: 4, headDim: 8 }, 2), 2 * 4 * 8 * 2 * 2);
});

test("F4 MLA KV 使用压缩 latent 与 rotary 分量", () => {
  assert.equal(kvBytesPerToken({ layers: 2, kvHeads: 16, headDim: 128, kvLoraRank: 512, qkRopeHeadDim: 64 }, 2), 2 * (512 + 64) * 2);
});

test("memory breakdown exposes token KV and request state separately", () => {
  const result = memoryBreakdown({ weightBytes: 10, config: { layers: 1, kvHeads: 1, headDim: 1 }, batch: 1, tokens: 2,
    activationPeak: 3, runtimeConst: 4, commBuffer: 5, kvBytes: 1 });
  assert.deepEqual(result, { weightBytes: 10, kvBytes: 4, kvBytesPerToken: 2, stateBytes: 0, stateBytesPerSequence: 0, activationBytes: 3, runtimeBytes: 4, commBufferBytes: 5, totalBytes: 26 });
});

test("KDA recurrent and convolution state is request-scoped, not token KV", () => {
  const config = { layers: 2, attentionSchedule: ["linear", "gqa"], kvHeads: 2, headDim: 4, linearKeyHeads: 2, linearValueHeads: 2, linearKeyDim: 4, linearValueDim: 4, linearConvKernelSize: 3 };
  assert.equal(linearStateBytesPerSequence(config, 2), 160);
  assert.equal(kvBytesPerToken(config, 2), 2 * 2 * 4 * 2);
  const result = memoryBreakdown({ weightBytes: 0, config, batch: 2, tokens: 100, kvBytes: 2, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.stateBytes, 320);
  assert.equal(result.kvBytes, 6400);
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

test("未知视觉输入尺寸不被当作文本 sequence", () => {
  assert.equal(tensorElements([-1, -1, -1, -1, -1], { batch: 1, sequence: 2048 }), 0);
});
