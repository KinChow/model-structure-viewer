import assert from "node:assert/strict";
import test from "node:test";
import { kvBytesPerToken, memoryBreakdown } from "../memory.js";

test("KV cache uses two tensors and KV heads", () => {
  assert.equal(kvBytesPerToken({ layers: 2, kvHeads: 4, headDim: 8 }, 2), 2 * 4 * 8 * 2 * 2);
});

test("MLA uses compressed latent plus rotary component", () => {
  assert.equal(kvBytesPerToken({ layers: 2, kvHeads: 16, headDim: 128, kvLoraRank: 512, qkRopeHeadDim: 64 }, 2), 2 * 2 * (512 + 64) * 2);
});

test("memory breakdown exposes five additive components", () => {
  const result = memoryBreakdown({ weightBytes: 10, config: { layers: 1, kvHeads: 1, headDim: 1 }, batch: 1, tokens: 2,
    activationPeak: 3, runtimeConst: 4, commBuffer: 5, kvBytes: 1 });
  assert.deepEqual(result, { weightBytes: 10, kvBytes: 4, kvBytesPerToken: 2, activationBytes: 3, runtimeBytes: 4, commBufferBytes: 5, totalBytes: 26 });
});
