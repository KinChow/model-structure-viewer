import assert from "node:assert/strict";
import test from "node:test";

import { tensorDims } from "../../structure/model_executor/dims.js";
import { tensorShapes } from "../../structure/model_executor/shapes.js";

const QWEN = {
  hiddenSize: 1024,
  attentionHeads: 8,
  kvHeads: 4,
  headDim: 256,
  intermediateSize: 4096,
  experts: 64,
  expertsPerToken: 4,
  vocabSize: 151936,
};

test("tensorDims 返回数值形状，自由维用 -1 占位", () => {
  const dims = tensorDims(QWEN);
  assert.deepEqual(dims.hidden, [-1, -1, 1024]);
  assert.deepEqual(dims.attentionQuery, [-1, -1, 8, 256]);
  assert.deepEqual(dims.attentionKey, [-1, -1, 4, 256]);
  assert.deepEqual(dims.attentionValue, [-1, -1, 4, 256]);
  assert.deepEqual(dims.intermediate, [-1, -1, 4096]);
  assert.deepEqual(dims.routerLogits, [-1, -1, 64]);
  assert.deepEqual(dims.topExperts, [-1, -1, 4]);
  assert.deepEqual(dims.logits, [-1, -1, 151936]);
  assert.deepEqual(dims.attentionScores, [-1, -1, -1, -1]);
  assert.deepEqual(dims.tokenIds, [-1, -1]);
});

test("kvHeads 缺省时回退 attentionHeads；valueHeadDim 缺省时回退 headDim", () => {
  const dims = tensorDims({ hiddenSize: 512, attentionHeads: 16, headDim: 64 });
  assert.deepEqual(dims.attentionKey, [-1, -1, 16, 64]);
  assert.deepEqual(dims.attentionValue, [-1, -1, 16, 64]);
});

test("缺字段（无 experts）→ 对应位为 null/undefined，不产生伪数值", () => {
  const dims = tensorDims({ hiddenSize: 512, attentionHeads: 8, headDim: 64 });
  assert.equal(dims.routerLogits[2], undefined);
  assert.equal(dims.expertsPerToken?.[2], undefined);
});

test("tensorShapes 展示串与旧版逐字一致", () => {
  const shapes = tensorShapes(QWEN);
  assert.equal(shapes.hidden, "[batch, sequence, hidden size=1024]");
  assert.equal(shapes.attentionQuery, "[batch, sequence, attention heads=8, head dimension=256]");
  assert.equal(shapes.attentionKey, "[batch, sequence, key value heads=4, head dimension=256]");
  assert.equal(shapes.attentionValue, "[batch, sequence, key value heads=4, value head dimension=256]");
  assert.equal(shapes.attentionScores, "[batch, attention heads, query sequence, key sequence]");
  assert.equal(shapes.intermediate, "[batch, sequence, intermediate size=4096]");
  assert.equal(shapes.routerLogits, "[batch, sequence, experts=64]");
  assert.equal(shapes.logits, "[batch, sequence, vocab size=151936]");
  assert.equal(shapes.expertInput, "[tokens_per_expert, hidden size=1024]");
  assert.equal(shapes.tokenIds, "[batch, sequence]");
});
