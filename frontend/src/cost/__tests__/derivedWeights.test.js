import assert from "node:assert/strict";
import test from "node:test";
import { derivedWeightParameters } from "../derivedWeights.js";
import { aggregateCost } from "../aggregate.js";

test("dense decoder fallback 计算 embedding、attention、MLP、norm 和 tied head", () => {
  const config = { layers: 1, hiddenSize: 4, attentionHeads: 2, kvHeads: 1, headDim: 2, valueHeadDim: 2, intermediateSize: 8, vocabSize: 10, tieWordEmbeddings: true };
  assert.equal(derivedWeightParameters(config), 40 + 48 + 8 + 96 + 4);
});

test("无 checkpoint 和节点权重时使用 derived fallback", () => {
  const result = aggregateCost({ root: { children: [] }, config: { hiddenSize: 4, vocabSize: 10, tieWordEmbeddings: true }, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 88);
  assert.equal(result.weightSource, "derived");
});

test("权重 what-if 只在显式指定时覆盖默认字节数", () => {
  const result = aggregateCost({ root: { children: [] }, config: { hiddenSize: 4, vocabSize: 10, tieWordEmbeddings: true }, weightBytesPerParameter: 1, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 44);
  assert.equal(result.weightSource, "what-if");
});

test("量化配置进入 derived 权重估计并明确标记来源", () => {
  const result = aggregateCost({
    root: { children: [] },
    config: {
      hiddenSize: 4,
      vocabSize: 10,
      tieWordEmbeddings: true,
      quantizationBytesPerParameter: 0.5,
      quantizationMethod: "gptq",
    },
    activationPeak: 0,
    runtimeConst: 0,
  });
  assert.equal(result.memory.weightBytes, 22);
  assert.equal(result.weightSource, "derived-quantized");
  assert.equal(result.assumptions.weightBytesPerParameter, 0.5);
});
