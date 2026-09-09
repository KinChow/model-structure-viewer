import assert from "node:assert/strict";
import test from "node:test";
import { derivedWeightParameters, derivedBufferBytes, derivedWeightBytes, derivedFp32Parameters } from "../derivedWeights.js";
import { aggregateCost } from "../aggregate.js";

test("dense decoder fallback 计算 embedding、attention、MLP、norm 和 tied head", () => {
  const config = { layers: 1, hiddenSize: 4, attentionHeads: 2, kvHeads: 1, headDim: 2, valueHeadDim: 2, intermediateSize: 8, vocabSize: 10, tieWordEmbeddings: true };
  // 40 = embedding(10×4) · 48 = attention 四个投影 H·(hq+hk+hv+ho) = 4·(4+2+2+4)
  // · 4 = 逐头 QK-norm 2·head_dim（2026-09-09 补：`RMSNorm(head_dim)` 跨头共享，
  //   出处见 derivedWeights.js 的 qkNorm 注释）· 8 = 层内两个 RMSNorm 2·H
  // · 96 = MLP 3·H·I · 4 = final norm。tied head 不额外计参数。
  assert.equal(derivedWeightParameters(config), 40 + 48 + 4 + 8 + 96 + 4);
});

test("tid2eid buffer 与 fp32 参数的字节宽（paramDtypes 登记表）", () => {
  // tid2eid：buffer 不是参数（Megatron-Bridge 明文）——容量 = hash 层数 × vocab × k × 4B int32，
  // 不进 derivedWeightParameters / derivedWeightBytes。
  const hash = { numHashLayers: 3, vocabSize: 100, expertsPerToken: 2, layers: 4, hiddenSize: 4, attentionHeads: 1, kvHeads: 1, headDim: 2, valueHeadDim: 2, intermediateSize: 4, experts: 8 };
  assert.equal(derivedBufferBytes(hash), 3 * 100 * 2 * 4);
  assert.equal(derivedBufferBytes({}), 0);
  // fp32 参数（dt_bias/A_log、mHC base/scale）：容量字节按 4B，权重字节恒等式两侧同表。
  const gdn = { layers: 1, attentionSchedule: ["linear"], hiddenSize: 4, attentionHeads: 2, kvHeads: 1, headDim: 2, valueHeadDim: 2, intermediateSize: 8, vocabSize: 10, tieWordEmbeddings: true, linearKeyHeads: 2, linearValueHeads: 2, linearKeyDim: 2, linearValueDim: 2 };
  // 1 个 linear 层：dt_bias + A_log = 2·valueHeads = 4 个 fp32 参数。
  assert.equal(derivedFp32Parameters(gdn), 4);
  // 容量字节 = 非 fp32 参数 × 2 + fp32 参数 × 4。
  assert.equal(derivedWeightBytes(gdn, 2), (derivedWeightParameters(gdn) - 4) * 2 + 4 * 4);
  // buffer 不进参数量：哈希层没有 router GEMM，所以 numHashLayers=3 的参数量
  // 恰好比 0 少 3·hidden·experts（那三层本来该有的 router），buffer 字节另计。
  assert.equal(
    derivedWeightParameters({ ...hash, numHashLayers: 0 }) - derivedWeightParameters(hash),
    3 * hash.hiddenSize * hash.experts,
  );
  assert.equal(derivedBufferBytes(hash), 3 * hash.vocabSize * hash.expertsPerToken * 4);
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
