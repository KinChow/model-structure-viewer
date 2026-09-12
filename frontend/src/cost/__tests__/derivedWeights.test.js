import assert from "node:assert/strict";
import test from "node:test";
import { derivedWeightParameters, derivedBufferBytes, derivedWeightBytes, derivedFp32Parameters } from "../derivedWeights.js";
import { materializeStructureGraph } from "../../structure/graph/materializeStructureGraph.js";
import { aggregateCost } from "../aggregate.js";

// P7（步骤 7）：夹具 tree root 经 materializeStructureGraph 转 Graph IR。
const toGraph = (root) => materializeStructureGraph(root);

test("dense decoder fallback 计算 embedding、attention、MLP、norm 和 tied head", () => {
  const config = { layers: 1, hiddenSize: 4, attentionHeads: 2, kvHeads: 1, headDim: 2, valueHeadDim: 2, intermediateSize: 8, vocabSize: 10, tieWordEmbeddings: true };
  // 40 = embedding(10×4) · 48 = attention 四个投影 H·(hq+hk+hv+ho) = 4·(4+2+2+4)
  // · 4 = 逐头 QK-norm 2·head_dim（2026-09-09 补：`RMSNorm(head_dim)` 跨头共享，
  //   出处见 derivedWeights.js 的 qkNorm 注释）· 8 = 层内两个 RMSNorm 2·H
  // · 96 = MLP 3·H·I · 4 = final norm。tied head 不额外计参数。
  assert.equal(derivedWeightParameters(config), 40 + 48 + 4 + 8 + 96 + 4);
});

test("fused vs 非 fused shared expert：参数量按形态而非个数（P3）", () => {
  // 取证（models/moonshotai/Kimi-K3）：k3-index.json 每个 MoE 层只有
  // shared_experts.{gate,up,down}_proj.weight 各一个；
  // modeling_kimi_linear.py:797-801 先把 intermediate_size 乘 num_shared_experts
  // 再实例化**单个** KimiMLP —— 融合形态 = 一个更宽的 MLP。
  // Transformers / vLLM / SGLang 的 DeepSeek 与 K3 都是「一个更宽的 MLP」
  // （modeling 先 moeI×n_shared 再实例化单个 MLP）。目录里 DeepSeek 系 n_shared=1，
  // fused vs 计数乘子在 n=1 时观察不到差别。本测试锁的是宽度语义：
  // fused 传模块宽且 count=1，非 fused 传单专家宽且 count=n。
  const base = {
    layers: 1, hiddenSize: 8, attentionHeads: 2, kvHeads: 1, headDim: 4, valueHeadDim: 4,
    intermediateSize: 8, vocabSize: 10, tieWordEmbeddings: true,
    experts: 4, expertsPerToken: 2, moeIntermediateSize: 6, layerSchedule: ["moe"],
    sharedExperts: 2,
  };
  // fused：normalize 已把 sharedExpertIntermediateSize 折成模块宽（6×2=12），
  // 期望侧按 1 组 × 3 矩阵 × H × 12。
  const fused = derivedWeightParameters({ ...base, sharedExpertIntermediateSize: 12, sharedExpertsAreFused: true });
  // 非 fused：单专家宽 6，期望侧按 2 组 × 3 矩阵 × H × 6 —— 总量与 fused 相等。
  const separate = derivedWeightParameters({ ...base, sharedExpertIntermediateSize: 6, sharedExpertsAreFused: false });
  assert.equal(fused, separate, "两形态的 shared expert 总参数量应相等（3·H·moeI·n）");
  // 若把 fused 的模块宽错按 n 份计（3·H·12·2），会多出 3·H·12 —— 该差额即回归信号。
  const wrong = derivedWeightParameters({ ...base, sharedExpertIntermediateSize: 12, sharedExpertsAreFused: false });
  assert.equal(wrong - fused, 3 * base.hiddenSize * 12, "fused 声明被当成 n 份时的差额（回归探针）");
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
  const result = aggregateCost({ graph: toGraph({ children: [] }), config: { hiddenSize: 4, vocabSize: 10, tieWordEmbeddings: true }, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 88);
  assert.equal(result.weightSource, "derived");
});

test("权重 what-if 只在显式指定时覆盖默认字节数", () => {
  const result = aggregateCost({ graph: toGraph({ children: [] }), config: { hiddenSize: 4, vocabSize: 10, tieWordEmbeddings: true }, weightBytesPerParameter: 1, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 44);
  assert.equal(result.weightSource, "what-if");
});

test("量化配置进入 derived 权重估计并明确标记来源", () => {
  const result = aggregateCost({
    graph: toGraph({ children: [] }),
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
