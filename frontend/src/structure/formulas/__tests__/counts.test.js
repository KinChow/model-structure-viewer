// golden per-op 用例：手算期望值（照 PyTorch test_flop_counter 风格）。
// 期望值独立于实现手算而来；发现不一致时先查数学，不得改期望值迁就实现。
import assert from "node:assert/strict";
import test from "node:test";
import { formulaForOperator } from "../index.js";
import {
  linearCounts, attentionCounts, rmsnormCounts, gateCounts, swigluCounts,
  ropeCounts, causalConvCounts, linearAttentionStateCounts, topkCounts,
  moeDispatchCounts, moeCombineCounts, addCounts, hashRouteCounts,
  rearrangeCounts, softmaxCounts,
} from "../counts.js";

const B = 2; // bf16 每元素 2 字节

test("F1 linear：T·out·in MACs，权重整表读一遍", () => {
  const c = linearCounts({ logicalShape: [8, 4], tokens: 3, bytesPerElement: B, bias: true });
  assert.equal(c.matrix, 3 * 8 * 4);
  assert.equal(c.vector, 3 * 8); // bias 加法
  assert.equal(c.sfu, 0);
  assert.equal(c.bytes.weights, 8 * 4 * B);
  assert.equal(c.bytes.actIn, 3 * 4 * B);
  assert.equal(c.bytes.actOut, 3 * 8 * B);
});

test("F2 选择集注意力：两组 bmm + softmax（A2 单遍）", () => {
  const c = attentionCounts({ heads: 2, queryTokens: 4, keyTokens: 8, headDim: 16, valueDim: 16, bytesPerElement: B });
  const scores = 2 * 4 * 8;
  assert.equal(c.matrix, scores * (16 + 16)); // scores bmm + context bmm
  assert.equal(c.vector, 3 * scores);
  assert.equal(c.sfu, 2 * scores);
  const q = 2 * 4 * 16, k = 2 * 8 * 16, v = 2 * 8 * 16;
  assert.equal(c.bytes.actIn, (q + k + v + 2 * scores) * B); // Q/K/V 各一遍 + scores 写读
  assert.equal(c.bytes.actOut, (2 * scores + 2 * 4 * 16) * B);
});

test("F3 rmsnorm：3TH 向量 + T 次 rsqrt；gemma 多 TH；gated 多一路门", () => {
  const base = rmsnormCounts({ tokens: 5, hidden: 10, bytesPerElement: B });
  assert.equal(base.matrix, 0);
  assert.equal(base.vector, 3 * 5 * 10);
  assert.equal(base.sfu, 5);
  assert.equal(base.bytes.weights, 10 * B);
  const gemma = rmsnormCounts({ tokens: 5, hidden: 10, bytesPerElement: B, weightOne: true });
  assert.equal(gemma.vector, 4 * 5 * 10);
  const gated = rmsnormCounts({ tokens: 5, hidden: 10, bytesPerElement: B, gated: true });
  assert.equal(gated.vector, 4 * 5 * 10);
  assert.equal(gated.sfu, 5 + 5 * 10);
});

test("F4 门控乘：TW 次 sigmoid + TW 次乘；带输入投影时计权重", () => {
  const bare = gateCounts({ tokens: 4, width: 8, bytesPerElement: B });
  assert.equal(bare.matrix, 0);
  assert.equal(bare.sfu, 4 * 8);
  assert.equal(bare.bytes.weights, 0);
  const projected = gateCounts({ tokens: 4, width: 8, bytesPerElement: B, gateProjection: true, gateProjectionInput: 6 });
  assert.equal(projected.bytes.weights, 6 * 8 * B);
});

test("F5 SwiGLU：2TI 乘 + TI 次 sigmoid；输入是 gate/up 两路", () => {
  const c = swigluCounts({ tokens: 3, intermediate: 8, bytesPerElement: B });
  assert.equal(c.matrix, 0);
  assert.equal(c.vector, 2 * 3 * 8);
  assert.equal(c.sfu, 3 * 8);
  assert.equal(c.bytes.actIn, 2 * 3 * 8 * B);
});

test("F6 rope：查表假设下 sfu=0，每元素 3 flop，读+写 2TD", () => {
  const c = ropeCounts({ tokens: 4, ropeDims: 32, bytesPerElement: B });
  assert.equal(c.matrix, 0);
  assert.equal(c.vector, 3 * 4 * 32);
  assert.equal(c.sfu, 0);
  assert.equal(c.bytes.actIn, 2 * 4 * 32 * B);
});

test("F7a 因果卷积：T·C·w MACs + SiLU", () => {
  const c = causalConvCounts({ tokens: 4, channels: 6, kernel: 3, bytesPerElement: B });
  assert.equal(c.matrix, 4 * 6 * 3);
  assert.equal(c.vector, 4 * 6);
  assert.equal(c.sfu, 4 * 6);
});

test("F7b 递推状态：plain 2T·dk·dv，delta 3T·dk·dv（delta matvec 是矩阵 MACs）；bytes 由状态主导", () => {
  const plain = linearAttentionStateCounts({ tokens: 3, keyDim: 8, valueDim: 8, bytesPerElement: B });
  assert.equal(plain.matrix, 2 * 3 * 64);
  assert.equal(plain.vector, 3 * 64);
  assert.equal(plain.sfu, 3);
  const delta = linearAttentionStateCounts({ tokens: 3, keyDim: 8, valueDim: 8, bytesPerElement: B, delta: true });
  assert.equal(delta.matrix, 3 * 3 * 64);
  assert.equal(delta.vector, 2 * 3 * 64);
  assert.equal(delta.sfu, 2 * 3); // exp decay + sigmoid beta
  // 状态读+写主导：actIn = 2·T·dk·dv·b ≥ q/k/v 流（此处无其他流）
  assert.equal(plain.bytes.actIn, 2 * 3 * 64 * B);
});

test("F8 MoE：topk 选择 + dispatch/combine 搬运 + combine 加权求和计 vector", () => {
  const top = topkCounts({ tokens: 2, experts: 8, topk: 2, bytesPerElement: B });
  assert.equal(top.vector, 2 * 8);
  assert.equal(top.sfu, 2 * 2); // norm_topk_prob
  const dispatch = moeDispatchCounts({ tokens: 2, hidden: 16, topk: 2, bytesPerElement: B });
  assert.equal(dispatch.matrix, 0);
  assert.equal(dispatch.bytes.actOut, 2 * 2 * 16 * B);
  const combine = moeCombineCounts({ tokens: 2, hidden: 16, topk: 2, bytesPerElement: B });
  assert.equal(combine.vector, 2 * 2 * 16 * 2); // 乘 + 累加
  assert.equal(combine.bytes.actIn, (2 * 2 * 16 + 2 * 2) * B); // 专家输出 + 路由权重
});

test("F8 hash 路由：纯查表，零计算", () => {
  const c = hashRouteCounts({ tokens: 2, topk: 2, tableRows: 100, bytesPerElement: B });
  assert.equal(c.matrix + c.vector + c.sfu, 0);
  assert.equal(c.bytes.weights, 100 * B);
});

test("F9 重排：split 是视图零流量（A1）；vision_merge 真拷贝", () => {
  const view = rearrangeCounts();
  assert.deepEqual(view.bytes, { weights: 0, actIn: 0, actOut: 0 });
  const copy = rearrangeCounts({ copy: true, inElements: 24, outElements: 24, bytesPerElement: B });
  assert.equal(copy.bytes.actIn, 24 * B);
  assert.equal(copy.bytes.actOut, 24 * B);
});

test("独立 softmax：3E 向量 + 2E SFU（exp+div），A2 单遍", () => {
  const c = softmaxCounts({ elements: 12, bytesPerElement: B });
  assert.equal(c.matrix, 0);
  assert.equal(c.vector, 3 * 12);
  assert.equal(c.sfu, 2 * 12);
});

test("复合节点：mla_query_compress = 两段 linear + 一段 rmsnorm", () => {
  const entry = formulaForOperator("mla_query_compress");
  const c = entry.counts({
    qa: { logicalShape: [16, 32], tokens: 2, bytesPerElement: B },
    norm: { tokens: 2, hidden: 16, bytesPerElement: B },
    qb: { logicalShape: [32, 16], tokens: 2, bytesPerElement: B },
  });
  assert.equal(c.matrix, 2 * 16 * 32 + 2 * 32 * 16);
  assert.equal(c.vector, 3 * 2 * 16); // rmsnorm 部分
  assert.equal(c.bytes.weights, (16 * 32 + 16 + 32 * 16) * B);
});

test("注册表完整性：42 个条目全部终止于 counts（无白名单，§3.1）", () => {
  const live = ["linear","matmul","softmax","split","causal_conv1d","rope","vision_position","vision_merge","vision_activation","rmsnorm","gemma_rmsnorm","swiglu","topk","moe_dispatch","moe_combine","moe_add","linear_attention","linear_attention_gate","gated_delta_attention","gated_rmsnorm","mhc_pre","mhc_fused_post_pre","mhc_post","mhc_contract","mla_query_compress","mla_kv_compress","mla_kv_split","mla_output_gate","attention_residual","hyper_connection","ple","shared_expert_gate","qsa_indexer","qsa_attention","qwen_qkvz_split","attention_qkv_split","attention_output_gate","minimax_sparse_indexer","minimax_sparse_attention","dsv4_hash_route","dsv4_swa_attention","dsv4_compressed_attention"];
  // 复合节点的 ctx 是嵌套结构，数值由各自的复合用例覆盖（如 mla_query_compress）
  const composites = new Set(["mhc_pre","mhc_fused_post_pre","mhc_post","mhc_contract","mla_query_compress","mla_kv_compress","attention_residual","hyper_connection","ple","qsa_indexer","minimax_sparse_indexer"]);
  for (const key of live) {
    const entry = formulaForOperator(key);
    assert.ok(entry, `条目缺失: ${key}`);
    assert.equal(typeof entry.counts, "function", `counts 未接线: ${key}`);
    if (composites.has(key)) continue;
    const sample = entry.counts({ elements: 1, tokens: 1, hidden: 1, bytesPerElement: 1, width: 1, intermediate: 1, experts: 1, topk: 1, keyDim: 1, valueDim: 1, keyTokens: 1, headDim: 1, heads: 1, queryTokens: 1, valueDim2: 1, ropeDims: 1, channels: 1, kernel: 1, tableRows: 1, logicalShape: [1, 1], inElements: 1, outElements: 1, gateProjection: false, gateProjectionInput: 0, weightOne: false, gated: false, delta: false, normTopkProb: false, copy: false });
    assert.ok(Number.isFinite(sample.matrix), `matrix 非有限: ${key}`);
    assert.ok(Number.isFinite(sample.bytes.actIn), `bytes 非有限: ${key}`);
  }
});
