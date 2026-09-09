// golden per-op 用例：手算期望值（照 PyTorch test_flop_counter 风格）。
// 期望值独立于实现手算而来；发现不一致时先查数学，不得改期望值迁就实现。
import assert from "node:assert/strict";
import test from "node:test";
import { formulaForOperator } from "../index.js";
import {
  linearCounts, attentionCounts, rmsnormCounts, gateCounts, swigluCounts,
  ropeCounts, causalConvCounts, linearAttentionStateCounts, topkCounts,
  moeDispatchCounts, moeCombineCounts, addCounts, hashRouteCounts,
  rearrangeCounts, softmaxCounts, scoredPairs, causalDensity,
} from "../counts.js";

const B = 2; // bf16 每元素 2 字节

test("F1 linear：T·out·in MACs，权重整表读一遍", () => {
  const c = linearCounts({ logicalShape: [8, 4], tokens: 3, bytesPerElement: B, bias: true });
  assert.equal(c.matrix, 3 * 8 * 4);
  assert.equal(c.vector, 3 * 8); // bias 加法
  assert.equal(c.sfu, 0);
  // bias 也是权重（out 个），与本模块的原子分解（add 原子带 weightElements: out）
  // 和 compulsoryBytes 同口径。2026-09-09 补齐前只记权重矩阵。
  assert.equal(c.bytes.weights, (8 * 4 + 8) * B);
  assert.equal(c.bytes.actIn, 3 * 4 * B);
  assert.equal(c.bytes.actOut, 3 * 8 * B);
});

test("F2 选择集注意力：两组 bmm + softmax（A2 单遍）+ KV cache 写", () => {
  const c = attentionCounts({ heads: 2, queryTokens: 4, keyTokens: 8, headDim: 16, valueDim: 16, bytesPerElement: B });
  // W3 因果口径手算：T=4、S=8，前缀 4 个 key 全可见（4·4=16）+ 新 4 个因果
  // （1+2+3+4=10）= 26 对/头；scores = heads·26 = 52。
  const scores = 2 * 26;
  assert.equal(scores, 52);
  assert.equal(c.matrix, scores * (16 + 16)); // scores bmm + context bmm
  // W5：4·scores = scale(1) + softmax(3)
  assert.equal(c.vector, 4 * scores);
  assert.equal(c.sfu, 2 * scores);
  const q = 2 * 4 * 16, k = 2 * 8 * 16, v = 2 * 8 * 16;
  assert.equal(c.bytes.actIn, (q + k + v + 2 * scores) * B); // Q/K/V 各一遍 + scores 写读
  // actOut：scores + probs 写读 + 输出 + 新算 K/V 写回 cache（T 个 token，非 S）
  assert.equal(c.bytes.actOut, (2 * scores + 2 * 4 * 16 + 2 * 4 * 32) * B);
});

test("F2 GQA/MQA：matrix 不随 kvHeads 变，K/V 流量随 kvHeads 缩小", () => {
  // 32 query 头、8 KV 头（GQA）：K/V 读/写是 MHA 的 1/4
  const scores = 32 * 26; // 同上：T=4、S=8 → 26 对/头
  const mha = attentionCounts({ heads: 32, queryTokens: 4, keyTokens: 8, headDim: 16, valueDim: 16, bytesPerElement: B });
  const gqa = attentionCounts({ heads: 32, queryTokens: 4, keyTokens: 8, headDim: 16, valueDim: 16, bytesPerElement: B, kvHeads: 8 });
  assert.equal(gqa.matrix, mha.matrix); // 每个 query head 都要做完整点积
  const nonKv = (32 * 4 * 16 + 2 * scores) * B; // Q 读 + scores 写读
  const kvBytesMha = mha.bytes.actIn - nonKv;
  const kvBytesGqa = gqa.bytes.actIn - nonKv;
  assert.equal(kvBytesGqa * 4, kvBytesMha);
  // MQA：kvHeads=1
  const mqa = attentionCounts({ heads: 32, queryTokens: 4, keyTokens: 8, headDim: 16, valueDim: 16, bytesPerElement: B, kvHeads: 1 });
  assert.equal(mqa.bytes.actIn - nonKv, kvBytesMha / 32);
});

test("F2 MLA：共享 latent → kvHeads=1，打分宽度 = latent + rope，V 宽 = latent", () => {
  // DeepSeek MLA 典型值：kv_lora_rank=512, rope=64
  const scores = 128 * 26; // T=4、S=8 因果口径
  const c = attentionCounts({ heads: 128, queryTokens: 4, keyTokens: 8, headDim: 512 + 64, valueDim: 512, bytesPerElement: B, kvHeads: 1 });
  assert.equal(c.matrix, scores * (576 + 512));
  const kvShared = (8 * 576 + 8 * 512) * B;
  assert.equal(c.bytes.actIn - (128 * 4 * 576 + 2 * scores) * B, kvShared);
});

test("F2 prefill vs decode：因果三角 vs 全长，decode 的 K/V 读即读 cache", () => {
  const heads = 8, headDim = 128, valueDim = 128, S = 4096;
  // prefill：T=S=4096，因果对数 = Σ_{t=1..4096} t = 4096·4097/2 = 8,390,656
  const prefillPairs = (S * (S + 1)) / 2;
  assert.equal(prefillPairs, 8390656);
  const prefill = attentionCounts({ heads, queryTokens: S, keyTokens: S, headDim, valueDim, bytesPerElement: B, phase: "prefill" });
  // decode：T=1，S=4096（全部 cached K/V）
  const decode = attentionCounts({ heads, queryTokens: 1, keyTokens: S, headDim, valueDim, bytesPerElement: B, phase: "decode" });
  assert.equal(prefill.matrix, heads * prefillPairs * (headDim + valueDim));
  assert.equal(decode.matrix, heads * S * (headDim + valueDim));
  // 两相位比值 = (S+1)/2，不是 S —— 因果减半是 W3 的核心口径修正
  assert.equal(prefill.matrix / decode.matrix, (S + 1) / 2);
  // decode actIn 的 K/V 部分 = 读整个 KV cache（S 个 token）
  const kvRead = heads * S * (headDim + valueDim) * B;
  assert.equal(decode.bytes.actIn - (heads * 1 * headDim + 2 * heads * S) * B, kvRead);
  // decode 写回 cache 只有当前 1 个 token 的 K/V
  const kvWritePerToken = heads * 1 * (headDim + valueDim) * B;
  const decodeIntermediate = (2 * heads * S + heads * 1 * valueDim) * B;
  assert.equal(decode.bytes.actOut - decodeIntermediate, kvWritePerToken);
  // prefill 写回全量 cache
  const prefillKvWrite = heads * S * (headDim + valueDim) * B;
  const prefillIntermediate = (2 * heads * prefillPairs + heads * S * valueDim) * B;
  assert.equal(prefill.bytes.actOut - prefillIntermediate, prefillKvWrite);
});

test("F3 rmsnorm：3TH 向量 + T 次 rsqrt；gemma 多 TH；gated 多一路门", () => {
  const base = rmsnormCounts({ tokens: 5, hidden: 10, bytesPerElement: B });
  assert.equal(base.matrix, 0);
  assert.equal(base.vector, 4 * 5 * 10 - 5); // 均方求和每 token 少一次加法
  assert.equal(base.sfu, 5);
  assert.equal(base.bytes.weights, 10 * B);
  const gemma = rmsnormCounts({ tokens: 5, hidden: 10, bytesPerElement: B, weightOne: true });
  assert.equal(gemma.vector, 5 * 5 * 10 - 5);
  const gated = rmsnormCounts({ tokens: 5, hidden: 10, bytesPerElement: B, gated: true });
  assert.equal(gated.vector, 5 * 5 * 10 - 5);
  assert.equal(gated.sfu, 5 + 2 * 5 * 10);
});

test("F4 门控乘：TW 次 sigmoid + TW 次乘；带输入投影时计权重", () => {
  const bare = gateCounts({ tokens: 4, width: 8, bytesPerElement: B });
  assert.equal(bare.matrix, 0);
  assert.equal(bare.sfu, 2 * 4 * 8); // sigmoid = exp + rcp
  assert.equal(bare.bytes.weights, 0);
  const projected = gateCounts({ tokens: 4, width: 8, bytesPerElement: B, gateProjection: true, gateProjectionInput: 6 });
  assert.equal(projected.bytes.weights, 6 * 8 * B);
});

test("F5 SwiGLU：2TI 乘 + TI 次 sigmoid；输入是 gate/up 两路", () => {
  const c = swigluCounts({ tokens: 3, intermediate: 8, bytesPerElement: B });
  assert.equal(c.matrix, 0);
  assert.equal(c.vector, 2 * 3 * 8);
  assert.equal(c.sfu, 2 * 3 * 8); // silu = sigmoid(2 SFU) + mul
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
  assert.equal(c.sfu, 2 * 4 * 6); // silu
});

test("F7b 递推状态：plain 2T·dk·dv，delta 3T·dk·dv（delta matvec 是矩阵 MACs）；多头 state 流量显式", () => {
  const plain = linearAttentionStateCounts({ tokens: 3, heads: 4, keyDim: 8, valueDim: 8, bytesPerElement: B });
  assert.equal(plain.matrix, 2 * 3 * 4 * 64);
  assert.equal(plain.vector, 3 * 4 * 64);
  assert.equal(plain.sfu, 4 * 3); // decay exp 每 head 1 次
  const delta = linearAttentionStateCounts({ tokens: 3, heads: 4, keyDim: 8, valueDim: 8, bytesPerElement: B, delta: true });
  assert.equal(delta.matrix, 3 * 3 * 4 * 64);
  // W5：外积/delta matvec/query 三段归 matrix，vector 只剩 decay 逐元素乘 = steps·state
  assert.equal(delta.vector, 3 * 4 * 64);
  assert.equal(delta.sfu, 3 * 4 * 3); // exp decay + sigmoid beta(2)
  // 状态读+写主导：actIn = 2·steps·heads·dk·dv·b（多头 state 显式进入流量）
  assert.equal(plain.bytes.actIn, 2 * 3 * 4 * 64 * B);
  // W5：chunked 实现每块与状态交互一次 —— stateSteps 显式传入时按块数计
  const chunked = linearAttentionStateCounts({ tokens: 128, heads: 4, keyDim: 8, valueDim: 8, bytesPerElement: B, stateSteps: 2 });
  assert.equal(chunked.matrix, 2 * 128 * 4 * 64); // MACs 与分块无关
  assert.equal(chunked.vector, 2 * 4 * 64);       // decay 只做 2 次
  assert.equal(chunked.bytes.actIn, 2 * 2 * 4 * 64 * B);
});

test("F8 MoE：topk 选择 + dispatch/combine 搬运 + combine 加权求和计 vector", () => {
  const top = topkCounts({ tokens: 2, experts: 8, topk: 2, bytesPerElement: B });
  // W5：T·E 扫描 + 归一化求和 T·(k-1) = 16 + 2 = 18
  assert.equal(top.vector, 2 * 8 + 2 * (2 - 1));
  assert.equal(top.sfu, 2 * 2); // norm_topk_prob
  const dispatch = moeDispatchCounts({ tokens: 2, hidden: 16, topk: 2, bytesPerElement: B });
  assert.equal(dispatch.matrix, 0);
  assert.equal(dispatch.bytes.actOut, 2 * 2 * 16 * B);
  const combine = moeCombineCounts({ tokens: 2, hidden: 16, topk: 2, bytesPerElement: B });
  assert.equal(combine.vector, 2 * 2 * 16 * 2); // 乘 + 累加
  assert.equal(combine.bytes.actIn, (2 * 2 * 16 + 2 * 2) * B); // 专家输出 + 路由权重
});

test("F8 hash 路由：tid2eid 是 buffer（零权重字节），gather 读=写=tokens·topk", () => {
  // 2026-09-09 分类裁决：tid2eid 是 buffer 不是参数（Megatron-Bridge 明文），
  // 表的常驻容量（vocab·k·4B）由 derivedBufferBytes 计入显存，不进权重字节；
  // 本算子只计 gather 的真实拷贝流量。
  const c = hashRouteCounts({ tokens: 2, topk: 2, bytesPerElement: B });
  assert.equal(c.matrix + c.vector + c.sfu, 0);
  assert.equal(c.bytes.weights, 0);
  assert.equal(c.bytes.actIn, 2 * 2 * B);
  assert.equal(c.bytes.actOut, 2 * 2 * B);
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

test("复合节点：mla_query_compress 只计 q_a 投影（norm 与 q_b 都是独立叶）", () => {
  const entry = formulaForOperator("mla_query_compress");
  const c = entry.counts({
    qa: { logicalShape: [16, 32], tokens: 2, bytesPerElement: B },
  });
  assert.equal(c.matrix, 2 * 16 * 32);
  // q_a_layernorm 是独立的 `q_a_norm` 叶（结构树实证 decoder.N.self_attn.q_a_norm），
  // q_b_proj 是独立 linear 叶；算进本叶就是双计。norm 那份由 2026-09-09 的
  // 权重字节逐层归因抓出（Kimi-K2 每层 +1,536，61 层 = 93,696）。
  assert.equal(c.vector, 0);
  assert.equal(c.bytes.weights, 16 * 32 * B);
});

test("因果对数解析检查：scoredPairs 与逐 token 暴力求和一致（破同义重复）", () => {
  // 期望侧与实现侧共用 scoredPairs 会构成同义重复；这里用小尺寸暴力循环
  // 做独立 oracle：prefill 下第 t 个 query（1-based）可见 t 个 key。
  for (const T of [1, 2, 3, 4, 8, 17]) {
    let brute = 0;
    for (let t = 1; t <= T; t += 1) brute += t;
    assert.equal(scoredPairs({ phase: "prefill", queryTokens: T, keyTokens: T }), brute, `T=${T}`);
  }
  // 带前缀 cache：前缀全可见 + 新 token 内部因果
  // T=3, S=10 → 前缀 7 个全可见（3·7=21）+ 新 3 个因果（1+2+3=6）= 27
  assert.equal(scoredPairs({ phase: "prefill", queryTokens: 3, keyTokens: 10 }), 27);
  // decode：T=1 看全长
  assert.equal(scoredPairs({ phase: "decode", queryTokens: 1, keyTokens: 4096 }), 4096);
  // 密度 = 对数 / 稠密对数
  assert.equal(causalDensity({ phase: "prefill", queryTokens: 4, keyTokens: 4 }), 10 / 16);
  assert.equal(causalDensity({ phase: "decode", queryTokens: 1, keyTokens: 8 }), 1);
});

test("F2 分相位：prefill 因果三角、decode 全长", () => {
  const base = { heads: 2, headDim: 4, valueDim: 4, bytesPerElement: 2, kvHeads: 1 };
  const pre = attentionCounts({ ...base, queryTokens: 4, keyTokens: 4, phase: "prefill" });
  const dec = attentionCounts({ ...base, queryTokens: 1, keyTokens: 4, phase: "decode" });
  // prefill: scores = heads·10 = 20；matrix = 20·(4+4) = 160
  assert.equal(pre.matrix, 160);
  assert.equal(pre.vector, 80); // 4·scores = 4·20
  assert.equal(pre.sfu, 40);
  // decode: scores = heads·4 = 8；matrix = 8·8 = 64
  assert.equal(dec.matrix, 64);
});

test("注册表完整性：47 个条目全部终止于 counts（无白名单，§3.1）", () => {
  // W2：单一 qsa_indexer / qsa_attention 拆成四 indexer + 三 sparse attention
  // （算法出处不同不共用条目，见 formulas/index.js 各条 ref）。
  const live = ["linear","matmul","softmax","split","causal_conv1d","rope","vision_position","vision_merge","vision_activation","rmsnorm","gemma_rmsnorm","swiglu","topk","moe_dispatch","moe_combine","moe_add","linear_attention","linear_attention_gate","gated_delta_attention","gated_rmsnorm","mhc_pre","mhc_fused_post_pre","mhc_post","mhc_contract","mla_query_compress","mla_kv_compress","mla_kv_split","mla_output_gate","attention_residual","hyper_connection","ple","shared_expert_gate","qsa_indexer","dsa_indexer","dsa_kpool_indexer","dsv4_indexer","qsa_sparse_attention","dsa_sparse_mla","dsv4_sparse_mla","qwen_qkvz_split","attention_qkv_split","attention_output_gate","minimax_sparse_indexer","minimax_sparse_attention","dsv4_hash_route","dsv4_swa_attention","dsv4_compressed_attention"];
  // 复合节点的 ctx 是嵌套结构，数值由各自的复合用例覆盖（如 mla_query_compress）
  const composites = new Set(["mhc_pre","mhc_fused_post_pre","mhc_post","mhc_contract","mla_query_compress","mla_kv_compress","attention_residual","hyper_connection","ple","qsa_indexer","dsa_indexer","dsa_kpool_indexer","dsv4_indexer","minimax_sparse_indexer"]);
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
