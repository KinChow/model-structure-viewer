// moduleProbeParams.js —— 模块层恒等式的「代表工作点」参数。
//
// 为什么单独一个文件：这套参数被**两处**消费 ——
//   1. __tests__/identities.test.js（融合分解恒等式的断言）
//   2. scripts/gen-operators-reference.mjs（表的机器段要打印恒等式结果与融合收益）
// 抄两份必然漂移，所以定在这里，两边 import。
//
// 参数含义：给定 (normalized config, phase) 返回该模块在这个结构类下的代表参数；
// 返回 null 表示该结构类不含此模块（调用方跳过，不算不闭合）。

import { deriveBuildPlan } from "../model_executor/plan.js";

/** 操作符 id -> 模块 id。只登记「算子就是模块」的那些；一对多/多对一的不登记。 */
export const OPERATOR_TO_MODULE = Object.freeze({
  linear: "linear",
  rmsnorm: "rmsnorm",
  gemma_rmsnorm: "rmsnorm",
  gated_rmsnorm: "rmsnorm",
  rope: "rope",
  swiglu: "swiglu",
  softmax: "softmax",
  topk: "topk_router",
  dsa_indexer: "dsa_indexer",
  dsa_kpool_indexer: "dsa_kpool_indexer",
  qsa_indexer: "qsa_indexer",
  minimax_sparse_indexer: "minimax_block_indexer",
  gated_delta_attention: "linear_attention_state",
  mla_query_compress: "mla_query_compress",
  mla_kv_compress: "mla_kv_compress",
});

/** 该模块在给定 (config, phase) 下的代表参数；null = 本结构类不含此模块。 */
export function moduleParamsFor(id, c, ph, bytesPerElement = 2) {
  const b = bytesPerElement;
  const tokens = ph.tokens;
  const S = ph.sequence;
  const heads = c.attentionHeads || 0;
  const dim = c.indexerHeadDim || 0;
  switch (id) {
    case "linear":
      return { tokens, inDim: c.hiddenSize || 0, out: c.hiddenSize || 0, bias: false, b };
    case "rmsnorm":
      return { tokens, hidden: c.hiddenSize || 0, weightOne: false, gated: false, b };
    case "rope":
      return { tokens, ropeDims: c.qkRopeHeadDim || c.headDim || 0, b };
    case "swiglu":
      return { tokens, intermediate: c.intermediateSize || 0, b };
    case "gate":
      return { tokens, width: c.hiddenSize || 0, gateProjection: false, b };
    case "softmax":
      return { elements: heads * tokens * S, b };
    case "topk_router":
      return c.experts ? { tokens, experts: c.experts, topk: c.expertsPerToken || 1, normTopkProb: c.normTopkProb ?? true, b } : null;
    case "mla_query_compress":
      // 只有带 q_lora_rank 的 MLA 家族有 query 压缩；V4 无 kv_lora 但有 q_lora。
      return c.qLoraRank ? { tokens, hidden: c.hiddenSize || 0, rank: c.qLoraRank, b } : null;
    case "mla_kv_compress":
      // out = latent + rope 分量宽（与 ctxBuilder 的无形状回退同口径）。
      return c.kvLoraRank ? { tokens, hidden: c.hiddenSize || 0, out: c.kvLoraRank + (c.qkRopeHeadDim || 0), b } : null;
    case "dsv4_hash_route":
      // tid2eid 是 buffer（容量走 derivedBufferBytes）；模块只计 gather 流量。
      return c.numHashLayers ? { tokens, topk: c.expertsPerToken || 1, b } : null;
    case "vision_position":
    case "vision_activation":
      // 视觉部件只在有视觉塔的结构类存在
      return c.hasVision ? { tokens: c.visionTokens || 1, hidden: c.visionHiddenSize || 0, intermediate: c.visionIntermediateSize || 0, b } : null;
    case "mhc_pre":
    case "mhc_fused_post_pre":
      return c.multiHyperConnection ? {
        tokens, hidden: c.hiddenSize || 0,
        streams: c.mhcNumResidualStreams || 0,
        mixRows: (2 + (c.mhcNumResidualStreams || 0)) * (c.mhcNumResidualStreams || 0),
        hcDim: (c.mhcNumResidualStreams || 0) * (c.hiddenSize || 0),
        iterations: c.mhcSinkhornIterations || 0,
        b,
      } : null;
    case "mhc_contract":
      return c.multiHyperConnection ? { tokens, hidden: c.hiddenSize || 0, b } : null;
    case "ple":
      return c.pleEmbedDim ? { tokens, hidden: c.hiddenSize || 0, embedDim: c.pleEmbedDim, ngram: c.pleNgramSize || 1, b } : null;
    case "hyper_connection":
      return c.hyperConnectionCount ? { tokens, hidden: c.hiddenSize || 0, streams: c.hyperConnectionCount, lowrank: c.hyperConnectionLowrank || 0, b } : null;
    case "attention_residual":
      return c.attnResBlockSize ? { tokens, hidden: c.hiddenSize || 0, b } : null;
    case "vision_merge":
      // 内融合器只在 plan.visionInternalMerger 的结构类存在（S13 走外置
      // projector，树上没有 vision_merge 叶，模块恒等式不该覆盖它）。
      return c.hasVision && deriveBuildPlan(c.raw ?? c).visionInternalMerger
        ? { tokens: c.visionTokens || 1, inWidth: c.visionHiddenSize || 0, mergeSize: c.visionMergeSize || 1, b }
        : null;
    case "sdpa_attention":
      return heads ? {
        heads, kvHeads: c.kvHeads || heads, queryTokens: tokens, keyTokens: S,
        headDim: c.headDim || 0, valueDim: c.valueHeadDim || c.headDim || 0, phase: ph.name, b,
      } : null;
    case "dsa_indexer":
      return (c.indexerBudget && c.kvLoraRank && !(c.raw?.index_kpool > 1)) ? {
        heads: c.indexerNHeads || 0, dim, queryTokens: tokens, keyTokens: S,
        budget: c.indexerBudget, pool: 1, perHeadWeights: true, phase: ph.name, b,
      } : null;
    case "dsa_kpool_indexer":
      return (c.raw?.index_kpool > 1) ? {
        heads: c.indexerNHeads || 0, dim, queryTokens: tokens, keyTokens: S,
        budget: c.indexerBudget, pool: c.raw.index_kpool, perHeadWeights: true, phase: ph.name, b,
      } : null;
    case "qsa_indexer":
      return c.indexerKVHeads ? {
        heads: c.indexerNHeads || 0, dim, queryTokens: tokens, keyTokens: S,
        budget: c.indexerBudget, pool: c.indexerCompressRatio || 1, perHeadWeights: false, phase: ph.name, b,
      } : null;
    case "minimax_block_indexer":
      return c.sparseBlockSize ? {
        heads: c.sparseIndexHeads || 0, dim: c.sparseIndexDim || 0, queryTokens: tokens, keyTokens: S,
        budget: (c.sparseTopkBlocks || 0) * (c.sparseBlockSize || 1), pool: c.sparseBlockSize,
        perHeadWeights: false, phase: ph.name, b,
      } : null;
    case "linear_attention_state":
      return c.linearKeyHeads ? {
        tokens, heads: c.linearValueHeads || c.linearKeyHeads,
        keyDim: c.linearKeyDim || 0, valueDim: c.linearValueDim || 0,
        delta: true, phase: ph.name, chunkSize: 64, b,
      } : null;
    default:
      return null;
  }
}
