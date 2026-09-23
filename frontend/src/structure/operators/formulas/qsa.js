// Qwen technical report §2.1.2, Fig.3 / Eq.12–16 and the released
// Qwen4ExpTextQSAIndexer.forward. No per-GQA-group selector (unlike MSA).
import { evaluateDecomposition } from "./atoms.js";
import { linearAtomSteps, rmsnormAtomSteps, sparseLeafAttentionCounts } from "./counts.js";

// Sum floor(t/r), t=1..n. Closed form avoids work proportional to a million-token context.
function completeBlockPrefix(n, r) {
  const m = Math.floor(n / r);
  return r * m * (m - 1) / 2 + m * (n % r + 1);
}

export function qsaGeometry({ keyTokens = 1, queryTokens = keyTokens, batch = 1, phase = "prefill", pool = 4, budget = 2048 } = {}) {
  const r = Math.max(1, pool);
  const k = Math.floor(budget / r);
  const blocks = Math.floor(keyTokens / r);
  const tail = keyTokens % r;
  const selected = Math.min(blocks, k) * r + tail;
  const queries = queryTokens / batch;
  const end = keyTokens;
  const start = phase === "decode" ? Math.max(0, end - 1) : Math.max(0, end - queries);
  const prefix = n => {
    const complete = completeBlockPrefix(n, r);
    const capStart = Math.max(0, k * r - 1);
    const selectedBlocks = completeBlockPrefix(Math.min(n, capStart), r) + Math.max(0, n - capStart) * k;
    const tails = n * (n + 1) / 2 - r * complete;
    return { complete, selectedBlocks, selectedTokens: selectedBlocks * r + tails };
  };
  const hi = prefix(end), lo = prefix(start);
  return {
    blocks, tail, selected, blockBudget: k,
    blockPairs: batch * (hi.complete - lo.complete),
    selectedBlockPairs: batch * (hi.selectedBlocks - lo.selectedBlocks),
    selectedTokenPairs: batch * (hi.selectedTokens - lo.selectedTokens),
    // A per-forward pool-once estimate, not a claim about fused runtime kernels.
    pooledKeyRows: batch * blocks,
  };
}

export function qsaIndexerDecomposition(p) {
  const { heads = 1, dim = 1, inDim = 1, queryTokens = 1, keyTokens = 1, pool = 4, b = 2, ropeDim = 0, batch = 1 } = p;
  const g = qsaGeometry(p);
  const scoreElements = heads * g.blockPairs;
  const norm = rows => rmsnormAtomSteps({ tokens: rows, hidden: dim, b, weightOne: true }).decompose;
  return [
    ...linearAtomSteps({ tokens: queryTokens, inDim, out: (heads + 1) * dim, b }).decompose,
    ...norm(queryTokens * heads),
    { atom: "rope", args: { elements: queryTokens * heads * ropeDim, bytesPerElement: b } },
    { atom: "scatter", args: { rows: queryTokens, width: dim, bytesPerElement: b, readIn: false } },
    { atom: "reduce_sum", args: { elements: g.pooledKeyRows * pool * dim, groups: g.pooledKeyRows * dim, bytesPerElement: b } },
    { atom: "scale", args: { elements: g.pooledKeyRows * dim, bytesPerElement: b } },
    ...norm(g.pooledKeyRows),
    { atom: "rope", args: { elements: g.pooledKeyRows * ropeDim, bytesPerElement: b } },
    { atom: "matmul", args: { m: g.blockPairs, n: heads, k: dim, bytesPerElement: 4,
      lhsElements: queryTokens * heads * dim, rhsElements: g.pooledKeyRows * dim, outElements: scoreElements } },
    { atom: "relu", args: { elements: scoreElements, bytesPerElement: 4 } },
    { atom: "reduce_sum", args: { elements: scoreElements, groups: g.blockPairs, bytesPerElement: 4 } },
    { atom: "scale", args: { elements: g.blockPairs, bytesPerElement: 4 } },
    // Flatten the ragged query rows into their exact logical pair counts; scan
    // comparisons are a first-order estimate, not an implementation of Top-k sorting.
    { atom: "topk", args: { rows: 1, candidates: g.blockPairs, k: g.selectedBlockPairs, bytesPerElement: 4 } },
    { atom: "gather", args: { rows: g.selectedBlockPairs, width: pool, bytesPerElement: 4 } },
    { atom: "gather", args: { rows: g.selectedTokenPairs - g.selectedBlockPairs * pool, width: 1, bytesPerElement: 4 } },
  ];
}

export function qsaIndexerCounts(p) {
  const counts = evaluateDecomposition(qsaIndexerDecomposition(p));
  const { pooledKeyRows } = qsaGeometry(p);
  return { ...counts, bytes: { ...counts.bytes, indexRead: pooledKeyRows * (p.pool || 4) * (p.dim || 1) * (p.b || 2) } };
}

export function qsaSparseAttentionCounts(p) {
  const g = qsaGeometry(p);
  const counts = sparseLeafAttentionCounts({
    ...p, tokens: p.queryTokens, selected: g.selected,
    scorePairs: g.selectedTokenPairs,
    indexElements: g.selectedTokenPairs, indexBytes: 4,
    kvReadBatch: p.batch ?? 1,
  });
  // Scaling + softmax follow the repository's action convention (4 vector,
  // 2 SFU per visible head-score), rather than silently omitting softmax.
  const scores = (p.heads ?? 1) * g.selectedTokenPairs;
  return { ...counts, vector: 4 * scores, sfu: 2 * scores };
}
