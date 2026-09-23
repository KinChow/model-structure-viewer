import assert from "node:assert/strict";
import test from "node:test";
import { qsaGeometry, qsaIndexerCounts, qsaSparseAttentionCounts } from "../qsa.js";

// Independent literal causal enumeration, not copied closed-form prefixes.
test("QSA closed-form counts match complete blocks plus a visible tail", () => {
  for (const pool of [1, 2, 4]) for (const budget of [4, 8]) {
    for (const batch of [1, 3]) for (let sequence = 1; sequence <= 25; sequence++) {
      for (const phase of ["prefill", "decode"]) {
        let blocks = 0, chosen = 0, selected = 0;
        const positions = phase === "decode" ? [sequence] : Array.from({ length: sequence }, (_, i) => i + 1);
        for (const visible of positions) {
          const full = Math.floor(visible / pool);
          blocks += full;
          chosen += Math.min(full, Math.floor(budget / pool));
          selected += Math.min(full, Math.floor(budget / pool)) * pool + visible % pool;
        }
        const got = qsaGeometry({ pool, budget, batch, phase, keyTokens: sequence, queryTokens: batch * positions.length });
        assert.equal(got.blockPairs, batch * blocks);
        assert.equal(got.selectedBlockPairs, batch * chosen);
        assert.equal(got.selectedTokenPairs, batch * selected);
      }
    }
  }
});

test("QSA released budget permits 2051 tokens; next complete block removes the tail", () => {
  for (const [keyTokens, selected] of [[1, 1], [3, 3], [4, 4], [2048, 2048], [2051, 2051], [2052, 2048]]) {
    assert.equal(qsaGeometry({ keyTokens, queryTokens: 1, phase: "decode" }).selected, selected);
  }
  // Non-divisible synthetic budget follows the reference code floor, not the
  // report's ceil-and-truncate. No claim of equivalence for this configuration.
  assert.equal(qsaGeometry({ keyTokens: 8, queryTokens: 1, phase: "decode", budget: 5 }).selected, 4);
});

test("small QSA has exact projection/scoring MACs and separate raw-key traffic", () => {
  // B=2, T=S=5, H_i=2, d_i=2, hidden=3, r=4, budget=4.
  // Per sequence there are two full-block query pairs (t=4,5);
  // selected tokens across queries = 1+2+3+4+5 = 15.
  const p = { heads: 2, dim: 2, inDim: 3, ropeDim: 2, b: 2,
    batch: 2, queryTokens: 10, keyTokens: 5, pool: 4, budget: 4, phase: "prefill" };
  const index = qsaIndexerCounts(p);
  assert.equal(index.matrix, 10 * 3 * 6 + 4 * 2 * 2);
  assert.equal(index.bytes.weights, (3 * 6 + 2 + 2) * 2);
  assert.equal(index.bytes.indexRead, 2 * 4 * 2 * 2);
  const sparse = qsaSparseAttentionCounts({ ...p, heads: 4, kvHeads: 1, headDim: 2, valueDim: 2, bytesPerElement: 2 });
  assert.equal(sparse.matrix, 2 * 15 * 4 * (2 + 2));
  assert.equal(sparse.vector, 4 * 2 * 15 * 4);
  assert.equal(sparse.sfu, 2 * 2 * 15 * 4);
  assert.equal(sparse.bytes.kvRead, 2 * 1 * 5 * (2 + 2) * 2);
  assert.ok(index.vector > 0);
  assert.ok(index.sfu > 0);
});
