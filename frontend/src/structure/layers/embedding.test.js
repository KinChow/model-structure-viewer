import assert from "node:assert/strict";
import test from "node:test";
import { ngramEmbeddingTable } from "./embedding.js";

test("ngramEmbeddingTable copies Qwen4Exp prime-pad (Flash-Next config)", () => {
  const table = ngramEmbeddingTable({
    pleNgramSize: 3,
    pleHeadsPerNgram: 8,
    pleEmbedDim: 2560,
    ngramVocabSizeBase: 20_000_000,
    makeNgramVocabSizeDivisibleBy: 128,
  });
  assert.equal(table.ngramHeads, 16);
  assert.equal(table.headDim, 160);
  assert.equal(table.paddedVocab, 320001536);
  assert.equal(table.paddedVocab * table.headDim, 51200245760);
});
