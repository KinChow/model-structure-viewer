import { moduleSpec, withShapeDims } from "./base.js";
import { weightMatrixDecl } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";

export function embeddingModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(id, "embed tokens", "embedding", {
    class: "Embedding",
    hidden_size: normalized.hiddenSize,
    vocab_size: normalized.vocabSize,
    // P4-2：embed_tokens 持有 vocab×hidden 的权重表（容量/驻留口径）。gather
    // 的 counts.bytes.weights 为 0（流量按行计、M11 已入 actIn），所以锚 1 对
    // 本叶走登记例外（声明=驻留，不等于该相位读量）。vocab 亲和 =
    // ParallelLMEmbedding 语义（受 vocabParallel 支配，协议 §二）。
        // vLLM ParallelEmbedding 无 quant_method（量化只走 parallel Linear 包装），
    // embedding 表不参与量化 → quantizable=false。
    weightMatrices: [weightMatrixDecl("vocab", { shape: [normalized.vocabSize || 0, normalized.hiddenSize || 0], quantizable: false })],
    ...shapeFlow(shapes.tokenIds, shapes.hidden),
  }, [], undefined, "token_embd"), dims.tokenIds, dims.hidden);
}

// Qwen4ExpTextNGramEmbedding：每头一个 ≥ ngram_vocab_size_base 的素数，拼起来
// 再 pad 到 make_ngram_vocab_size_divisible_by。照抄
// models/Qwen/Qwen3.8-Flash-Next/modeling_qwen4_exp.py:_is_prime /
// _find_nth_prime_after / padded_vocab_size（:1058-1111）。
function isPrime(value) {
  if (value < 2) return false;
  if (value % 2 === 0) return value === 2;
  const limit = Math.floor(Math.sqrt(value));
  for (let divisor = 3; divisor <= limit; divisor += 2) {
    if (value % divisor === 0) return false;
  }
  return true;
}

function findNthPrimeAfter(start, count) {
  let prime = start;
  for (let i = 0; i < count; i += 1) {
    prime += 1;
    while (!isPrime(prime)) prime += 1;
  }
  return prime;
}

export function ngramEmbeddingTable(normalized, pleLayerIndex = 0) {
  const ngramSize = normalized.pleNgramSize || 0;
  const headsPerNgram = normalized.pleHeadsPerNgram || 0;
  const embedDim = normalized.pleEmbedDim || 0;
  const ngramHeads = Math.max((ngramSize - 1) * headsPerNgram, 0);
  const headDim = ngramHeads > 0 ? Math.floor(embedDim / ngramHeads) : 0;
  const base = normalized.ngramVocabSizeBase || 0;
  const divisor = normalized.makeNgramVocabSizeDivisibleBy || 1;
  let totalVocab = 0;
  for (let headIdx = 0; headIdx < ngramHeads; headIdx += 1) {
    totalVocab += findNthPrimeAfter(base - 1, pleLayerIndex * ngramHeads + headIdx + 1);
  }
  const paddedVocab = divisor > 0 ? Math.ceil(totalVocab / divisor) * divisor : totalVocab;
  return { paddedVocab, headDim, ngramHeads, embedDim };
}

/** PLE 的 ngram 表：torch.nn.Embedding，gather 无 MAC。分片 replicated——
 *  HF `_no_placement_params = ["ple.ple_embedding.ngram_embedding.weight"]`
 *  （modeling_qwen4_exp.py:1328），不是 ParallelLMEmbedding。 */
export function ngramEmbeddingModule(id, normalized, { pleLayerIndex = 0 } = {}) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const table = ngramEmbeddingTable(normalized, pleLayerIndex);
  const outputDims = [-1, -1, table.embedDim || 0];
  return withShapeDims(moduleSpec(id, "ngram embedding", "embedding", {
    class: "Embedding",
    hidden_size: table.headDim,
    vocab_size: table.paddedVocab,
    weightMatrices: [weightMatrixDecl("replicated", {
      shape: [table.paddedVocab, table.headDim],
      quantizable: false,
    })],
    ...shapeFlow(shapes.tokenIds, shapes.hidden),
  }), dims.tokenIds, outputDims);
}
