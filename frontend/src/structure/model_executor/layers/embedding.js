import { moduleSpec, withShapeDims } from "./base.js";
import { weightMatrixDecl } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../../config/dims.js";

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
