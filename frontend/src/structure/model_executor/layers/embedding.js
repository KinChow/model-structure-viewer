import { moduleSpec, withShapeDims } from "./base.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function embeddingModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(id, "embed tokens", "embedding", {
    class: "Embedding",
    hidden_size: normalized.hiddenSize,
    vocab_size: normalized.vocabSize,
    ...shapeFlow(shapes.tokenIds, shapes.hidden),
  }), dims.tokenIds, dims.hidden);
}
