import { moduleSpec } from "./base.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function embeddingModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  return moduleSpec(id, "embed tokens", "embedding", {
    class: "Embedding",
    hidden_size: normalized.hiddenSize,
    vocab_size: normalized.vocabSize,
    ...shapeFlow(shapes.tokenIds, shapes.hidden),
  });
}
