import { moduleSpec } from "./base.js";

export function embeddingModule(id, normalized) {
  return moduleSpec(id, "embed tokens", "embedding", {
    class: "Embedding",
    hidden_size: normalized.hiddenSize,
  });
}
