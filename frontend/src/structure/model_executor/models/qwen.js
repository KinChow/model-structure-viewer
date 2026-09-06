import { decoderStackNetwork } from "../layers/decoderStack.js";
import { embeddingModule } from "../layers/embedding.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { projectorModule } from "../layers/projector.js";
import { visionTowerModule } from "../layers/vision.js";
import { textDecoderNetwork } from "./common.js";
import { networkSpec } from "./common.js";

export function buildGqaDecoderNetwork(resolved, normalized) {
  return textDecoderNetwork(resolved, normalized, {
    attentionKind: "gqa",
    defaultLayerKind: "dense",
  });
}

export function buildGqaMoeDecoderNetwork(resolved, normalized) {
  return textDecoderNetwork(resolved, normalized, {
    attentionKind: "gqa",
    defaultLayerKind: "moe",
  });
}

export function buildQwenMultimodalNetwork(resolved, normalized) {
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.canonicalArchitecture, [
    visionTowerModule(normalized),
    projectorModule(normalized),
    embeddingModule("embed_tokens", normalized),
    decoderStackNetwork("decoder", normalized, {
      attentionKind: "gqa",
      defaultLayerKind: normalized.experts ? "moe" : "dense",
    }),
    lmHeadModule("lm_head", normalized),
  ]);
}
