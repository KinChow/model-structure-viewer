import { decoderStackNetwork } from "../layers/decoderStack.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { projectorModule } from "../layers/projector.js";
import { visionTowerModule } from "../layers/vision.js";
import { networkSpec } from "./common.js";

export function buildMiniMaxM3Network(resolved, normalized) {
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.canonicalArchitecture, [
    visionTowerModule(normalized),
    projectorModule(normalized),
    decoderStackNetwork("text_decoder", normalized, {
      attentionKind: "sparse",
      defaultLayerKind: normalized.experts ? "moe" : "dense",
    }),
    lmHeadModule("lm_head", normalized),
  ]);
}
