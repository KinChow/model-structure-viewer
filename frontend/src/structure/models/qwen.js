import { decoderStackNetwork } from "../layers/decoderStack.js";
import { embeddingModule } from "../layers/embedding.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { mtpChild } from "../layers/mtp.js";
import { projectorModule } from "../layers/projector.js";
import { visionTowerModule } from "../layers/vision.js";
import { textDecoderNetwork } from "./common.js";
import { networkSpec } from "./common.js";
import { outputAttentionResidualModule } from "../layers/residual.js";
import { rmsNormModule } from "../layers/norm.js";
import { hyperConnectionModule } from "../layers/hybrid.js";
import { hfLayersAttr, recipeVisionInternalMerger } from "../archs/index.js";

export function buildGqaDecoderNetwork(resolved, normalized) {
  return textDecoderNetwork(resolved, normalized, {
    defaultLayerKind: "dense",
  });
}

export function buildGqaMoeDecoderNetwork(resolved, normalized) {
  return textDecoderNetwork(resolved, normalized, {
    defaultLayerKind: "moe",
  });
}

export function buildQwenMultimodalNetwork(resolved, normalized) {
  return buildMultimodalDecoderNetwork(resolved, normalized, {
    defaultLayerKind: normalized.experts ? "moe" : "dense",
  });
}

export function buildMlaMultimodalNetwork(resolved, normalized) {
  return buildMultimodalDecoderNetwork(resolved, normalized, {
    attentionKind: "mla",
    defaultLayerKind: "moe",
  });
}

function buildMultimodalDecoderNetwork(resolved, normalized, { attentionKind, defaultLayerKind }) {
  const draft = mtpChild(normalized);
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.architecture, [
    visionTowerModule(normalized),
    ...(normalized.hasVisionProjector && !recipeVisionInternalMerger(normalized) ? [projectorModule(normalized)] : []),
    embeddingModule("embed_tokens", normalized),
    decoderStackNetwork(hfLayersAttr(normalized), normalized, {
      attentionKind,
      defaultLayerKind,
    }),
    ...(draft ? [draft] : []),
    ...(normalized.hyperConnectionCount ? [hyperConnectionModule("hyper_connection_mixer", normalized, "final")] : []),
    ...(normalized.attnResBlockSize ? [outputAttentionResidualModule("output_attn_residual", normalized)] : []),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ], { sequence: true });
}


