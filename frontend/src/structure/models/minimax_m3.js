// 对标 vLLM MiniMax-M3 VL
import { decoderStackNetwork } from "../layers/decoderStack.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { projectorModule } from "../layers/projector.js";
import { visionTowerModule } from "../layers/vision.js";
import { rmsNormModule } from "../layers/norm.js";
import { hfLayersAttr } from "../archs/index.js";
import { networkSpec } from "./common.js";
import { deepSeekMtpChild } from "./deepseek_mtp.js";

export function assembleMiniMaxM3(resolved, normalized) {
  const draft = deepSeekMtpChild(normalized);
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.architecture, [
    visionTowerModule(normalized),
    projectorModule(normalized),
    decoderStackNetwork(hfLayersAttr(normalized), normalized, {
      attentionKind: "sparse",
      defaultLayerKind: normalized.experts ? "moe" : "dense",
    }),
    ...(draft ? [draft] : []),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ], { sequence: true });
}
