import { decoderStackNetwork } from "../layers/decoderStack.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { projectorModule } from "../layers/projector.js";
import { visionTowerModule } from "../layers/vision.js";
import { networkSpec } from "./common.js";
import { rmsNormModule } from "../layers/norm.js";
import { hfLayersAttr } from "../archs/index.js";

export function buildMiniMaxM3Network(resolved, normalized) {
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.architecture, [
    visionTowerModule(normalized),
    projectorModule(normalized),
    decoderStackNetwork(hfLayersAttr(normalized), normalized, {
      attentionKind: "sparse",
      defaultLayerKind: normalized.experts ? "moe" : "dense",
    }),
    // lm_head 之前的 final norm：多模态那条支线此前整片缺失（权重字节恒等式
    // 差 hidden 个参数 = 12,288 字节，2026-09-09 逐层归因抓出）。
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ], { sequence: true });
}
