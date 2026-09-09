import { decoderStackNetwork } from "../layers/decoderStack.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { projectorModule } from "../layers/projector.js";
import { visionTowerModule } from "../layers/vision.js";
import { networkSpec } from "./common.js";
import { rmsNormModule } from "../layers/norm.js";

export function buildMiniMaxM3Network(resolved, normalized) {
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.canonicalArchitecture, [
    visionTowerModule(normalized),
    projectorModule(normalized),
    decoderStackNetwork("text_decoder", normalized, {
      attentionKind: "sparse",
      defaultLayerKind: normalized.experts ? "moe" : "dense",
    }),
    // lm_head 之前的 final norm：多模态那条支线此前整片缺失（权重字节恒等式
    // 差 hidden 个参数 = 12,288 字节，2026-09-09 逐层归因抓出）。
    rmsNormModule("norm", "final norm", normalized, "output_norm"),
    lmHeadModule("lm_head", normalized),
  ], { sequence: true });
}
