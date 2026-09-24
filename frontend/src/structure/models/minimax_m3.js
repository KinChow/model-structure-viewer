// 对标 vLLM MiniMax-M3 VL
import { decoderStackNetwork } from "../layers/decoderStack.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { multimodalEntry } from "../layers/multimodalEntry.js";
import { rmsNormModule } from "../layers/norm.js";
import { hfLayersAttr } from "../archs/index.js";
import { networkSpecWithDraft } from "./common.js";
import { deepSeekMtpChild } from "./deepseek_mtp.js";

export function assembleMiniMaxM3(resolved, normalized) {
  const draft = deepSeekMtpChild(normalized);
  const entry = multimodalEntry(normalized);
  const children = [
    ...entry.children,
    decoderStackNetwork(hfLayersAttr(normalized), normalized),
    ...(draft ? [draft] : []),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ];
  return networkSpecWithDraft("model", resolved.architecture || normalized.modelType || "Model", resolved.architecture, children, draft, {}, entry);
}
