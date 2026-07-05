import { decoderStackNetwork } from "../layers/decoderStack.js";
import { embeddingModule } from "../layers/embedding.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { rmsNormModule } from "../layers/norm.js";

export function networkSpec(id, name, canonicalArchitecture, children) {
  return {
    kind: "network",
    id,
    name,
    canonicalArchitecture,
    children,
  };
}

export function textDecoderNetwork(resolved, normalized, { attentionKind, defaultLayerKind }) {
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.canonicalArchitecture, [
    embeddingModule("embed_tokens", normalized),
    decoderStackNetwork("decoder", normalized, { attentionKind, defaultLayerKind }),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ]);
}
