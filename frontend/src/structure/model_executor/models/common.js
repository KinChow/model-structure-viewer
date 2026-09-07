import { decoderStackNetwork } from "../layers/decoderStack.js";
import { embeddingModule } from "../layers/embedding.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { rmsNormModule } from "../layers/norm.js";
import { outputAttentionResidualModule } from "../layers/residual.js";

export function networkSpec(id, name, canonicalArchitecture, children, attributes = {}) {
  return {
    kind: "network",
    id,
    name,
    canonicalArchitecture,
    attributes,
    children,
  };
}

export function textDecoderNetwork(resolved, normalized, { attentionKind, defaultLayerKind }) {
  // §2.1：网络级子节点（embed → decoder → norm → lm_head）显式声明顺序执行
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.canonicalArchitecture, [
    embeddingModule("embed_tokens", normalized),
    decoderStackNetwork("decoder", normalized, { attentionKind, defaultLayerKind }),
    ...(normalized.attnResBlockSize ? [outputAttentionResidualModule("output_attn_residual", normalized)] : []),
    rmsNormModule("norm", "final norm", normalized, "output_norm"),
    lmHeadModule("lm_head", normalized),
  ], { sequence: true });
}
