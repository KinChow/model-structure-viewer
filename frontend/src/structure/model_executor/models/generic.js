import { decoderStackNetwork } from "../layers/decoderStack.js";
import { networkSpec } from "./common.js";

export function buildGenericDecoderNetwork(resolved, normalized) {
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.canonicalArchitecture, [
    decoderStackNetwork("decoder", normalized, { attentionKind: "gqa", defaultLayerKind: "dense" }),
  ]);
}

export function buildGenericConfigNetwork(resolved, normalized) {
  return networkSpec(
    "model",
    resolved.architecture || normalized.modelType || "Configuration",
    resolved.canonicalArchitecture,
    [],
  );
}
