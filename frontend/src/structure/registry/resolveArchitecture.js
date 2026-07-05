import { ARCHITECTURE_ALIASES } from "./aliases.js";

export function resolveArchitecture(normalized, options = {}) {
  if (normalized.architecture && ARCHITECTURE_ALIASES[normalized.architecture]) {
    return {
      canonicalArchitecture: ARCHITECTURE_ALIASES[normalized.architecture],
      architecture: normalized.architecture,
      resolution: "architecture-alias",
    };
  }

  const probe = `${normalized.architecture || ""} ${normalized.modelType || ""} ${options.modelId || ""}`.toLowerCase();
  if (probe.includes("minimax")) {
    return { canonicalArchitecture: "multimodal-sparse-moe-decoder", architecture: normalized.architecture, resolution: "model-type" };
  }
  if (probe.includes("deepseek") || probe.includes("glm_moe_dsa") || probe.includes("glmmoedsa")) {
    return { canonicalArchitecture: "mla-moe-decoder", architecture: normalized.architecture, resolution: "model-type" };
  }
  if (probe.includes("qwen") && normalized.experts) {
    return { canonicalArchitecture: "gqa-moe-decoder", architecture: normalized.architecture, resolution: "model-type" };
  }
  if (probe.includes("qwen")) {
    return { canonicalArchitecture: "gqa-decoder", architecture: normalized.architecture, resolution: "model-type" };
  }
  if (normalized.layers) {
    return { canonicalArchitecture: "generic-decoder", architecture: normalized.architecture, resolution: "field-inference" };
  }
  return { canonicalArchitecture: "generic-config", architecture: normalized.architecture, resolution: "generic-config" };
}
