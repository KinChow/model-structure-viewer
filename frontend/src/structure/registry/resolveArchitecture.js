import { ARCHITECTURE_ALIASES } from "./aliases.js";
import { multimodalVariant } from "./architectureCatalog.js";

function withVision(normalized, canonicalArchitecture) {
  return normalized.hasVision
    ? multimodalVariant(canonicalArchitecture) || canonicalArchitecture
    : canonicalArchitecture;
}

export function resolveArchitecture(normalized, options = {}) {
  if (normalized.visionConfig && normalized.architecture === "Qwen4ExpForConditionalGeneration") {
    return { canonicalArchitecture: "multimodal-gqa-moe-decoder", architecture: normalized.architecture, resolution: "architecture-alias" };
  }
  if (normalized.architecture && ARCHITECTURE_ALIASES[normalized.architecture]) {
    return {
      canonicalArchitecture: withVision(normalized, ARCHITECTURE_ALIASES[normalized.architecture]),
      architecture: normalized.architecture,
      resolution: "architecture-alias",
    };
  }

  const probe = `${normalized.architecture || ""} ${normalized.modelType || ""} ${options.modelId || ""}`.toLowerCase();
  if (probe.includes("minimax")) {
    return { canonicalArchitecture: "multimodal-sparse-moe-decoder", architecture: normalized.architecture, resolution: "model-type" };
  }
  if (probe.includes("deepseek") || probe.includes("glm_moe_dsa") || probe.includes("glmmoedsa")) {
    return { canonicalArchitecture: withVision(normalized, "mla-moe-decoder"), architecture: normalized.architecture, resolution: "model-type" };
  }
  if (probe.includes("qwen") && normalized.experts) {
    return { canonicalArchitecture: withVision(normalized, "gqa-moe-decoder"), architecture: normalized.architecture, resolution: "model-type" };
  }
  if (probe.includes("qwen")) {
    return { canonicalArchitecture: withVision(normalized, "gqa-decoder"), architecture: normalized.architecture, resolution: "model-type" };
  }
  if (normalized.layers) {
    return { canonicalArchitecture: "generic-decoder", architecture: normalized.architecture, resolution: "field-inference" };
  }
  return { canonicalArchitecture: "generic-config", architecture: normalized.architecture, resolution: "generic-config" };
}
