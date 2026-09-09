import { SUPPORTED_MODEL_ARCHITECTURES } from "../model_executor/models/index.js";

function countOperatorSpecs(spec) {
  if (!spec) return 0;
  const own = spec.kind === "operator" ? 1 : 0;
  return own + (spec.children || []).reduce((total, child) => total + countOperatorSpecs(child), 0);
}

export function collectDiagnostics({ network, normalized, resolved }) {
  const warnings = [];
  const unsupported = [];

  // 步骤 2（结构正确性收口）：resolution 只剩 architecture-alias | unsupported。
  // 未知架构不再走 architecture-inferred 警告通道，统一进 unsupported。
  if (resolved.resolution === "architecture-alias" && !normalized.layers) {
    warnings.push({
      code: "missing-layer-count",
      message: "No text layer count was found in config",
    });
  }
  if (resolved.canonicalArchitecture === "unsupported") {
    // vLLM _raise_for_unsupported 模式：不支持即枚举支持项，让用户知道下一步。
    unsupported.push({
      code: "unsupported-architecture",
      message: `Config does not map to a supported architecture template. `
        + `Supported architectures: ${SUPPORTED_MODEL_ARCHITECTURES.join(", ")}`,
    });
  }

  return {
    resolution: resolved.resolution,
    canonical_architecture: resolved.canonicalArchitecture,
    operator_count: countOperatorSpecs(network),
    warnings,
    unsupported,
  };
}
