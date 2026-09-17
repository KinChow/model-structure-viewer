import { SUPPORTED_MODEL_ARCHITECTURES } from "../models/index.js";

function countOperatorSpecs(spec) {
  if (!spec) return 0;
  const own = spec.kind === "operator" ? 1 : 0;
  return own + (spec.children || []).reduce((total, child) => total + countOperatorSpecs(child), 0);
}

export function collectDiagnostics({ network, normalized, resolved }) {
  const warnings = [];
  const unsupported = [];

  if (resolved.resolution === "architecture-alias" && !normalized.layers) {
    warnings.push({
      code: "missing-layer-count",
      // {code, params} 供 UI 按语言渲染（catalog: diag.missingLayerCount）；
      // message 保留为英文回退，不参与分派。
      params: {},
      message: "No text layer count was found in config",
    });
  }
  if (resolved.resolution === "unsupported") {
    // vLLM ModelRegistry._raise_for_unsupported：不支持即枚举支持项。
    unsupported.push({
      code: "unsupported-architecture",
      // catalog: diag.unsupportedArchitecture，{supported} 由 params 注入。
      params: { supported: SUPPORTED_MODEL_ARCHITECTURES.join(", ") },
      message: `Config does not map to a supported architecture. `
        + `Supported architectures: ${SUPPORTED_MODEL_ARCHITECTURES.join(", ")}`,
    });
  }

  return {
    resolution: resolved.resolution,
    architecture: resolved.architecture,
    operator_count: countOperatorSpecs(network),
    warnings,
    unsupported,
  };
}
