function countOperatorSpecs(spec) {
  if (!spec) return 0;
  const own = spec.kind === "operator" ? 1 : 0;
  return own + (spec.children || []).reduce((total, child) => total + countOperatorSpecs(child), 0);
}

export function collectDiagnostics({ network, normalized, resolved }) {
  const warnings = [];
  const unsupported = [];

  if (resolved.resolution !== "architecture-alias") {
    warnings.push({
      code: "architecture-inferred",
      message: `Architecture resolved by ${resolved.resolution}`,
    });
  }
  if (!normalized.layers && resolved.canonicalArchitecture !== "generic-config") {
    warnings.push({
      code: "missing-layer-count",
      message: "No text layer count was found in config",
    });
  }
  if (resolved.canonicalArchitecture === "generic-config") {
    unsupported.push({
      code: "generic-config",
      message: "Config does not expose enough fields to build a model network",
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
