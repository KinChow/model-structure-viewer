import { collectDiagnostics } from "../diagnostics/collectDiagnostics.js";

export function createStructureIr({ network, normalized, resolved, options = {} }) {
  return {
    version: 2,
    strategy: "frontend-architecture-template",
    network,
    normalized,
    resolved,
    options,
    diagnostics: collectDiagnostics({ network, normalized, resolved }),
  };
}
