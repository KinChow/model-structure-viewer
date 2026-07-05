import { normalizeConfig } from "./config/normalize.js";
import { resolveArchitecture } from "./registry/resolveArchitecture.js";
import { buildNetwork } from "./model_executor/models/index.js";
import { createStructureIr } from "./ir/createStructureIr.js";
import { materializeModelStructure } from "./materializers/toStructureNode.js";

export function buildStructureFromConfig(config, options = {}) {
  const normalized = normalizeConfig(config);
  const resolved = resolveArchitecture(normalized, options);
  const network = buildNetwork(resolved, normalized);
  const ir = createStructureIr({ network, normalized, resolved, options });
  return materializeModelStructure(ir);
}
