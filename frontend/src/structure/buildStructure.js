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

export function buildStructureFromArtifacts(artifacts) {
  return buildStructureFromConfig(artifacts.config, {
    modelId: artifacts.modelId,
    revision: artifacts.revision,
    source: artifacts.source?.kind || "model config",
    truth: artifacts.checkpointTruth,
    checkpointTruthStatus: artifacts.checkpointTruthStatus,
    checkpointTruthMethod: artifacts.checkpointTruth?.method || null,
    checkpointTruthError: artifacts.checkpointTruthError,
    configEndpoint: artifacts.configEndpoint,
    checkpointTruthEndpoint: artifacts.checkpointTruthEndpoint,
  });
}
