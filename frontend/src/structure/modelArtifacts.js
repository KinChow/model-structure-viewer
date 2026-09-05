import {
  fetchBuiltinConfigApi,
  fetchLocalConfigApi,
} from "../api/client.js";
import { fetchHfConfigDirect, resolveEndpoint } from "../api/hf.js";
import { fetchCheckpointTruth } from "../cost/weights.js";

export const CHECKPOINT_TRUTH_STATUS = {
  AVAILABLE: "available",
  EMPTY: "empty",
  UNAVAILABLE: "unavailable",
  NOT_REQUESTED: "not-requested",
};

const REMOTE_REQUEST_TIMEOUT_MS = 10000;

function revisionForEndpoint(endpoint, revision) {
  const { defaultRevision } = resolveEndpoint(endpoint);
  return endpoint === "modelscope" && (!revision || revision === "main")
    ? defaultRevision
    : revision || defaultRevision;
}

function remoteEndpoints(endpoint) {
  const primary = endpoint || "huggingface";
  return primary === "huggingface" ? ["huggingface", "modelscope"] : [primary];
}

function statusForTruth(truth) {
  if (truth?.tensors?.length > 0) return CHECKPOINT_TRUTH_STATUS.AVAILABLE;
  if (truth) return CHECKPOINT_TRUTH_STATUS.EMPTY;
  return CHECKPOINT_TRUTH_STATUS.UNAVAILABLE;
}

function statusForProvidedTruth(truth) {
  if (truth == null) return CHECKPOINT_TRUTH_STATUS.NOT_REQUESTED;
  return statusForTruth(truth);
}

function withTimeout(promise, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("checkpoint metadata request timed out")), timeoutMs);
    Promise.resolve(promise).then(resolve, reject).finally(() => clearTimeout(timer));
  });
}

export function createModelArtifacts({
  config,
  modelId = null,
  revision = null,
  source = "config",
  checkpointTruth = null,
  checkpointTruthStatus = statusForProvidedTruth(checkpointTruth),
  checkpointTruthError = null,
  configEndpoint = null,
  checkpointTruthEndpoint = null,
}) {
  if (!config || typeof config !== "object") throw new Error("Model artifacts require a config object");
  return {
    config,
    modelId,
    revision,
    source: typeof source === "string" ? { kind: source } : source,
    checkpointTruth,
    checkpointTruthStatus,
    checkpointTruthError,
    configEndpoint,
    checkpointTruthEndpoint,
  };
}

async function loadRemoteTruth({ modelId, endpoint, revision, fetchTruth }) {
  if (!modelId) {
    return { truth: null, status: CHECKPOINT_TRUTH_STATUS.NOT_REQUESTED, error: null };
  }
  const { hubUrl, resolvePrefix } = resolveEndpoint(endpoint);
  try {
    const truth = await withTimeout(
      fetchTruth({ modelId, revision, hubUrl, resolvePrefix }),
      REMOTE_REQUEST_TIMEOUT_MS,
    );
    return { truth, status: statusForTruth(truth), error: null };
  } catch (error) {
    return {
      truth: null,
      status: CHECKPOINT_TRUTH_STATUS.UNAVAILABLE,
      error: error?.message || String(error),
    };
  }
}

async function withRemoteTruth(data, { modelId, endpoint, revision, fetchTruth, configEndpoint = endpoint }) {
  let checkpoint = null;
  let checkpointEndpoint = null;
  let lastError = null;
  for (const candidate of remoteEndpoints(endpoint)) {
    const candidateRevision = revisionForEndpoint(candidate, revision);
    const result = await loadRemoteTruth({ modelId, endpoint: candidate, revision: candidateRevision, fetchTruth });
    if (result.status === CHECKPOINT_TRUTH_STATUS.AVAILABLE) {
      checkpoint = result;
      checkpointEndpoint = candidate;
      revision = candidateRevision;
      break;
    }
    lastError = result.error;
    if (!checkpoint || (checkpoint.status === CHECKPOINT_TRUTH_STATUS.UNAVAILABLE && result.status === CHECKPOINT_TRUTH_STATUS.EMPTY)) {
      checkpoint = result;
    }
  }
  if (lastError && checkpoint?.status === CHECKPOINT_TRUTH_STATUS.EMPTY) {
    checkpoint = { ...checkpoint, status: CHECKPOINT_TRUTH_STATUS.UNAVAILABLE };
  }
  const sourceKind = data.source?.kind || "model config";
  return createModelArtifacts({
    config: data.config,
    modelId,
    revision,
    source: sourceKind,
    checkpointTruth: checkpoint?.truth || null,
    checkpointTruthStatus: checkpoint?.status || CHECKPOINT_TRUTH_STATUS.UNAVAILABLE,
    checkpointTruthError: lastError || checkpoint?.error || null,
    configEndpoint,
    checkpointTruthEndpoint: checkpointEndpoint,
  });
}

async function loadRemoteConfig({ modelId, endpoint, revision, fetchHfConfig }) {
  let lastError = null;
  for (const candidate of remoteEndpoints(endpoint)) {
    const candidateRevision = revisionForEndpoint(candidate, revision);
    try {
      const config = await withTimeout(
        fetchHfConfig({ modelId, revision: candidateRevision, endpoint: candidate }),
        REMOTE_REQUEST_TIMEOUT_MS,
      );
      return { config, endpoint: candidate, revision: candidateRevision };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error(`Unable to fetch model config for ${modelId}`);
}

/**
 * Resolve every supported model entry into one artifact bundle.
 * Source-specific fallback policy stays here; structure construction does not
 * need to know whether the bundle came from HF, a built-in catalog, or local.
 */
export async function loadModelArtifacts(
  payload,
  {
    fetchBuiltinConfig = fetchBuiltinConfigApi,
    fetchHfConfig = fetchHfConfigDirect,
    fetchLocalConfig = fetchLocalConfigApi,
    fetchTruth = fetchCheckpointTruth,
  } = {},
) {
  if (payload.source === "config" && payload.config_json) {
    return createModelArtifacts({
      config: payload.config_json,
      modelId: payload.model_id,
      revision: payload.revision,
      source: payload.source_label || "pasted",
      checkpointTruth: payload.checkpoint_truth || null,
    });
  }

  if ((payload.source === "builtin" || payload.source === "auto") && (payload.builtin_entry || payload.model_id)) {
    try {
      const data = await fetchBuiltinConfig({ entry: payload.builtin_entry, modelId: payload.model_id });
      const modelId = data.model_id || payload.model_id;
      const endpoint = payload.endpoint || "huggingface";
      const revision = revisionForEndpoint(endpoint, payload.revision);
      return withRemoteTruth(data, { modelId, endpoint, revision, fetchTruth, configEndpoint: "built-in" });
    } catch (error) {
      if (payload.source !== "auto") throw error;
    }
  }

  if ((payload.source === "local" || payload.source === "auto") && (payload.config_path || payload.model_id)) {
    try {
      const data = await fetchLocalConfig({ modelId: payload.model_id, configPath: payload.config_path });
      return createModelArtifacts({
        config: data.config,
        modelId: data.model_id || payload.model_id,
        revision: payload.revision,
        source: data.source?.kind || "local config",
        checkpointTruth: data.checkpoint_truth || null,
      });
    } catch (error) {
      if (payload.source !== "auto") throw error;
    }
  }

  if (payload.source === "hf" && payload.model_id) {
    const remoteConfig = await loadRemoteConfig({
      modelId: payload.model_id,
      endpoint: payload.endpoint,
      revision: payload.revision,
      fetchHfConfig,
    });
    return withRemoteTruth(
      {
        config: remoteConfig.config,
        source: { kind: `hf config (${remoteConfig.endpoint})` },
      },
      {
        modelId: payload.model_id,
        endpoint: remoteConfig.endpoint,
        revision: remoteConfig.revision,
        fetchTruth,
        configEndpoint: remoteConfig.endpoint,
      },
    );
  }

  return null;
}
