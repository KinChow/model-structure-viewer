import { useCallback, useState } from "react";
import {
  buildStructureApi,
  fetchBuiltinConfigApi,
  fetchHfConfigApi,
  fetchLocalConfigApi,
} from "../api/client.js";
import { resolveEndpoint } from "../api/hf.js";
import { buildStructureFromConfig } from "../structure/buildStructure.js";
import { fetchCheckpointTruth } from "../cost/weights.js";

export async function buildStructureForPayload(
  payload,
  buildApi = buildStructureApi,
  fetchLocalConfig = fetchLocalConfigApi,
  fetchHfConfig = fetchHfConfigApi,
  fetchBuiltinConfig = fetchBuiltinConfigApi,
  fetchTruth = fetchCheckpointTruth,
) {
  if (payload.source === "config" && payload.config_json) {
    return buildStructureFromConfig(payload.config_json, {
      modelId: payload.model_id,
      revision: payload.revision,
      source: "pasted",
    });
  }
  if ((payload.source === "builtin" || payload.source === "auto") && (payload.builtin_entry || payload.model_id)) {
    try {
      const data = await fetchBuiltinConfig({
        entry: payload.builtin_entry,
        modelId: payload.model_id,
      });
      return buildStructureFromConfig(data.config, {
        modelId: data.model_id || payload.model_id,
        revision: payload.revision,
        source: data.source?.kind || "built-in config",
      });
    } catch (err) {
      if (payload.source !== "auto") throw err;
    }
  }
  if ((payload.source === "local" || payload.source === "auto") && (payload.config_path || payload.model_id)) {
    try {
      const data = await fetchLocalConfig({
        modelId: payload.model_id,
        configPath: payload.config_path,
      });
      return buildStructureFromConfig(data.config, {
        modelId: data.model_id || payload.model_id,
        revision: payload.revision,
        source: data.source?.kind || "local config",
      });
    } catch (err) {
      if (payload.source !== "auto") throw err;
    }
  }
  if (payload.source === "hf" && payload.model_id) {
    const { hubUrl, defaultRevision } = resolveEndpoint(payload.endpoint);
    // modelscope 默认 master；用户显式改过（非 main）时尊重用户值
    const revision =
      payload.endpoint === "modelscope" && (!payload.revision || payload.revision === "main")
        ? "master"
        : payload.revision || defaultRevision;
    const config = await fetchHfConfig({
      modelId: payload.model_id,
      revision,
      endpoint: payload.endpoint,
    });
    // 真值（safetensors header）失败不阻断：离线/gated/无 safetensors → 降级模板路径
    let truth = null;
    try {
      truth = await fetchTruth({ modelId: payload.model_id, revision, hubUrl });
    } catch {
      truth = null;
    }
    return buildStructureFromConfig(config, {
      modelId: payload.model_id,
      revision,
      source: `hf config (${payload.endpoint || "huggingface"})`,
      truth,
    });
  }
  return buildApi(payload);
}

export function useStructure() {
  const [structure, setStructure] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const build = useCallback(async (payload) => {
    setError("");
    setLoading(true);
    try {
      const data = await buildStructureForPayload(payload);
      setStructure(data);
      return data;
    } catch (err) {
      setStructure(null);
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { structure, build, loading, error };
}
