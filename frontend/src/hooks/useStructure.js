import { useCallback, useState } from "react";
import {
  buildStructureApi,
  fetchBuiltinConfigApi,
  fetchHfConfigApi,
  fetchLocalConfigApi,
} from "../api/client.js";
import { buildStructureFromConfig } from "../structure/buildStructure.js";

export async function buildStructureForPayload(
  payload,
  buildApi = buildStructureApi,
  fetchLocalConfig = fetchLocalConfigApi,
  fetchHfConfig = fetchHfConfigApi,
  fetchBuiltinConfig = fetchBuiltinConfigApi,
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
    const config = await fetchHfConfig({
      modelId: payload.model_id,
      revision: payload.revision || "main",
    });
    return buildStructureFromConfig(config, {
      modelId: payload.model_id,
      revision: payload.revision,
      source: "hf config",
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
