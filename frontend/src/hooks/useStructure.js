import { useCallback, useRef, useState } from "react";
import { buildStructureApi } from "../api/client.js";
import { buildStructureFromArtifacts } from "../structure/buildStructure.js";
import { loadModelArtifacts } from "../structure/modelArtifacts.js";

export async function buildStructureForPayload(
  payload,
  buildApi = buildStructureApi,
  fetchLocalConfig,
  fetchHfConfig,
  fetchBuiltinConfig,
  fetchTruth,
) {
  const artifacts = await loadModelArtifacts(payload, {
    fetchLocalConfig,
    fetchHfConfig,
    fetchBuiltinConfig,
    fetchTruth,
  });
  if (artifacts) return buildStructureFromArtifacts(artifacts);
  return buildApi(payload);
}

export function useStructure() {
  const [structure, setStructure] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestRef = useRef(0);

  const build = useCallback(async (payload) => {
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    setError("");
    setLoading(true);
    try {
      const data = await buildStructureForPayload(payload);
      if (requestId !== requestRef.current) return null;
      setStructure(data);
      return data;
    } catch (err) {
      if (requestId !== requestRef.current) return null;
      setStructure(null);
      setError(err.message);
      return null;
    } finally {
      if (requestId === requestRef.current) setLoading(false);
    }
  }, []);

  return { structure, build, loading, error };
}
