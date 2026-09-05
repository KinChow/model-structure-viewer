import { useCallback, useRef, useState } from "react";
import { buildStructureApi } from "../api/client.js";
import { buildStructureFromArtifacts } from "../structure/buildStructure.js";
import { loadModelArtifacts, resolveDeferredCheckpointTruth } from "../structure/modelArtifacts.js";

export async function buildStructureForPayload(
  payload,
  buildApi = buildStructureApi,
  fetchLocalConfig,
  fetchHfConfig,
  fetchBuiltinConfig,
  fetchTruth,
  onBackgroundUpdate,
) {
  const artifacts = await loadModelArtifacts(payload, {
    fetchLocalConfig,
    fetchHfConfig,
    fetchBuiltinConfig,
    fetchTruth,
    deferCheckpointTruth: Boolean(onBackgroundUpdate && (payload.source === "builtin" || payload.source === "auto")),
  });
  if (artifacts) {
    const structure = buildStructureFromArtifacts(artifacts);
    if (artifacts.deferredTruth) {
      void resolveDeferredCheckpointTruth(artifacts, { fetchTruth }).then((updatedArtifacts) => {
        onBackgroundUpdate?.(buildStructureFromArtifacts(updatedArtifacts));
      });
    }
    return structure;
  }
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
      const data = await buildStructureForPayload(
        payload,
        buildStructureApi,
        undefined,
        undefined,
        undefined,
        undefined,
        (updated) => {
          if (requestId === requestRef.current) setStructure(updated);
        },
      );
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
