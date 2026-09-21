import { useCallback, useRef, useState } from "react";
import { buildStructureFromArtifacts } from "../structure/buildStructure.js";
import { loadModelArtifacts, resolveDeferredCheckpointTruth } from "../model/loadModelArtifacts.js";

export async function buildStructureForPayload(payload, {
  fetchHfConfig,
  fetchBuiltinConfig,
  fetchTruth,
  onBackgroundUpdate,
  onProgress,
  fetchBuiltinSkeletonTruth,
  frameworkProfile,
} = {}) {
  const artifacts = await loadModelArtifacts(payload, {
    fetchHfConfig,
    fetchBuiltinConfig,
    fetchTruth,
    fetchBuiltinSkeletonTruth,
    deferCheckpointTruth: Boolean(onBackgroundUpdate && (payload.source === "builtin" || payload.source === "auto")),
    onProgress,
  });
  onProgress?.("building");
  const structure = buildStructureFromArtifacts(artifacts, { frameworkProfile });
  if (artifacts.deferredTruth) {
    void resolveDeferredCheckpointTruth(artifacts, { fetchTruth }).then((updatedArtifacts) => {
      onBackgroundUpdate?.(buildStructureFromArtifacts(updatedArtifacts, { frameworkProfile }));
    });
  }
  return structure;
}

export function useStructure() {
  const [structure, setStructure] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState("reading");
  const [error, setError] = useState("");
  const requestRef = useRef(0);

  const build = useCallback(async (payload, buildOptions = {}) => {
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    setError("");
    setLoadingPhase("reading");
    setLoading(true);
    try {
      const data = await buildStructureForPayload(payload, {
        frameworkProfile: buildOptions.frameworkProfile,
        onBackgroundUpdate: (updated) => {
          if (requestId === requestRef.current) setStructure(updated);
        },
        onProgress: (phase) => {
          if (requestId === requestRef.current) setLoadingPhase(phase);
        },
      });
      if (requestId !== requestRef.current) return null;
      setStructure(data);
      return data;
    } catch (err) {
      if (requestId !== requestRef.current) return null;
      setStructure(null);
      setError(err.issue || err.message);
      return null;
    } finally {
      if (requestId === requestRef.current) {
        setLoading(false);
        setLoadingPhase("ready");
      }
    }
  }, []);

  return { structure, build, loading, loadingPhase, error };
}
