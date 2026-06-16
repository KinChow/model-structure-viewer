import { useCallback, useState } from "react";
import { buildStructureApi } from "../api/client";

export function useStructure() {
  const [structure, setStructure] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const build = useCallback(async (payload) => {
    setError("");
    setLoading(true);
    try {
      const data = await buildStructureApi(payload);
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
