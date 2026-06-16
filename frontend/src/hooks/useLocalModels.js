import { useCallback, useEffect, useState } from "react";
import { fetchModels } from "../api/client";

export function useLocalModels() {
  const [models, setModels] = useState([]);

  const refresh = useCallback(async () => {
    try {
      setModels(await fetchModels());
    } catch {
      setModels([]);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { models, refresh };
}
