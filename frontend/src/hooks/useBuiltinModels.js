import { useCallback, useEffect, useState } from "react";
import { fetchBuiltinCatalogApi } from "../api/client";

export function useBuiltinModels() {
  const [models, setModels] = useState([]);

  const refresh = useCallback(async () => {
    try {
      const catalog = await fetchBuiltinCatalogApi();
      setModels(catalog.models);
    } catch {
      setModels([]);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { models, refresh };
}
