import { useCallback, useState } from "react";
import { searchHfApi } from "../api/client";

export function useHfSearch(initialQuery = "DeepSeek-V3.1") {
  const [query, setQuery] = useState(initialQuery);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const search = useCallback(async () => {
    if (!query.trim()) return;
    setError("");
    setLoading(true);
    try {
      setResults(await searchHfApi(query.trim()));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [query]);

  return { query, setQuery, results, search, loading, error };
}
