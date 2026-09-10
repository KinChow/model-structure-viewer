import { useCallback, useRef, useState } from "react";
import { verifyStructureApi } from "../api/client.js";

export function useVerify() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestRef = useRef(0);

  const verify = useCallback(async (payload) => {
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    setError("");
    setLoading(true);
    try {
      const data = await verifyStructureApi(payload);
      if (requestId !== requestRef.current) return null;
      setResult(data);
      return data;
    } catch (err) {
      if (requestId !== requestRef.current) return null;
      setResult(null);
      setError(err.message);
      return null;
    } finally {
      if (requestId === requestRef.current) setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    requestRef.current += 1;
    setResult(null);
    setError("");
    setLoading(false);
  }, []);

  return { result, loading, error, verify, reset };
}
