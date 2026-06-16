import { useCallback, useEffect, useState } from "react";
import { fetchSettings, saveSettingsApi } from "../api/client";

export const DEFAULT_SETTINGS = {
  model_root: "/Users/zhouzijian01/Desktop/workspace/models",
  hf_endpoint: "https://huggingface.co",
  cache_policy: "prefer-local",
  offline: false,
};

function pickFields(data) {
  return {
    model_root: data.model_root,
    hf_endpoint: data.hf_endpoint,
    cache_policy: data.cache_policy,
    offline: data.offline,
  };
}

export function useSettings() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchSettings()
      .then((data) => setSettings(pickFields(data)))
      .catch(() => setSettings(DEFAULT_SETTINGS));
  }, []);

  const save = useCallback(async () => {
    setError("");
    try {
      const data = await saveSettingsApi(settings);
      setSettings(pickFields(data));
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    }
  }, [settings]);

  return { settings, setSettings, save, error };
}
