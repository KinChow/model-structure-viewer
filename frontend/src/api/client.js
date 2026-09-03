import { catalogPath, modelConfigPath, normalizeCatalog } from "../structure/catalog/manifest.js";
import { fetchHfConfigDirect, searchHfDirect } from "./hf.js";

export async function requestJson(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(payload?.detail || `HTTP ${response.status}`);
  }
  return payload;
}

export function fetchSettings() {
  return requestJson("/api/settings");
}

export function saveSettingsApi(settings) {
  return requestJson("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
}

export function fetchModels() {
  return requestJson("/api/models");
}

export async function fetchBuiltinCatalogApi() {
  return normalizeCatalog(await requestJson(catalogPath()));
}

export async function fetchBuiltinConfigApi({ entry, modelId }) {
  const target = entry || (await findBuiltinModelEntry(modelId));
  if (!target) throw new Error(`Built-in model not found: ${modelId}`);
  return {
    model_id: target.modelId,
    config: await requestJson(modelConfigPath(target)),
    source: {
      kind: "built-in config",
      model_id: target.modelId,
      config_path: target.configPath,
    },
  };
}

async function findBuiltinModelEntry(modelId) {
  if (!modelId) return null;
  const catalog = await fetchBuiltinCatalogApi();
  return catalog.models.find((entry) => entry.modelId === modelId) || null;
}

export function fetchLocalConfigApi({ modelId, configPath, source = "local" }) {
  const params = new URLSearchParams();
  if (modelId) params.set("model_id", modelId);
  if (configPath) params.set("config_path", configPath);
  if (source !== "local") params.set("source", source);
  return requestJson(`/api/local/config?${params.toString()}`);
}

// 搜索/配置读取：前端直连 HF 优先（静态部署可用），失败回退后端代理（本地开发/受限网络）。
export async function searchHfApi(query, limit = 10) {
  try {
    return await searchHfDirect(query, limit);
  } catch {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    return requestJson(`/api/hf/search?${params.toString()}`);
  }
}

export async function fetchHfConfigApi({ modelId, revision = "main" }) {
  try {
    return await fetchHfConfigDirect({ modelId, revision });
  } catch {
    const params = new URLSearchParams({ model_id: modelId, revision });
    return requestJson(`/api/hf/config?${params.toString()}`);
  }
}

export function buildStructureApi(payload) {
  return requestJson("/api/structure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
