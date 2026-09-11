import { catalogPath, modelConfigPath, normalizeCatalog, staticAssetPath } from "../structure/catalog/manifest.js";
import { searchHfDirect } from "./hf.js";

export async function requestJson(path, options) {
  const response = await fetch(path, options);
  let text = "";
  try {
    text = await response.text();
  } catch {
    text = "";
  }
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }
  if (!response.ok) {
    const error = new Error(describeHttpError(response.status, payload, path));
    error.status = response.status;
    error.path = path;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function describeHttpError(status, payload, path) {
  if (payload?.detail) return typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
  if (status >= 500 && !payload) {
    if (path === "/api/verify") {
      return "后端不可用：校验需要本地 Python 服务（默认 :8000）跑 transformers meta 构型。先 `.venv/bin/msv serve --root ./models --port 8000`，再点校验。静态部署没有这条通路。";
    }
    return `HTTP ${status}：后端不可用或返回了非 JSON 错误。请启动后端（uvicorn），或改用 source=hf + endpoint=modelscope 走前端直连（无需后端）。`;
  }
  return `HTTP ${status}`;
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

/**
 * 离线 checkpoint 真值（N2-2）：models/<configPath 目录>/skeleton-truth.json，
 * 由 `node scripts/fetch-evidence.mjs <org>/<id> --headers` 从 safetensors
 * 头部构建后入库。文件是**可选**资产——404 视为该模型未取证，返回 null。
 */
export async function fetchBuiltinSkeletonTruthApi({ entry, modelId }) {
  const target = entry || (await findBuiltinModelEntry(modelId));
  if (!target) return null;
  try {
    const truthDir = String(target.configPath).replace(/\\/g, "/").split("/").slice(0, -1).join("/");
    return await requestJson(staticAssetPath(`models/${truthDir}/skeleton-truth.json`));
  } catch {
    return null;
  }
}

/**
 * §5.3 离线 source_ref：models/<configPath 目录>/source-ref.json，
 * 由 `msv dump-source-ref --model <id>` 从 transformers meta 构型采集。
 * 文件可选——404 视为该模型未采集，返回 null，不编造链接。
 */
export async function fetchBuiltinSourceRefApi({ entry, modelId }) {
  const target = entry || (await findBuiltinModelEntry(modelId));
  if (!target) return null;
  try {
    const refDir = String(target.configPath).replace(/\\/g, "/").split("/").slice(0, -1).join("/");
    return await requestJson(staticAssetPath(`models/${refDir}/source-ref.json`));
  } catch {
    return null;
  }
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

// 搜索/配置读取：前端直连优先（静态部署可用），失败回退后端代理（本地开发/受限网络）。
export async function searchHfApi(query, limit = 10, endpoint) {
  try {
    return await searchHfDirect(query, limit, endpoint);
  } catch {
    const params = new URLSearchParams({ q: query, limit: String(limit), endpoint: endpoint || "huggingface" });
    return requestJson(`/api/hf/search?${params.toString()}`);
  }
}

export function buildStructureApi(payload) {
  return requestJson("/api/structure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function verifyStructureApi(payload) {
  return requestJson("/api/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
