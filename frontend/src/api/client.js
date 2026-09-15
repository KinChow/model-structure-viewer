import { catalogPath, modelConfigPath, normalizeCatalog, staticAssetPath } from "../structure/catalog/manifest.js";
import { issueError } from "../i18n/format.js";

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
    const issue = describeHttpError(response.status, payload);
    const error = issueError(issue.code, issue.params);
    error.status = response.status;
    error.path = path;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function describeHttpError(status, payload) {
  if (payload?.detail) {
    return {
      code: "http.detail",
      params: { detail: typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail) },
    };
  }
  return { code: "http.status", params: { status } };
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
 * 离线 checkpoint 真值：优先 skeleton-truth.json（折叠树 + 总量），
 * 其次 header-truth.json（只存 parameterTotal，S3 一次性 header 证据）。
 * 两者都由 safetensors header range-read 生成，不下载权重。
 * 文件可选——404 视为该模型未取证，返回 null。
 */
export async function fetchBuiltinSkeletonTruthApi({ entry, modelId }) {
  const target = entry || (await findBuiltinModelEntry(modelId));
  if (!target) return null;
  const truthDir = String(target.configPath).replace(/\\/g, "/").split("/").slice(0, -1).join("/");
  for (const name of ["skeleton-truth.json", "header-truth.json"]) {
    try {
      const payload = await requestJson(staticAssetPath(`models/${truthDir}/${name}`));
      if (payload) return payload;
    } catch {
      // 该 sidecar 缺席，试下一个
    }
  }
  return null;
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
