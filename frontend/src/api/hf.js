// 前端直连模型源公开 API（静态部署形态下不依赖后端）。
// 所有函数都接受可注入的 fetchImpl，便于测试与后续支持带 token 的 fetch 包装。
//
// endpoint：huggingface（默认）/ modelscope（国内可达镜像，resolve 路径带 /models 前缀，
// 默认 revision 为 master）。ModelScope 的 CORS 与 Range(206) 支持已于 2026-09-04 实测确认。
export const HF_ENDPOINTS = {
  huggingface: { hubUrl: "https://huggingface.co", defaultRevision: "main", resolvePrefix: "" },
  modelscope: { hubUrl: "https://www.modelscope.cn", defaultRevision: "master", resolvePrefix: "/models" },
};

export function resolveEndpoint(endpoint = "huggingface") {
  return HF_ENDPOINTS[endpoint] || HF_ENDPOINTS.huggingface;
}

export async function fetchHfConfigDirect({
  modelId,
  revision,
  endpoint = "huggingface",
  fetchImpl = fetch,
}) {
  const { hubUrl, defaultRevision, resolvePrefix } = resolveEndpoint(endpoint);
  const rev = revision || defaultRevision;
  const url = `${hubUrl}${resolvePrefix}/${modelId}/resolve/${rev}/config.json`;
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`model config HTTP ${res.status}`);
  return res.json();
}

// GET /api/models/{id}?expand[]=safetensors → 逐 dtype 精确参数量（仅 huggingface endpoint 支持）。
export async function fetchHfModelInfoDirect({ modelId, endpoint = "huggingface", fetchImpl = fetch }) {
  if (endpoint !== "huggingface") return null;
  const url = `${HF_ENDPOINTS.huggingface.hubUrl}/api/models/${encodeURIComponent(modelId)}?expand[]=safetensors`;
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`model info HTTP ${res.status}`);
  return res.json();
}

// 搜索结果形态与后端 /api/hf/search 保持一致；仅 huggingface endpoint 支持（modelscope 无公开搜索 API）。
export async function searchHfDirect(query, limit = 10, endpoint = "huggingface", fetchImpl = fetch) {
  if (endpoint !== "huggingface") {
    throw new Error(`search is not supported on endpoint: ${endpoint}`);
  }
  const url = `${HF_ENDPOINTS.huggingface.hubUrl}/api/models?search=${encodeURIComponent(query)}&limit=${limit}`;
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`HF search HTTP ${res.status}`);
  const items = await res.json();
  return (Array.isArray(items) ? items : []).map((m) => ({
    model_id: m.id,
    pipeline_tag: m.pipeline_tag ?? null,
    tags: Array.isArray(m.tags) ? m.tags : [],
    downloads: m.downloads ?? null,
    likes: m.likes ?? null,
  }));
}
