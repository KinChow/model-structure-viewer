// 前端直连 Hugging Face 公开 API（静态部署形态下不依赖后端）。
// 所有函数都接受可注入的 fetchImpl，便于测试与后续支持带 token 的 fetch 包装。
const HF_ENDPOINT = "https://huggingface.co";

export async function fetchHfConfigDirect({ modelId, revision = "main", fetchImpl = fetch }) {
  const url = `${HF_ENDPOINT}/${modelId}/resolve/${revision}/config.json`;
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`HF config HTTP ${res.status}`);
  return res.json();
}

// GET /api/models/{id}?expand[]=safetensors → 逐 dtype 精确参数量（CORS 回显 Origin，静态站点可直调）。
export async function fetchHfModelInfoDirect({ modelId, fetchImpl = fetch }) {
  const url = `${HF_ENDPOINT}/api/models/${encodeURIComponent(modelId)}?expand[]=safetensors`;
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`HF model info HTTP ${res.status}`);
  return res.json();
}

// 搜索结果形态与后端 /api/hf/search 保持一致，useHfSearch 无需改动。
export async function searchHfDirect(query, limit = 10, fetchImpl = fetch) {
  const url = `${HF_ENDPOINT}/api/models?search=${encodeURIComponent(query)}&limit=${limit}`;
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
