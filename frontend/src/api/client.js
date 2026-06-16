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

export function searchHfApi(query, limit = 10) {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return requestJson(`/api/hf/search?${params.toString()}`);
}

export function buildStructureApi(payload) {
  return requestJson("/api/structure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function exportStructureApi(structure, format) {
  const response = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ structure, format }),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(text);
  return text;
}
