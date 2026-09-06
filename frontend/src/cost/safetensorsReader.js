// safetensorsReader.js —— 浏览器跨域安全的最小 safetensors header 读取器。
//
// 为什么需要（2026-09-04 实测）：@huggingface/hub 的 parseSafetensorsMetadata 会从响应头读取
// content-range / etag / x-linked-etag（见其 fileDownloadInfo），而 ModelScope CDN 的 206 响应
// 没有 Access-Control-Expose-Headers → 浏览器跨域时这些头读不到 → 库硬抛错。
// 本读取器只读响应体（header 的纯 JSON），跨域仅需 Access-Control-Allow-Origin，不需要 expose-headers。
//
// 边界：这里只解析 header 结构（dtype/shape，格式即"8 字节小端 u64 长度 + 纯 JSON"，§11.5），
// 不做子字节量化的打包换算——parameterCount 按 Σnumel 计（未量化模型即精确值）。
// 模型级精确参数量优先走 parseSafetensorsMetadata（weights.js 库优先）；本路径为浏览器降级。

import { normalizeModelId } from "../api/hf.js";

// 与 @huggingface/hub parseSafetensorsMetadata 保持相同上限，避免畸形文件
// 诱导浏览器申请无界内存。
const MAX_HEADER_LENGTH = 25_000_000;

/**
 * 读取模型 repo 的全部 safetensors header，归一化为 {tensors, parameterCount, parameterTotal}。
 * @param {{modelId: string, revision?: string, hubUrl?: string, resolvePrefix?: string, fetchImpl?: typeof fetch}} params
 */
export async function readSafetensorsHeaders({
  modelId,
  revision = "main",
  hubUrl = "https://huggingface.co",
  resolvePrefix = "",
  fetchImpl = fetch,
}) {
  const normalizedModelId = normalizeModelId(modelId, resolvePrefix === "/models" ? "modelscope" : "huggingface");
  const base = `${hubUrl}${resolvePrefix}/${normalizedModelId}/resolve/${encodeURIComponent(revision)}`;

  // 分片 index 存在 → 按 index 的 shard 读；否则单文件
  const indexUrl = `${base}/model.safetensors.index.json`;
  let shardFiles = null;
  try {
    const res = await fetchImpl(indexUrl);
    if (res.ok) {
      const index = await res.json();
      shardFiles = [...new Set(Object.values(index.weight_map || {}))];
    }
  } catch {
    // 404/网络异常 → 按单文件处理
  }
  if (!shardFiles || shardFiles.length === 0) shardFiles = ["model.safetensors"];

  const headers = [];
  for (const shard of shardFiles) headers.push(await readShardHeader(`${base}/${shard}`, fetchImpl));
  return summarizeHeaders(headers);
}

/** Read model safetensors headers already selected by a browser directory picker. */
export async function readLocalSafetensorsHeaders(files = []) {
  const entries = [...files].filter((file) => /(?:^|\/)\w*(?:model|pytorch_model)[^/]*\.safetensors$/i.test(file.name || ""));
  if (entries.length === 0) return null;

  const indexFile = [...files].find((file) => file.name === "model.safetensors.index.json");
  let selected = entries;
  if (indexFile) {
    try {
      const index = JSON.parse(await indexFile.text());
      const shardNames = new Set(Object.values(index.weight_map || {}));
      const indexed = entries.filter((file) => shardNames.has(file.name) || [...shardNames].some((name) => String(file.name).endsWith(`/${name}`)));
      if (indexed.length > 0) selected = indexed;
    } catch {
      // A malformed optional index should not prevent reading the shard headers.
    }
  }
  const headers = [];
  for (const file of selected) headers.push(await readLocalShardHeader(file));
  return summarizeHeaders(headers);
}

function summarizeHeaders(headers) {
  const tensors = [];
  for (const header of headers) {
    for (const [name, info] of Object.entries(header)) {
      if (name.startsWith("__") || name.startsWith("bitsandbytes__")) continue;
      tensors.push({ name, dtype: info.dtype, shape: info.shape });
    }
  }
  tensors.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const parameterCount = {};
  for (const t of tensors) {
    const numel = t.shape.reduce((a, b) => a * b, 1);
    parameterCount[t.dtype] = (parameterCount[t.dtype] || 0) + numel;
  }
  return { tensors, parameterCount, parameterTotal: Object.values(parameterCount).reduce((a, b) => a + b, 0) };
}

/** 读取单个分片：先取 8 字节 header 长度，再取该长度 JSON。 */
async function readShardHeader(shardUrl, fetchImpl) {
  const lenBuf = await rangeBytes(fetchImpl, shardUrl, 0, 7);
  const headerLen = readU64LE(lenBuf);
  const jsonBytes = await rangeBytes(fetchImpl, shardUrl, 8, 8 + headerLen - 1);
  return JSON.parse(new TextDecoder().decode(jsonBytes));
}

async function readLocalShardHeader(file) {
  const lenBuf = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  const headerLen = readU64LE(lenBuf);
  const jsonBytes = new Uint8Array(await file.slice(8, 8 + headerLen).arrayBuffer());
  return JSON.parse(new TextDecoder().decode(jsonBytes));
}

async function rangeBytes(fetchImpl, url, start, end) {
  const res = await fetchImpl(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (!res.ok) throw new Error(`safetensors range HTTP ${res.status}`);
  if (res.status !== 206) {
    await res.body?.cancel?.();
    throw new Error("safetensors server does not support byte ranges");
  }
  const buf = new Uint8Array(await res.arrayBuffer());
  if (buf.length < end - start + 1) throw new Error("safetensors range response is truncated");
  return buf.slice(0, end - start + 1);
}

function readU64LE(buf) {
  if (buf.length < 8) throw new Error("safetensors header length too short");
  let n = 0n;
  for (let i = 7; i >= 0; i--) n = (n << 8n) | BigInt(buf[i]);
  if (n > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("safetensors header length is unsafe");
  const value = Number(n);
  validateHeaderLength(value);
  return value;
}

function validateHeaderLength(value) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error("safetensors header length is invalid");
  if (value > MAX_HEADER_LENGTH) {
    throw new Error(`safetensors header exceeds ${MAX_HEADER_LENGTH} bytes`);
  }
}
