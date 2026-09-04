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
  const base = `${hubUrl}${resolvePrefix}/${modelId}/resolve/${revision}`;

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

  const tensors = [];
  for (const shard of shardFiles) {
    const header = await readShardHeader(`${base}/${shard}`, fetchImpl);
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
  const parameterTotal = Object.values(parameterCount).reduce((a, b) => a + b, 0);

  return { tensors, parameterCount, parameterTotal };
}

/** 读取单个分片：先取 8 字节 header 长度，再取该长度 JSON。 */
async function readShardHeader(shardUrl, fetchImpl) {
  const lenBuf = await rangeBytes(fetchImpl, shardUrl, 0, 7);
  const headerLen = readU64LE(lenBuf);
  const jsonBytes = await rangeBytes(fetchImpl, shardUrl, 8, 8 + headerLen - 1);
  return JSON.parse(new TextDecoder().decode(jsonBytes));
}

async function rangeBytes(fetchImpl, url, start, end) {
  const res = await fetchImpl(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (!res.ok) throw new Error(`safetensors range HTTP ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  // 服务端若不支持 Range 会返回整文件——只取需要的切片，避免意外下载整个权重
  return buf.slice(0, end - start + 1);
}

function readU64LE(buf) {
  if (buf.length < 8) throw new Error("safetensors header length too short");
  let n = 0n;
  for (let i = 7; i >= 0; i--) n = (n << 8n) | BigInt(buf[i]);
  return Number(n);
}
