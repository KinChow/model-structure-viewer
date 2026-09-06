// weights.js —— 从 safetensors header 读 checkpoint 真值（零权重下载，KB 级 range read）。
//
// 依据 evolution_design.md §4.2：参数量换算优先用 @huggingface/hub 的 parseSafetensorsMetadata，
// 不手写（子字节量化打包容器宽度、bitsandbytes__ 前缀、exponent-only dtype 等边界都在库里）。
// 但浏览器 + 非 HF 源（如 ModelScope CDN 不暴露 content-range/etag 响应头）时库会硬抛错，
// 此时回退到自研 readSafetensorsHeaders（只读响应体，浏览器跨域安全；未量化模型结果等价）。
// checkpoint 树由 structure/truth/skeleton.js 构建；这里只负责读取和换算。

import { parseSafetensorsMetadata } from "@huggingface/hub";
import { readSafetensorsHeaders } from "./safetensorsReader.js";

/** 过滤非张量 key（safetensors 元数据 / bitsandbytes 量化状态）。 */
function isTensorKey(name) {
  if (name.startsWith("__")) return false;
  if (name.startsWith("bitsandbytes__")) return false;
  return true;
}

function normalizeFromHeaders(headers) {
  const tensors = [];
  for (const header of headers) {
    for (const [name, info] of Object.entries(header)) {
      if (!isTensorKey(name)) continue;
      tensors.push({ name, dtype: info.dtype, shape: info.shape });
    }
  }
  tensors.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return tensors;
}

/**
 * 读取模型源的 safetensors 元数据，归一化为：
 * - tensors: [{name, dtype, shape}]（跨分片合并）
 * - parameterCount: 逐 dtype 参数量（库计算优先；手动回退按 Σnumel，未量化即精确）
 * - parameterTotal: 模型级总参数量
 * - method: "hub" | "manual"（便于诊断/UI 标注）
 *
 * 失败（无 safetensors / gated / 网络）时抛错，由调用方降级为模板路径。
 */
export async function fetchCheckpointTruth({
  modelId,
  revision = "main",
  hubUrl,
  resolvePrefix = "",
  fetchImpl = fetch,
}) {
  try {
    const parsed = await parseSafetensorsMetadata({
      repo: { type: "model", name: modelId },
      revision,
      hubUrl,
      computeParametersCount: true,
      fetch: fetchImpl,
    });
    const headers = parsed.sharded ? Object.values(parsed.headers) : [parsed.header];
    const parameterCount = parsed.parameterCount ?? null;
    return {
      tensors: normalizeFromHeaders(headers),
      parameterCount,
      parameterTotal:
        parsed.parameterTotal ??
        (parameterCount ? Object.values(parameterCount).reduce((sum, n) => sum + n, 0) : null),
      method: "hub",
    };
  } catch (err) {
    // 浏览器 + 非 HF 源（缺 expose-headers 读不到 content-range/etag）→ 手动读 header 结构
    const manual = await readSafetensorsHeaders({ modelId, revision, hubUrl, resolvePrefix, fetchImpl });
    return { ...manual, method: "manual" };
  }
}
