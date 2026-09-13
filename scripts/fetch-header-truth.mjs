#!/usr/bin/env node
// fetch-header-truth.mjs —— 一次性把 safetensors header 的 parameterTotal 存成 sidecar。
//
// 对照：@huggingface/hub parseSafetensorsMetadata（principles §4.1，禁止手写换算）。
// 只 range-read header（名 / dtype / shape），不下载权重数据区。
// 产物 models/<org>/<id>/header-truth.json 只存总量与按 dtype 计数，不落逐张量表
//（K3 量级 59.7MB；轻量元数据纪律，与 source-ref sidecar 同形态）。
//
// 用法：
//   node scripts/fetch-header-truth.mjs --model=Qwen/Qwen3.5-0.8B
//   node scripts/fetch-header-truth.mjs              # catalog 全量，跳过 Kimi-K3
//   node scripts/fetch-header-truth.mjs --overwrite
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { fetchCheckpointTruth } from "../frontend/src/cost/weights.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SKIP = new Set(["moonshotai/Kimi-K3"]);
const overwrite = process.argv.includes("--overwrite");
const only = process.argv.find((arg) => arg.startsWith("--model="))?.slice("--model=".length);

const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models", "catalog.json"), "utf8"));
const entries = (catalog.models || []).filter((entry) => !only || entry.model_id === only);

// 与 loadModelArtifacts.withRemoteTruth 同序：HF 直连失败再镜像 / ModelScope。
const ENDPOINTS = [
  { name: "huggingface", hubUrl: "https://huggingface.co", revision: "main", resolvePrefix: "" },
  { name: "hf-mirror", hubUrl: "https://hf-mirror.com", revision: "main", resolvePrefix: "" },
  { name: "modelscope", hubUrl: "https://www.modelscope.cn", revision: "master", resolvePrefix: "/models" },
];

async function fetchHeader(modelId) {
  let lastError = null;
  for (const endpoint of ENDPOINTS) {
    try {
      const truth = await fetchCheckpointTruth({
        modelId,
        revision: endpoint.revision,
        hubUrl: endpoint.hubUrl,
        resolvePrefix: endpoint.resolvePrefix,
      });
      if (Number.isFinite(truth?.parameterTotal) && truth.parameterTotal > 0) {
        return { ...truth, endpoint: endpoint.name };
      }
      lastError = new Error(`${endpoint.name}: parameterTotal missing`);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("all header endpoints failed");
}

if (only && entries.length === 0) {
  entries.push({ model_id: only, config_path: `${only}/config.json` });
}

function sidecarPath(entry) {
  return path.join(repoRoot, "models", path.dirname(entry.config_path), "header-truth.json");
}

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

const report = { wrote: [], skipped: [], failed: [] };

for (const entry of entries) {
  const modelId = entry.model_id;
  if (SKIP.has(modelId)) {
    report.skipped.push({ model_id: modelId, reason: "kimi-k3-no-dump" });
    continue;
  }
  const dest = sidecarPath(entry);
  if (!overwrite && await exists(dest)) {
    report.skipped.push({ model_id: modelId, reason: "exists" });
    continue;
  }
  try {
    const truth = await fetchHeader(modelId);
    const payload = {
      generated: "safetensors header (fetch-header-truth)",
      source: `https://huggingface.co/${modelId}/`,
      endpoint: truth.endpoint,
      method: truth.method || "hub",
      tensor_count: Array.isArray(truth.tensors) ? truth.tensors.length : null,
      parameterTotal: truth.parameterTotal,
      parameterCount: truth.parameterCount || null,
    };
    await fs.mkdir(path.dirname(dest), { recursive: true });
    await fs.writeFile(dest, `${JSON.stringify(payload, null, 2)}\n`);
    report.wrote.push({
      model_id: modelId,
      parameterTotal: payload.parameterTotal,
      tensor_count: payload.tensor_count,
      method: payload.method,
    });
    console.log(`✓ ${modelId}  parameterTotal=${payload.parameterTotal.toLocaleString("en-US")}  tensors=${payload.tensor_count}  method=${payload.method}  via=${payload.endpoint}`);
  } catch (error) {
    report.failed.push({ model_id: modelId, error: error.message });
    console.log(`✗ ${modelId}: ${error.message}`);
  }
}

console.log(JSON.stringify({
  wrote: report.wrote.length,
  skipped: report.skipped.length,
  failed: report.failed.length,
  failed_ids: report.failed.map((row) => row.model_id),
}, null, 2));
if (report.failed.length && !report.wrote.length) process.exit(1);
