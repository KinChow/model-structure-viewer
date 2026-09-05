import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelsRoot = path.join(repoRoot, "models");
const overwrite = process.argv.includes("--overwrite");
const allowedNames = new Set(["model.safetensors.index.json"]);

const sources = [
  {
    name: "huggingface-mirror",
    revision: "main",
    list: (modelId) => `https://hf-mirror.com/api/models/${modelId}`,
    file: (modelId, filename) => `https://hf-mirror.com/${modelId}/resolve/main/${filename}`,
  },
  {
    name: "modelscope",
    revision: "master",
    list: (modelId) => `https://www.modelscope.cn/api/v1/models/${modelId}/repo/files?Revision=master&Recursive=True`,
    file: (modelId, filename) => `https://www.modelscope.cn/models/${modelId}/resolve/master/${filename}`,
  },
];

async function readCatalog() {
  return JSON.parse(await fs.readFile(path.join(modelsRoot, "catalog.json"), "utf8"));
}

async function getJson(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function listFiles(source, modelId) {
  const payload = await getJson(source.list(modelId));
  if (source.name === "modelscope") {
    return new Set((payload.Data?.Files || []).map((file) => file.Path));
  }
  return new Set((payload.siblings || []).map((file) => file.rfilename));
}

async function download(source, modelId, filename) {
  const response = await fetch(source.file(modelId, filename), { signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length === 0) throw new Error("empty response");
  return bytes;
}

async function main() {
  const catalog = await readCatalog();
  const report = { downloaded: [], skipped: [], unavailable: [], sourceErrors: [] };

  for (const entry of catalog.models || []) {
    const modelId = entry.model_id;
    const modelDir = path.join(modelsRoot, ...modelId.split("/"));
    const sourceFiles = [];
    for (const source of sources) {
      try {
        sourceFiles.push({ source, files: await listFiles(source, modelId) });
      } catch (error) {
        report.sourceErrors.push({ modelId, source: source.name, error: error.message });
      }
    }

    const remoteNames = new Set();
    for (const { files } of sourceFiles) {
      for (const filename of files) {
        if (allowedNames.has(path.basename(filename)) && filename === path.basename(filename)) remoteNames.add(filename);
      }
    }

    for (const filename of [...remoteNames].sort()) {
      const target = path.join(modelDir, filename);
      try {
        await fs.access(target);
        if (!overwrite) {
          report.skipped.push({ modelId, filename, reason: "exists" });
          continue;
        }
      } catch {
        // Missing target: download it below.
      }

      let saved = false;
      for (const { source, files } of sourceFiles) {
        if (!files.has(filename)) continue;
        try {
          const bytes = await download(source, modelId, filename);
          await fs.mkdir(modelDir, { recursive: true });
          await fs.writeFile(target, bytes);
          report.downloaded.push({ modelId, filename, source: source.name, bytes: bytes.length });
          saved = true;
          break;
        } catch (error) {
          report.sourceErrors.push({ modelId, filename, source: source.name, error: error.message });
        }
      }
      if (!saved) report.unavailable.push({ modelId, filename });
    }
  }

  console.log(JSON.stringify({
    models: catalog.models?.length || 0,
    downloaded: report.downloaded.length,
    skipped: report.skipped.length,
    unavailable: report.unavailable,
    sourceErrors: report.sourceErrors,
    files: report.downloaded,
  }, null, 2));
}

await main();
