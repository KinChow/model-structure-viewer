import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelsRoot = path.join(repoRoot, "models");
const allowedMetadata = /^(configuration_.*\.py|modeling_.*\.py|tokenization_.*\.py)$/;

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function listDirectories(dirPath) {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  return entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort();
}

async function listFiles(dirPath) {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  return entries.filter((entry) => entry.isFile()).map((entry) => entry.name).sort();
}

const models = [];
for (const org of await listDirectories(modelsRoot)) {
  for (const model of await listDirectories(path.join(modelsRoot, org))) {
    const modelDir = path.join(modelsRoot, org, model);
    const configPath = path.join(modelDir, "config.json");
    if (!(await exists(configPath))) continue;

    const config = await readJson(configPath);
    const files = await listFiles(modelDir);
    models.push({
      model_id: `${org}/${model}`,
      config_path: `${org}/${model}/config.json`,
      model_type: config.model_type || null,
      architectures: Array.isArray(config.architectures) ? config.architectures : [],
      metadata_files: files.filter((file) => allowedMetadata.test(file)),
      verified: true,
    });
  }
}

await fs.writeFile(
  path.join(modelsRoot, "catalog.json"),
  `${JSON.stringify({ models }, null, 2)}\n`,
);
