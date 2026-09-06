import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelsRoot = path.join(repoRoot, "models");
const previousCatalogPath = path.join(modelsRoot, "catalog.json");

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

const models = [];
let previousModels = [];
try {
  const previousCatalog = await readJson(previousCatalogPath);
  previousModels = Array.isArray(previousCatalog.models) ? previousCatalog.models : [];
} catch {
  previousModels = [];
}
const previousById = new Map(previousModels.map((entry) => [entry.model_id, entry]));
for (const org of await listDirectories(modelsRoot)) {
  for (const model of await listDirectories(path.join(modelsRoot, org))) {
    const modelDir = path.join(modelsRoot, org, model);
    const configPath = path.join(modelDir, "config.json");
    if (!(await exists(configPath))) continue;

    const config = await readJson(configPath);
    const previous = previousById.get(`${org}/${model}`) || {};
    models.push({
      model_id: `${org}/${model}`,
      ...(previous.display_name ? { display_name: previous.display_name } : {}),
      ...(previous.release_time ? { release_time: previous.release_time } : {}),
      config_path: `${org}/${model}/config.json`,
      model_type: config.model_type || null,
      architectures: Array.isArray(config.architectures) ? config.architectures : [],
      verified: true,
    });
  }
}

await fs.writeFile(
  path.join(modelsRoot, "catalog.json"),
  `${JSON.stringify({ models }, null, 2)}\n`,
);
