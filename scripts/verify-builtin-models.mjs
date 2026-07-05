import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../frontend/src/structure/buildStructure.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models", "catalog.json"), "utf8"));

const results = [];
for (const entry of catalog.models) {
  try {
    const config = JSON.parse(await fs.readFile(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const structure = buildStructureFromConfig(config, {
      modelId: entry.model_id,
      source: "built-in config verification",
    });
    const ok = Boolean(structure?.summary && structure?.root && structure.root.children?.length > 0);
    results.push({
      model_id: entry.model_id,
      ok,
      canonical_architecture: structure?.summary?.canonical_architecture || null,
      root: structure?.root?.name || null,
      children: structure?.root?.children?.length || 0,
      error: ok ? "" : "missing summary/root/root.children",
    });
  } catch (error) {
    results.push({
      model_id: entry.model_id,
      ok: false,
      canonical_architecture: null,
      root: null,
      children: 0,
      error: error.stack || error.message,
    });
  }
}

const failed = results.filter((result) => !result.ok);
console.log(JSON.stringify({ total: results.length, passed: results.length - failed.length, failed: failed.length }, null, 2));

if (failed.length > 0) {
  console.log(JSON.stringify(failed, null, 2));
  process.exit(1);
}

for (const result of results) {
  console.log(`${result.model_id}\t${result.canonical_architecture}\tchildren=${result.children}`);
}
