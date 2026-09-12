#!/usr/bin/env node
// Bind catalog source-ref.json onto the frontend Graph for every builtin model
// except Kimi-K3. Optional --verify writes a temp Graph JSON and runs
// `msv verify --graph` (Transformers meta walk + reconciliation).
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../frontend/src/structure/buildStructure.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models", "catalog.json"), "utf8"));
const SKIP = new Set(["moonshotai/Kimi-K3"]);
const runVerify = process.argv.includes("--verify");
const verbose = process.argv.includes("--verbose");
const only = process.argv.find((arg) => arg.startsWith("--model="))?.slice("--model=".length);
const workerTimeoutSeconds = Number(process.env.MSV_STRUCTURE_WORKER_TIMEOUT_SECONDS || 180);
const spawnTimeoutMs = Math.max(30, workerTimeoutSeconds + 30) * 1000;

const results = [];
for (const entry of catalog.models) {
  if (only && entry.model_id !== only) continue;
  if (SKIP.has(entry.model_id)) {
    results.push({ model_id: entry.model_id, skipped: true, reason: "kimi-k3-no-dump" });
    continue;
  }
  const modelDir = path.join(repoRoot, "models", path.dirname(entry.config_path));
  const configPath = path.join(repoRoot, "models", entry.config_path);
  const sourceRefPath = path.join(modelDir, "source-ref.json");
  try {
    const config = JSON.parse(await fs.readFile(configPath, "utf8"));
    let sourceRef = null;
    try {
      sourceRef = JSON.parse(await fs.readFile(sourceRefPath, "utf8"));
    } catch {
      results.push({ model_id: entry.model_id, ok: false, error: "missing source-ref.json" });
      continue;
    }
    const structure = buildStructureFromConfig(config, {
      modelId: entry.model_id,
      source: "source-ref bind",
      sourceRef,
    });
    const bind = structure.source?.diagnostics?.source_ref || {};
    const row = {
      model_id: entry.model_id,
      ok: true,
      catalog_modules: sourceRef.modules?.length || 0,
      graph_nodes: structure.graph?.nodes?.length || 0,
      bound: bind.bound ?? 0,
      unmatched: bind.unmatched ?? 0,
    };
    if (runVerify) {
      Object.assign(row, await verifyGraph(entry.model_id, structure.graph));
      if (row.verify_ok === false) {
        row.ok = false;
        row.error = row.error || "verify failed";
      }
    }
    results.push(row);
  } catch (error) {
    results.push({ model_id: entry.model_id, ok: false, error: error.stack || error.message });
  }
}

const failed = results.filter((row) => row.ok === false);
const skipped = results.filter((row) => row.skipped);
console.log(JSON.stringify({
  total: results.length,
  passed: results.filter((row) => row.ok).length,
  failed: failed.length,
  skipped: skipped.length,
}, null, 2));
for (const row of results) {
  if (row.skipped) {
    console.log(`${row.model_id}\tskipped\t${row.reason}`);
    continue;
  }
  const extra = runVerify
    ? `\tverify=${row.verify_ok}\tconsistent=${row.structurally_consistent}`
      + `\tonly_tf=${row.only_transformers}\tonly_msv=${row.only_msv}\tmismatch=${row.mismatches}`
    : "";
  console.log(
    `${row.model_id}\tbound=${row.bound}\tunmatched=${row.unmatched}`
    + `\tmodules=${row.catalog_modules}\tnodes=${row.graph_nodes}${extra}`
    + (row.error ? `\t${row.error}` : ""),
  );
  if ((row.ok === false || verbose) && runVerify) {
    if (row.only_transformers_sample?.length) {
      console.log(`  only_tf ${JSON.stringify(row.only_transformers_sample)}`);
    }
    if (row.only_msv_sample?.length) {
      console.log(`  only_msv ${JSON.stringify(row.only_msv_sample)}`);
    }
    if (row.mismatch_sample?.length) {
      console.log(`  mismatch ${JSON.stringify(row.mismatch_sample)}`);
    }
  }
}
if (failed.length) process.exit(1);

async function verifyGraph(modelId, graph) {
  const graphPath = path.join(
    os.tmpdir(),
    `msv-graph-${modelId.replaceAll("/", "_")}-${process.pid}.json`,
  );
  await fs.writeFile(graphPath, `${JSON.stringify({ nodes: graph.nodes })}\n`);
  try {
    const proc = spawnSync(
      path.join(repoRoot, ".venv/bin/msv"),
      [
        "verify",
        "--model", modelId,
        "--source", "builtin",
        "--cache-policy", "offline",
        "--graph", graphPath,
        "--format", "json",
      ],
      {
        encoding: "utf8",
        timeout: spawnTimeoutMs,
        maxBuffer: 32 * 1024 * 1024,
        env: {
          ...process.env,
          MSV_STRUCTURE_WORKER_TIMEOUT_SECONDS: String(workerTimeoutSeconds),
        },
      },
    );
    let verify = {};
    try {
      verify = JSON.parse(proc.stdout || "{}");
    } catch {
      verify = {};
    }
    const diff = verify.evidence?.diff || {};
    const row = {
      verify_ok: Boolean(verify.ok),
      constructed: verify.evidence?.summary?.constructed ?? null,
      structurally_consistent: verify.evidence?.summary?.structurally_consistent ?? null,
      classified: diff.classified || {},
      only_transformers: diff.only_transformers?.length ?? null,
      only_msv: diff.only_msv?.length ?? null,
      mismatches: diff.mismatches?.length ?? null,
      only_transformers_sample: (diff.only_transformers || []).slice(0, 8),
      only_msv_sample: (diff.only_msv || []).slice(0, 8),
      mismatch_sample: (diff.mismatches || []).slice(0, 3),
    };
    if (proc.error?.code === "ETIMEDOUT") {
      row.verify_ok = false;
      row.error = `verify timeout ${spawnTimeoutMs}ms`;
    } else if (!verify.ok) {
      row.error = verify.error || proc.stderr?.trim()?.split("\n").at(-1) || `verify exit ${proc.status}`;
    }
    return row;
  } finally {
    await fs.unlink(graphPath).catch(() => {});
  }
}
