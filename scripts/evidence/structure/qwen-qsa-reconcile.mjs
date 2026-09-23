// Compare an explicitly supplied read-only baseline checkout/archive to this
// checkout. Output is an artifact, never a silently refreshed committed oracle.
// node scripts/evidence/structure/qwen-qsa-reconcile.mjs BASELINE_ROOT OUTPUT.json
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const [baseline, output] = process.argv.slice(2);
if (!baseline || !output) throw new Error("usage: qwen-qsa-reconcile.mjs BASELINE_ROOT OUTPUT.json");
const read = file => JSON.parse(fs.readFileSync(file, "utf8"));
async function snapshot(checkout, modelId, loading) {
  const load = suffix => import(pathToFileURL(path.join(checkout, `frontend/src/${suffix}`)));
  const { buildStructureFromConfig, buildStructureFromArtifacts } = await load("structure/buildStructure.js");
  const { normalizeConfig } = await load("structure/config/normalize.js");
  const { aggregateCost } = await load("cost/aggregate.js");
  const { graphWeightCapacity } = await load("cost/memory.js");
  const dir = path.join(checkout, "models", modelId);
  const config = read(path.join(dir, "config.json"));
  const structure = loading === "config" ? buildStructureFromConfig(config, { modelId })
    : buildStructureFromArtifacts({ config, modelId,
      checkpointTruth: read(path.join(dir, "header-truth.json")),
      sourceRef: read(path.join(dir, "source-ref.json")) });
  const normalized = normalizeConfig(config);
  const phases = {};
  for (const phase of ["prefill", "decode"]) {
    const cost = aggregateCost({ graph: structure.graph, config: normalized,
      batch: 1, sequence: 2051, phase, parameterCount: structure.summary?.parameters_by_dtype });
    phases[phase] = { totalMacs: cost.totalMacs, actions: cost.actions, memory: cost.memory, weightSource: cost.weightSource,
      unknownComputePaths: cost.unknownComputePaths,
      modules: Object.fromEntries(cost.nodes.map(row => [row.node.id, {
        macs: row.compute_macs, vector: row.actions?.vector ?? null,
        sfu: row.actions?.sfu ?? null, bytes: row.actions?.bytes ?? null,
        residentWeightBytes: row.weightBytes,
      }])) };
  }
  return { graph: { nodes: structure.graph.nodes.length, edges: structure.graph.edges.length },
    declared: graphWeightCapacity(structure.graph), phases };
}
const rows = [];
for (const modelId of ["Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next-FP8"]) {
  for (const loading of ["config", "artifacts"]) {
    const before = await snapshot(path.resolve(baseline), modelId, loading);
    const after = await snapshot(root, modelId, loading);
    const modules = {};
    for (const phase of ["prefill", "decode"]) {
      const old = before.phases[phase].modules, now = after.phases[phase].modules;
      modules[phase] = Object.fromEntries([...new Set([...Object.keys(old), ...Object.keys(now)])]
        .filter(id => JSON.stringify(old[id]) !== JSON.stringify(now[id]))
        .map(id => [id, { before: old[id] ?? null, after: now[id] ?? null }]));
      delete before.phases[phase].modules;
      delete after.phases[phase].modules;
    }
    rows.push({ modelId, loading, before, after, modules });
  }
}
fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
fs.writeFileSync(output, JSON.stringify({ baseline: path.resolve(baseline),
  workload: { batch: 1, sequence: 2051 }, theoreticalOnly: true, rows }, null, 2) + "\n");
for (const row of rows) console.log(row.modelId, row.loading, JSON.stringify({
  graph: [row.before.graph, row.after.graph], parameters: [row.before.declared.elements, row.after.declared.elements],
  prefillMacs: [row.before.phases.prefill.totalMacs, row.after.phases.prefill.totalMacs],
  decodeMacs: [row.before.phases.decode.totalMacs, row.after.phases.decode.totalMacs],
}));
