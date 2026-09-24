// Pure local comparison. BASELINE_ROOT is an archive, not a branch checkout.
// node scripts/evidence/structure/multimodal-entry-reconcile.mjs BASELINE_ROOT OUTPUT
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const [baseline, output] = process.argv.slice(2);
if (!baseline || !output) throw new Error("provide BASELINE_ROOT and OUTPUT");
const read = p => fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
async function api(dir) {
  const load = suffix => import(pathToFileURL(path.join(dir, "frontend/src", suffix)));
  return {
    ...await load("structure/buildStructure.js"), ...await load("structure/config/normalize.js"),
    ...await load("cost/aggregate.js"), ...await load("cost/memory.js"),
  };
}
const beforeApi = await api(path.resolve(baseline)), afterApi = await api(root);
const rows = [];
for (const entry of read(path.join(root, "models/catalog.json")).models) {
  const config = read(path.join(root, "models", entry.config_path));
  const normalized = afterApi.normalizeConfig(config);
  const dir = path.dirname(path.join(root, "models", entry.config_path));
  for (const loading of ["config", "artifacts"]) {
    const snapshot = a => {
      const s = loading === "config" ? a.buildStructureFromConfig(config, { modelId: entry.model_id })
        : a.buildStructureFromArtifacts({ config, modelId: entry.model_id,
          checkpointTruth: read(path.join(dir, "header-truth.json")), sourceRef: read(path.join(dir, "source-ref.json")) });
      const phases = Object.fromEntries(["prefill", "decode"].map(phase => {
        const c = a.aggregateCost({ graph: s.graph, config: normalized, phase, batch: 1, sequence: 16,
          parameterCount: s.summary.parameters_by_dtype });
        return [phase, { macs: c.totalMacs, actions: c.actions,
          weightBytes: c.memory.weightBytes, kvBytes: c.memory.kvBytes, stateBytes: c.memory.stateBytes,
          modules: Object.fromEntries(c.nodes.map(r => [r.node.id, { macs: r.compute_macs, weights: r.weightBytes, actions: r.actions }])) }];
      }));
      return { nodes: s.graph.nodes.length, edges: s.graph.edges.length, capacity: a.graphWeightCapacity(s.graph), phases };
    };
    const before = snapshot(beforeApi), after = snapshot(afterApi);
    assert.deepEqual(after.capacity, before.capacity, `${entry.model_id}: fusion cannot change weights`);
    const deltas = {};
    for (const phase of ["prefill", "decode"]) {
      const b = before.phases[phase], a = after.phases[phase];
      for (const key of ["macs", "weightBytes", "kvBytes", "stateBytes"]) assert.equal(a[key], b[key], `${entry.model_id}/${phase}/${key}`);
      const changed = [...new Set([...Object.keys(a.modules), ...Object.keys(b.modules)])]
        .filter(id => JSON.stringify(a.modules[id]) !== JSON.stringify(b.modules[id]));
      deltas[phase] = Object.fromEntries(changed.map(id => [id, { before: b.modules[id] ?? null, after: a.modules[id] ?? null }]));
      delete b.modules; delete a.modules;
    }
    rows.push({ model_id: entry.model_id, loading, multimodal: Boolean(normalized.hasVision), before, after, changedModules: deltas });
  }
}
fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
fs.writeFileSync(output, JSON.stringify({ workload: { batch: 1, sequence: 16 }, rows }, null, 2) + "\n");
console.log(`${rows.length} model/path pairs: parameter capacity, MACs, weight/KV/state residency unchanged`);
console.log("Changed modules:", [...new Set(rows.flatMap(r => Object.keys(r.changedModules.prefill)))].join(", "));
