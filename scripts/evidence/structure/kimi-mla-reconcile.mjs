// Run against a read-only git archive, never another development branch.
// node scripts/evidence/structure/kimi-mla-reconcile.mjs BASELINE_ROOT OUTPUT
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
  return { ...await load("structure/buildStructure.js"), ...await load("structure/config/normalize.js"),
    ...await load("cost/aggregate.js"), ...await load("cost/memory.js") };
}
const oldApi = await api(path.resolve(baseline)), newApi = await api(root);
const rows = [];
for (const entry of read(path.join(root, "models/catalog.json")).models) {
  const config = read(path.join(root, "models", entry.config_path));
  const dir = path.dirname(path.join(root, "models", entry.config_path));
  const { mlaUseNope, ...normalized } = newApi.normalizeConfig(config);
  assert.deepEqual(normalized, oldApi.normalizeConfig(config), "normalization only adds mlaUseNope");
  for (const loading of ["config", "artifacts"]) {
    const snapshot = a => {
      const s = loading === "config" ? a.buildStructureFromConfig(config, { modelId: entry.model_id })
        : a.buildStructureFromArtifacts({ config, modelId: entry.model_id,
          checkpointTruth: read(path.join(dir, "skeleton-truth.json")) || read(path.join(dir, "header-truth.json")),
          sourceRef: read(path.join(dir, "source-ref.json")) });
      const phases = Object.fromEntries(["prefill", "decode"].map(phase => {
        const c = a.aggregateCost({ graph: s.graph, config: a.normalizeConfig(config), phase, batch: 1, sequence: 16,
          parameterCount: s.summary.parameters_by_dtype });
        return [phase, { macs: c.totalMacs, actions: c.actions,
          weightBytes: c.memory.weightBytes, kvBytes: c.memory.kvBytes, stateBytes: c.memory.stateBytes,
          modules: Object.fromEntries(c.nodes.map(r => [r.node.id, { macs: r.compute_macs, weights: r.weightBytes, actions: r.actions }])) }];
      }));
      return { nodes: s.graph.nodes.length, edges: s.graph.edges.length, capacity: a.graphWeightCapacity(s.graph), phases };
    };
    const before = snapshot(oldApi), after = snapshot(newApi);
    if (entry.model_id !== "moonshotai/Kimi-K3") assert.deepEqual(after, before, `${entry.model_id}: no unrelated cost/shape changes`);
    else {
      // 24 executing MLA layers; no extra gated MLA draft in this release.
      // Remove query-width gate and replace with value-width g_proj.
      assert.equal(after.capacity.elements - before.capacity.elements, -24 * 96 * 64 * 7168);
      assert.equal(after.capacity.bytes - before.capacity.bytes, -24 * 96 * 64 * 7168 * 2);
      for (const [phase, tokens] of [["prefill", 16], ["decode", 1]]) {
        assert.equal(after.phases[phase].macs - before.phases[phase].macs, -24 * 96 * 64 * 7168 * tokens);
        // Added 27 visual Q/K rotations, removed 24 text rotations.
        assert.equal(after.phases[phase].actions.vector - before.phases[phase].actions.vector,
          27 * 1024 * 2 * 12 * 128 * 3 - 24 * tokens * 2 * 96 * 192 * 3);
        assert.equal(after.phases[phase].actions.sfu, before.phases[phase].actions.sfu);
      }
    }
    const changedModules = {};
    for (const phase of ["prefill", "decode"]) {
      const b = before.phases[phase], a = after.phases[phase];
      assert.equal(a.kvBytes, b.kvBytes);
      assert.equal(a.stateBytes, b.stateBytes);
      const changed = [...new Set([...Object.keys(a.modules), ...Object.keys(b.modules)])]
        .filter(id => JSON.stringify(a.modules[id]) !== JSON.stringify(b.modules[id]));
      changedModules[phase] = Object.fromEntries(changed.map(id => [id, { before: b.modules[id] ?? null, after: a.modules[id] ?? null }]));
      delete b.modules; delete a.modules;
    }
    rows.push({ model_id: entry.model_id, loading, before, after, changedModules });
  }
}
fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
fs.writeFileSync(output, JSON.stringify({ workload: { batch: 1, sequence: 16 }, rows }, null, 2) + "\n");
console.log(`${rows.length} model/path pairs; only K3 changes; all KV/state residency unchanged`);
for (const row of rows.filter(r => r.model_id === "moonshotai/Kimi-K3")) {
  console.log(row.loading, JSON.stringify({ before: row.before, after: row.after }));
}
