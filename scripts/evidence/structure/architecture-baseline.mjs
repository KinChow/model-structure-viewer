// Capture both public loading paths. stdout only; no metadata/weights are changed.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig, buildStructureFromArtifacts } from "../../../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../../../frontend/src/structure/config/normalize.js";
import { computeNodeCosts } from "../../../frontend/src/cost/compute.js";
import { graphWeightCapacity } from "../../../frontend/src/cost/memory.js";
const root = fileURLToPath(new URL("../../../", import.meta.url));
const read = p => fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
const catalog = read(path.join(root, "models/catalog.json"));
const rows = catalog.models.map(entry => {
  const file = path.join(root, "models", entry.config_path);
  const config = read(file), dir = path.dirname(file);
  const truth = read(path.join(dir, "skeleton-truth.json")) || read(path.join(dir, "header-truth.json"));
  const sourceRef = read(path.join(dir, "source-ref.json"));
  const paths = {
    config: buildStructureFromConfig(config, { modelId: entry.model_id }),
    artifacts: buildStructureFromArtifacts({ modelId: entry.model_id, config, checkpointTruth: truth, sourceRef }),
  };
  const normalized = normalizeConfig(config);
  return { model_id: entry.model_id, model_type: entry.model_type, truth: truth ? (truth.skeleton ? "skeleton" : "header") : "unavailable",
    paths: Object.fromEntries(Object.entries(paths).map(([name, structure]) => {
      const { graph } = structure;
      const phases = Object.fromEntries(["prefill", "decode"].map(phase => {
        const costs = computeNodeCosts(graph, normalized, { batch: 1, sequence: 2048, phase });
        return [phase, costs.filter(r => r.node.type === "operator").map(r => ({
          id: r.node.id, repeat: r.multiplier, resident: r.weightBytes,
          matrix: r.compute_macs, actions: r.actions,
        }))];
      }));
      return [name, { nodes: graph.nodes.length, edges: graph.edges.length,
        mtp: graph.nodes.filter(n => n.type === "mtp").length,
        capacity: graphWeightCapacity(graph), phases }];
    })) };
});
console.log(JSON.stringify({ total: rows.length, rows }));
