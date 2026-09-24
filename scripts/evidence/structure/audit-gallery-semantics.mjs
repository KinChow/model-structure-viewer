// Read-only Graph IR observations, not an oracle for paper/architecture correctness.
// Run from any directory: node scripts/evidence/structure/audit-gallery-semantics.mjs
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../../../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../../../frontend/src/structure/config/normalize.js";
import {
  attentionKindOf, attentionScheduleOf, indexerScheduleOf,
} from "../../../frontend/src/structure/layers/schedule.js";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const catalog = JSON.parse(fs.readFileSync(`${root}models/catalog.json`, "utf8"));
const tally = values => values.reduce((out, value) => {
  out[value] = (out[value] || 0) + 1;
  return out;
}, {});
const models = catalog.models.map(entry => {
  const config = JSON.parse(fs.readFileSync(`${root}models/${entry.config_path}`, "utf8"));
  const normalized = normalizeConfig(config);
  const graph = buildStructureFromConfig(config, { modelId: entry.model_id }).graph;
  const nodes = graph.nodes;
  const canonical = new Map(nodes.map(node => [node.id, node.canonical_id]));
  const dataflow = graph.edges.filter(edge => edge.kind === "dataflow");
  const degree = id => dataflow.filter(edge => edge.source === id || edge.target === id).length;
  const pairs = dataflow.map(edge => [
    canonical.get(edge.source), canonical.get(edge.target),
  ]);
  const top = nodes.filter(node => node.parent_id === graph.root_id);
  const topIds = new Set(top.map(node => node.id));
  const indexers = nodes.filter(node => ["dsa_indexer", "dsa_kpool_indexer"].includes(node.attributes?.operator_id));
  const gateNodes = nodes.filter(node => /MLA.*gate/.test(node.name));
  const attnRes = nodes.filter(node => node.attributes?.class === "AttentionResidual");
  const groups = nodes.filter(node => node.type === "layer-group" && /^layers\./.test(node.canonical_id));
  const blockSize = normalized.attnResBlockSize;
  const crossedAttnResBoundaries = groups.flatMap(node => {
    if (!blockSize || !node.attributes?.range) return [];
    const [start, end] = node.attributes.range.split("..").map(Number);
    const boundaries = [];
    for (let layer = start + 1; layer <= end; layer++) {
      if (layer % blockSize === 0) boundaries.push(layer);
    }
    return boundaries.length ? [{ range: node.attributes.range, boundaries }] : [];
  });
  return {
    model_id: entry.model_id,
    model_type: entry.model_type,
    layers: normalized.layers,
    attention_mix: tally(attentionScheduleOf(normalized)
      || Array.from({ length: normalized.layers }, () => attentionKindOf(normalized))),
    indexer_mix: tally(indexerScheduleOf(normalized) || []),
    has_vision: normalized.hasVision,
    top_level_edges: dataflow.filter(edge => topIds.has(edge.source) && topIds.has(edge.target))
      .map(edge => [canonical.get(edge.source), canonical.get(edge.target)]),
    isolated_mla_gates: gateNodes.filter(node => degree(node.id) === 0)
      .map(node => ({ id: node.canonical_id, output_shape: node.output_shape })),
    attention_residual_incoming: pairs.filter(([, target]) => attnRes.some(node => node.canonical_id === target)),
    folded_attnres_boundary_crossings: crossedAttnResBoundaries,
    reuse_indexers: indexers.filter(node => node.attributes.indexer_mode === "reuse").length,
    cross_layer_indexer_edges: pairs.filter(([source, target]) => {
      // IndexShare targets are semantic `index_reuse` nodes, not indexer
      // nodes. The previous predicate required both sides to contain the
      // literal "indexer" and therefore reported a false zero for all
      // declared cross-layer reuse relations.
      const sourceIsIndexer = source?.endsWith(".indexer");
      const targetIsIndexReference = target?.endsWith(".indexer")
        || target?.endsWith(".index_reuse");
      if (!sourceIsIndexer || !targetIsIndexReference) return false;
      const layer = value => value.match(/(?:^|\.)layers\.(\d+)\./)?.[1];
      return layer(source) != null && layer(target) != null && layer(source) !== layer(target);
    }),
    activation_operators: [...new Set(nodes.filter(node => node.attributes?.operator_id === "swiglu")
      .map(node => `${node.name}:${node.attributes.activation || "unspecified"}`))],
    isolated_compressor_operators: nodes.filter(node => node.type === "operator"
      && node.canonical_id?.includes("compressor") && degree(node.id) === 0)
      .map(node => node.canonical_id),
    ple_modules: nodes.filter(node => node.type === "ple").map(node => node.canonical_id),
  };
});
console.log(JSON.stringify({
  schema: "gallery-semantics-observations/v1",
  note: "Config-derived graph observations only. External-source review is separate; zero findings is not an architecture pass.",
  total: models.length,
  family_counts: tally(models.map(model => model.model_type)),
  vision_models: models.filter(model => model.has_vision).length,
  models,
}, null, 2));
