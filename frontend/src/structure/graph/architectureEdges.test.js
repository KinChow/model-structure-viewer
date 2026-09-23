import test from "node:test";
import assert from "node:assert/strict";
import { materializeStructureGraph } from "./materializeStructureGraph.js";
import { layoutGraph } from "../../diagram/layout.js";
const leaf = id => ({ id, type: "operator", children: [] });
const tree = () => ({ id: "model", children: [
  { id: "layers.2", children: [leaf("layers.2.indexer")] },
  { id: "layers.3", children: [leaf("layers.3.index_reuse")] },
] });
test("exact cross-parent canonical references survive collapsed view without mutating IR", () => {
  const root = tree();
  root.attributes = {
    dataflow_edges: [["layers.2.indexer", "layers.3.index_reuse"]],
    dataflow_edge_relations: [{ from: "layers.2.indexer", to: "layers.3.index_reuse", relation: "index-reuse" }],
  };
  const graph = materializeStructureGraph(root);
  const before = JSON.stringify(graph);
  assert.equal(graph.edges[0].source_canonical_id, "layers.2.indexer");
  const view = layoutGraph({ graph }, new Set(["root"]));
  assert.equal(view.edges.length, 1);
  assert.equal(view.edges[0].source, "root.0");
  assert.equal(view.edges[0].target, "root.1");
  assert.equal(view.edges[0].originalSource, "root.0.0");
  assert.equal(JSON.stringify(graph), before);
  const expanded = layoutGraph({ graph }, new Set(["root", "root.0", "root.1"]));
  assert.equal(expanded.edges[0].source, "root.0.0");
});
test("invalid declared endpoints never silently fall back to module order", () => {
  const root = tree();
  root.attributes = { dataflow_edges: [["missing", "layers.3"]] };
  assert.throws(() => materializeStructureGraph(root), /model.*missing/);
});
test("ambiguous suffixes require exact canonical references", () => {
  const root = { id: "model", attributes: { dataflow_edges: [["proj", "out"]] },
    children: [leaf("a.proj"), leaf("b.proj"), leaf("out")] };
  assert.throws(() => materializeStructureGraph(root), /ambiguous.*proj/);
});
test("explicit empty declaration means no edges, not inferred sequencing", () => {
  const root = tree();
  root.attributes = { dataflow_edges: [] };
  assert.deepEqual(materializeStructureGraph(root).edges, []);
});
