import assert from "node:assert/strict";
import test from "node:test";
import { bindTruthToGraph, skeletonTruthGraph } from "../graphTruth.js";

test("bindTruthToGraph binds checkpoint facts by canonical graph id", () => {
  const truthGraph = skeletonTruthGraph({
    id: "model",
    name: "Model",
    type: "module",
    params: 0,
    weight_shapes: {},
    children: [{
      id: "model.language_model.norm",
      name: "norm",
      type: "module",
      params: 8,
      weight_shapes: { weight: [8] },
      dtype: "BF16",
      tensor_names: ["model.language_model.norm.weight"],
      children: [],
    }],
  });
  const result = bindTruthToGraph({
    version: 2,
    schema_version: 2,
    root_id: "root",
    nodes: [{ id: "root.0", canonical_id: "norm", module_id: "norm", parent_id: "root", name: "final norm", type: "operator", attributes: {} }],
    edges: [],
  }, truthGraph);
  assert.equal(result.graph.nodes[0].params, 8);
  assert.equal(result.graph.nodes[0].value_source, "checkpoint");
  assert.deepEqual(result.graph.nodes[0].tensor_names, ["model.language_model.norm.weight"]);
  assert.equal(result.diagnostics.graph_bound_tensors, 1);
});
