import assert from "node:assert/strict";
import test from "node:test";
import { bindTruthToGraph, skeletonTruthGraph, enrichGraphWithTruth } from "../graphTruth.js";

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

test("bound checkpoint modules are not appended a second time as gaps", () => {
  const template = { version: 2, schema_version: 2, root_id: "root",
    nodes: [
      { id: "root", canonical_id: "root", parent_id: null, type: "model", attributes: {} },
      { id: "root.0", canonical_id: "norm", parent_id: "root", type: "operator", attributes: {} },
    ], edges: [] };
  const tensors = [
    { name: "model.language_model.norm.weight", dtype: "BF16", shape: [8] },
    { name: "model.language_model.extra.weight", dtype: "BF16", shape: [4, 8] },
  ];
  const { graph, diagnostics } = enrichGraphWithTruth(template, { tensors }, { hasBuilder: true });
  assert.equal(diagnostics.bound_tensors, 1);
  assert.deepEqual(diagnostics.template_gaps, ["model.language_model.extra"]);
  const bound = graph.nodes.filter(node => node.tensor_names?.includes(tensors[0].name));
  assert.equal(bound.length, 1, "graph layout ids must not be compared to skeleton canonical ids");
  assert.equal(graph.nodes.filter(node => node.tensor_names?.includes(tensors[1].name)).length, 1);
  assert.equal(graph.nodes.reduce((sum, node) => sum + (node.params || 0), 0), 40);
});
