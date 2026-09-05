import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../structure/buildStructure.js";
import { layoutGraph } from "./layout.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");

function expandedPaths(root) {
  const paths = new Set();
  function visit(node, currentPath) {
    paths.add(currentPath);
    (node.children || []).forEach((child, index) => visit(child, `${currentPath}.${index}`));
  }
  visit(root, "root");
  return paths;
}

test("all built-in multi-operator modules use semantic graph edges", async () => {
  const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const checked = [];
  for (const entry of catalog.models) {
    const config = JSON.parse(await fs.readFile(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const structure = buildStructureFromConfig(config, { modelId: entry.model_id, source: "builtin-layout-test" });
    const graph = layoutGraph(structure.root, expandedPaths(structure.root));
    for (const parent of graph.nodes) {
      if (!(["attention", "mlp", "moe"].includes(parent.node?.type))) continue;
      const childDepth = parent.path.split(".").length + 1;
      const children = graph.nodes.filter((node) => node.path.startsWith(`${parent.path}.`) && node.path.split(".").length === childDepth);
      const edges = graph.edges.filter((edge) => edge.source.startsWith(`${parent.path}.`) && edge.target.startsWith(`${parent.path}.`));
      assert.equal(edges.some((edge) => edge.evidence === "module-order"), false, `${entry.model_id}: ${parent.node.name} has a sequential-only layout edge`);
      assert.ok(edges.filter((edge) => edge.evidence === "semantic-flow").length >= 2, `${entry.model_id}: ${parent.node.name} has no semantic flow`);
      checked.push({ modelId: entry.model_id, parent: parent.node.name, children: children.length });
    }
  }
  assert.equal(catalog.models.length, 44);
  assert.ok(checked.length > 0);
});
