import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { normalizeConfig } from "./config/normalize.js";
import { resolveArchitecture } from "./registry/resolveArchitecture.js";
import { buildNetwork } from "./model_executor/models/index.js";
import { createStructureIr } from "./ir/createStructureIr.js";
import { materializeModelStructure } from "./materializers/toStructureNode.js";
import { formulaForOperator } from "./formulas/index.js";
import { derivedWeightParameters } from "../cost/derivedWeights.js";
import { kvBytesPerToken, linearStateBytesPerSequence } from "../cost/memory.js";
import { aggregateCost } from "../cost/aggregate.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");

function walk(node, visit) {
  visit(node);
  (node.children || []).forEach((child) => walk(child, visit));
}

test("all built-in models have modules, formulas, and finite cost inputs", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  assert.equal(catalog.models.length, 59);
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const canonicalIds = structure.graph.nodes.map((node) => node.canonical_id);
    assert.equal(new Set(canonicalIds).size, canonicalIds.length, `${entry.model_id}: duplicate canonical graph ids`);
    assert.ok(structure.graph.edges.every((edge) => edge.source_canonical_id && edge.target_canonical_id), `${entry.model_id}: graph edge missing canonical endpoints`);
    assert.ok(structure.root.children.length > 0, `${entry.model_id}: empty module tree`);
    if (normalized.hasVision) {
      assert.ok(
        structure.root.children.some((node) => node.type === "vision-encoder"),
        `${entry.model_id}: vision config is missing from the structure`,
      );
    }
    walk(structure.root, (node) => {
      if (node.type !== "operator") return;
      assert.ok(node.attributes.formula, `${entry.model_id}: missing formula for ${node.attributes.operator_id}`);
      assert.ok(formulaForOperator(node.attributes.formula_id), `${entry.model_id}: unknown formula id ${node.attributes.formula_id}`);
      assert.ok(Array.isArray(node.input_shape), `${entry.model_id}: ${node.id} missing numeric input shape`);
      assert.ok(Array.isArray(node.output_shape), `${entry.model_id}: ${node.id} missing numeric output shape`);
    });
    const cost = aggregateCost({ graph: structure.graph, config: normalized, phase: "prefill", batch: 1, sequence: 128 });
    assert.equal(cost.computeComplete, true, `${entry.model_id}: ${cost.unknownComputePaths.join(", ")}`);
    assert.ok(Number.isFinite(derivedWeightParameters(normalized)), `${entry.model_id}: invalid derived weights`);
    assert.ok(Number.isFinite(kvBytesPerToken(normalized)), `${entry.model_id}: invalid KV cost`);
    assert.ok(Number.isFinite(linearStateBytesPerSequence(normalized)), `${entry.model_id}: invalid recurrent state cost`);
    if (entry.model_id === "moonshotai/Kimi-K3") {
      assert.equal(resolved.canonicalArchitecture, "hybrid-multimodal-moe-decoder");
      assert.deepEqual(normalized.attentionSchedule.filter((kind) => kind === "linear").length, 69);
      assert.deepEqual(normalized.attentionSchedule.filter((kind) => kind === "mla").length, 24);
      assert.equal(normalized.sharedExperts, 2);
      assert.equal(normalized.attnResBlockSize, 12);
      const nodes = [];
      walk(structure.root, (node) => nodes.push(node));
      assert.ok(nodes.some((node) => node.name === "Attention Residual"), "Kimi-K3: missing per-layer AttnRes module");
      assert.ok(nodes.some((node) => node.name === "Output Attention Residual"), "Kimi-K3: missing output AttnRes module");
      assert.ok(nodes.some((node) => node.name === "MLA output gate"), "Kimi-K3: missing MLA output gate");
      assert.ok(nodes.some((node) => node.id.endsWith(".moe.shared_experts")), "Kimi-K3: missing shared experts");
    }
  }
});
