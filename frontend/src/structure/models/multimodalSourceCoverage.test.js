import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { buildStructureFromArtifacts } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));

function readModelFile(modelId, file, { optional = false } = {}) {
  const filePath = path.join(repoRoot, "models", modelId, file);
  if (optional && !fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function canonicalWithoutRoot(modulePath) {
  return String(modulePath || "").replace(/^root\./, "");
}

function graphRepresentsModule(graphIds, modulePath) {
  const canonical = canonicalWithoutRoot(modulePath);
  return graphIds.has(canonical)
    || [...graphIds].some((id) => id.startsWith(`${canonical}.`) || canonical.startsWith(`${id}.`));
}

test("multimodal source-ref visual modules are represented by Graph IR", () => {
  const visualModuleGaps = [];
  const checkedModels = [];
  let sourceModules = 0;

  for (const entry of catalog.models) {
    const modelDir = path.join(repoRoot, "models", entry.model_id);
    const sourcePath = path.join(modelDir, "source-ref.json");
    if (!fs.existsSync(sourcePath)) {
      assert.fail(`${entry.model_id}: source-ref sidecar is missing`);
    }

    const sourceRef = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
    const visualModules = (sourceRef.modules || []).filter((module) =>
      /visual|vision|projector/i.test(module.module_path || ""));
    if (!visualModules.length) {
      if (structureHasVision(entry.model_id)) visualModuleGaps.push(entry.model_id);
      continue;
    }

    const structure = buildStructureFromArtifacts({
      modelId: entry.model_id,
      config: readModelFile(entry.model_id, "config.json"),
      checkpointTruth: readModelFile(entry.model_id, "header-truth.json", { optional: true }),
      sourceRef,
    });
    const graphIds = new Set(
      structure.graph.nodes.map((node) => node.canonical_id).filter(Boolean),
    );
    const missing = visualModules.filter((module) =>
      !graphRepresentsModule(graphIds, module.module_path));

    assert.deepEqual(
      missing,
      [],
      `${entry.model_id}: source-ref visual modules missing from Graph IR: ${
        missing.map((module) => module.module_path).join(", ")}`,
    );
    sourceModules += visualModules.length;
    checkedModels.push(entry.model_id);
  }

  assert.deepEqual(new Set(visualModuleGaps), new Set());
  assert.equal(checkedModels.length, 39);
  assert.equal(sourceModules, 788);

  const visualEvidence = JSON.parse(fs.readFileSync(
    path.join(
      repoRoot,
      "docs/details/evidence/structure/deepseek_v4_vision_sources.json",
    ),
    "utf8",
  ));
  assert.deepEqual(
    new Set(visualEvidence.sources.map((source) => source.model_id)),
    new Set([
      "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
      "deepseek-ai/DeepSeek-V4.1-Flash",
    ]),
  );
  for (const source of visualEvidence.sources) {
    assert.match(source.forward_url, /\/inference\/model\.py$/);
    assert.match(source.vision_url, /\/inference\/vision\.py$/);
    assert.match(source.forward_sha256, /^[0-9a-f]{64}$/);
    assert.match(source.vision_sha256, /^[0-9a-f]{64}$/);
    assert.ok(source.anchors.vit && source.anchors.aligner);
  }
});

function structureHasVision(modelId) {
  const config = readModelFile(modelId, "config.json");
  return Boolean(normalizeConfig(config).hasVision);
}
