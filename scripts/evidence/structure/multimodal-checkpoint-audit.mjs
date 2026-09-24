#!/usr/bin/env node
/**
 * Audit the visual/projector/merger tensor paths against the production Graph IR.
 *
 * This deliberately keeps the downloaded safetensors metadata in memory only.
 * It reads the 8B+JSON headers through the existing checkpoint truth reader and
 * never downloads tensor payloads or writes model sidecars.
 *
 * Usage:
 *   node scripts/evidence/structure/multimodal-checkpoint-audit.mjs \
 *     --out docs/details/evidence/structure/multimodal-checkpoint-audit-2026-09-24.json
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromArtifacts } from "../../../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../../../frontend/src/structure/config/normalize.js";
import { fetchCheckpointTruth } from "../../../frontend/src/cost/weights.js";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const readJson = file => JSON.parse(fs.readFileSync(file, "utf8"));
const catalog = readJson(path.join(root, "models/catalog.json"));
const variantsPath = path.join(root, "docs/details/evidence/structure/multimodal-entry-variants.json");
const variants = new Map(readJson(variantsPath).map(row => [row.model_id, row]));
const output = process.argv.find(arg => arg.startsWith("--out="))?.slice("--out=".length)
  || "artifacts/architecture-repair/multimodal-checkpoint-audit.json";
const onlyModels = new Set((process.argv.find(arg => arg.startsWith("--model="))?.slice("--model=".length) || "")
  .split(",").map(value => value.trim()).filter(Boolean));
const timeoutMs = Math.max(5000, Number(process.env.MM_AUDIT_TIMEOUT_MS || 30000));
const concurrency = Math.max(1, Number(process.env.MM_AUDIT_CONCURRENCY || 3));

const ENDPOINTS = [
  { name: "hf-mirror", hubUrl: "https://hf-mirror.com", revision: "main", resolvePrefix: "" },
  { name: "huggingface", hubUrl: "https://huggingface.co", revision: "main", resolvePrefix: "" },
  { name: "modelscope", hubUrl: "https://www.modelscope.cn", revision: "master", resolvePrefix: "/models" },
];

function fetchWithTimeout(url, options = {}) {
  return fetch(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
}

function canonicalTensorName(name) {
  return String(name || "")
    .replace(/^model\./, "")
    .replace(/^(language_model|text_model)\./, "")
    .replace(/\.weight$/, "")
    .replace(/\.bias$/, "")
    .replace(/\.scale$/, "")
    .replace(/\.scales$/, "");
}

function tensorParam(name) {
  return String(name || "").split(".").at(-1);
}

function visualTensor(tensor) {
  return /(?:^|\.)(?:visual|vision_tower|vision|vit|image|multi_modal_projector|mm_projector|projector|aligner|patch_merge_mlp|merger)(?:\.|$)/i
    .test(tensor.name);
}

function visualSourceModule(module) {
  return /visual|vision|projector|patch_merge_mlp|merger|aligner/i.test(module.module_path || "");
}

function sourceModuleRepresented(graphIds, modulePath) {
  const canonical = String(modulePath || "").replace(/^root\./, "");
  return graphIds.has(canonical)
    || [...graphIds].some(id => id.startsWith(`${canonical}.`) || canonical.startsWith(`${id}.`));
}

function foldedOwner(graph, canonical, tensorName) {
  const actualParts = String(canonical || "").split(".");
  const byId = new Map(graph.nodes.map(node => [node.id, node]));
  const actualParam = tensorParam(tensorName);
  for (const node of graph.nodes) {
    if (!node.tensor_names?.some(name => tensorParam(name) === actualParam)) continue;
    const boundName = node.tensor_names.find(name => tensorParam(name) === actualParam);
    const boundParts = canonicalTensorName(boundName).split(".");
    if (boundParts.length !== actualParts.length) continue;
    const differingIndices = [];
    let compatible = true;
    for (let index = 0; index < actualParts.length; index += 1) {
      const actualPart = actualParts[index];
      const boundPart = boundParts[index];
      if (/^\d+$/.test(actualPart) && /^\d+$/.test(boundPart)) {
        if (actualPart !== boundPart) differingIndices.push(index);
        continue;
      }
      if (actualPart !== boundPart) {
        compatible = false;
        break;
      }
    }
    if (!compatible || !differingIndices.length) continue;
    const groups = [];
    let ancestor = node;
    while (ancestor) {
      if (ancestor.attributes?.range) {
        const [start, end] = String(ancestor.attributes.range).split("..").map(Number);
        for (const index of differingIndices) {
          const actual = Number(actualParts[index]);
          if (Number.isFinite(start) && Number.isFinite(end) && actual >= start && actual <= end) {
            groups.push({ index, group: ancestor });
          }
        }
      }
      ancestor = ancestor.parent_id ? byId.get(ancestor.parent_id) : null;
    }
    if (groups.some(({ index, group }) => {
      return Boolean(group) && Number.isFinite(index);
    })) {
      return { node, group: groups.find(({ group }) => group)?.group, index: differingIndices[0] };
    }
  }
  return null;
}

async function fetchTruth(modelId, preferredRevision) {
  const attempts = [];
  const endpoints = ENDPOINTS.map(endpoint => ({
    ...endpoint,
    revision: preferredRevision || endpoint.revision,
  }));
  for (const endpoint of endpoints) {
    try {
      const truth = await fetchCheckpointTruth({
        modelId,
        revision: endpoint.revision,
        hubUrl: endpoint.hubUrl,
        resolvePrefix: endpoint.resolvePrefix,
        fetchImpl: fetchWithTimeout,
      });
      if (truth?.tensors?.length && Number(truth.parameterTotal) > 0) {
        return { truth, endpoint: endpoint.name, revision: endpoint.revision, attempts };
      }
      attempts.push({ endpoint: endpoint.name, revision: endpoint.revision, error: "empty checkpoint header" });
    } catch (error) {
      attempts.push({ endpoint: endpoint.name, revision: endpoint.revision, error: error.message });
    }
  }
  return { truth: null, endpoint: null, revision: preferredRevision || null, attempts };
}

async function auditEntry(entry) {
  const modelDir = path.join(root, "models", entry.model_id);
  const config = readJson(path.join(modelDir, entry.config_path.split("/").slice(-1)[0]));
  const normalized = normalizeConfig(config);
  if (!normalized.hasVision) return null;

  const sourceRefPath = path.join(modelDir, "source-ref.json");
  const sourceRef = fs.existsSync(sourceRefPath) ? readJson(sourceRefPath) : null;
  const preferredRevision = variants.get(entry.model_id)?.revision || "main";
  const fetched = await fetchTruth(entry.model_id, preferredRevision);
  if (!fetched.truth) {
    return {
      model_id: entry.model_id,
      model_type: entry.model_type,
      revision: fetched.revision,
      endpoint: fetched.endpoint,
      status: "unknown",
      attempts: fetched.attempts,
    };
  }

  const structure = buildStructureFromArtifacts({
    modelId: entry.model_id,
    revision: fetched.revision,
    config,
    checkpointTruth: fetched.truth,
    sourceRef,
  });
  const graph = structure.graph;
  const graphIds = new Set(graph.nodes.map(node => node.canonical_id).filter(Boolean));
  const visualTensors = fetched.truth.tensors.filter(visualTensor);
  const owners = visualTensors.map(tensor => ({
    name: tensor.name,
    canonical: canonicalTensorName(tensor.name),
    owners: graph.nodes
      .filter(node => node.tensor_names?.includes(tensor.name))
      .map(node => node.canonical_id),
  }));
  const folded = owners.map(row => ({
    ...row,
    foldedOwner: row.owners.length === 0 ? foldedOwner(graph, row.canonical, row.name) : null,
  }));
  const shapeMismatches = folded.filter(row => {
    const owner = row.owners.length === 1
      ? graph.nodes.find(node => node.canonical_id === row.owners[0])
      : row.foldedOwner?.node;
    const expected = owner?.weight_shapes?.[tensorParam(row.name)];
    const actual = fetched.truth.tensors.find(tensor => tensor.name === row.name)?.shape;
    return expected && actual && JSON.stringify(expected) !== JSON.stringify(actual);
  });
  const unbound = folded.filter(row => row.owners.length === 0 && !row.foldedOwner);
  const duplicated = folded.filter(row => row.owners.length > 1);
  const represented = folded.filter(row => row.owners.length === 1 || row.foldedOwner);
  const sourceModules = (sourceRef?.modules || []).filter(visualSourceModule);
  const sourceGaps = sourceModules.filter(module => !sourceModuleRepresented(graphIds, module.module_path));
  const status = unbound.length || duplicated.length || shapeMismatches.length || sourceGaps.length ? "gap" : "verified";
  return {
    model_id: entry.model_id,
    model_type: entry.model_type,
    revision: fetched.revision,
    endpoint: fetched.endpoint,
    status,
    tensor_count: fetched.truth.tensors.length,
    parameterTotal: fetched.truth.parameterTotal,
    visual_tensor_count: visualTensors.length,
    visual_bound_tensor_count: represented.length,
    visual_folded_tensor_count: represented.filter(row => row.foldedOwner).length,
    visual_unbound_tensor_count: unbound.length,
    visual_duplicate_owner_count: duplicated.length,
    visual_shape_mismatch_count: shapeMismatches.length,
    source_module_count: sourceModules.length,
    source_module_gap_count: sourceGaps.length,
    unbound_visual_tensors: unbound.slice(0, 24).map(row => row.name),
    duplicate_visual_tensors: duplicated.slice(0, 24).map(row => row.name),
    shape_mismatch_visual_tensors: shapeMismatches.slice(0, 24).map(row => row.name),
    source_module_gaps: sourceGaps.slice(0, 24).map(module => module.module_path),
  };
}

async function pool(items, limit, worker) {
  let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      items[index].result = await worker(items[index].entry);
      console.log(`${index + 1}/${items.length} ${items[index].entry.model_id} ${items[index].result?.status || "skipped"}`);
    }
  }));
}

const entries = catalog.models
  .filter(entry => !onlyModels.size || onlyModels.has(entry.model_id))
  .filter(entry => normalizeConfig(readJson(path.join(root, "models", entry.config_path))).hasVision)
  .map(entry => ({ entry }));
await pool(entries, concurrency, auditEntry);
const rows = entries.map(item => item.result).filter(Boolean);
const report = {
  schema: "multimodal-checkpoint-audit/v1",
  generated_at: new Date().toISOString(),
  policy: "safetensors headers only; tensor payloads are never downloaded",
  total: rows.length,
  counts: rows.reduce((out, row) => {
    out[row.status] = (out[row.status] || 0) + 1;
    return out;
  }, {}),
  rows,
};
const destination = path.resolve(root, output);
fs.mkdirSync(path.dirname(destination), { recursive: true });
fs.writeFileSync(destination, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({
  output: destination,
  total: report.total,
  counts: report.counts,
  gaps: rows.filter(row => row.status === "gap").map(row => row.model_id),
  unknown: rows.filter(row => row.status === "unknown").map(row => row.model_id),
}, null, 2));
