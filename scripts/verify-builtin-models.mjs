import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../frontend/src/structure/config/normalize.js";
import { formulaForOperator } from "../frontend/src/structure/operators/formulas/index.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models", "catalog.json"), "utf8"));

// --dump-graphs <dir>：把每模型 structure.graph 落盘（供 NV-1 对账 harness 作 msv_graph）。
// 不传该参数时行为不变（仍只打印通过/节点数）。
const dumpArgIndex = process.argv.indexOf("--dump-graphs");
const dumpDir = dumpArgIndex !== -1 ? path.resolve(process.argv[dumpArgIndex + 1]) : null;
if (dumpDir) await fs.mkdir(dumpDir, { recursive: true });

const results = [];
// P7（步骤 7）：判据全部改走 structure.graph——legacy root 视图已停产。
// 层级判据 = root_id 直接子节点（parent_id 挂接 + order 排序）；算子注册
// 检查遍历全部图节点，语义 id 取 canonical_id（与原树节点 id 同源）。
function topLevelNodes(graph) {
  return (graph?.nodes || [])
    .filter((node) => node.parent_id === graph.root_id)
    .sort((left, right) => (left.order || 0) - (right.order || 0) || left.id.localeCompare(right.id));
}

function collectValidationErrors(structure, normalized) {
  const errors = [];
  const graph = structure?.graph;
  if (!structure?.summary || !graph?.nodes?.length || topLevelNodes(graph).length === 0) {
    errors.push("missing summary/graph/top-level graph nodes");
    return errors;
  }
  if (structure.source?.diagnostics?.resolution === "unsupported") {
    errors.push("resolved to unsupported");
  }
  const topLevel = topLevelNodes(graph);
  if (normalized.layers && !topLevel.some((node) => node.type === "decoder")) {
    errors.push("text layers exist but decoder is missing");
  }
  if (normalized.hasVision && !topLevel.some((node) => node.type === "vision-encoder")) {
    errors.push("vision config exists but vision graph is missing");
  }
  for (const node of graph.nodes) {
    if (node.type !== "operator") continue;
    const formulaId = node.attributes?.operator_id;
    if (!formulaId || !formulaForOperator(formulaId)) {
      errors.push(`operator ${node.canonical_id || node.id} has no registered formula`);
    }
  }
  return errors;
}

for (const entry of catalog.models) {
  try {
    const config = JSON.parse(await fs.readFile(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const structure = buildStructureFromConfig(config, {
      modelId: entry.model_id,
      source: "built-in config verification",
    });
    const errors = collectValidationErrors(structure, normalized);
    if (dumpDir && structure?.graph) {
      const file = path.join(dumpDir, `${entry.model_id.replace(/\//g, "__")}.graph.json`);
      await fs.writeFile(file, JSON.stringify(structure.graph));
    }
    const ok = errors.length === 0;
    results.push({
      model_id: entry.model_id,
      ok,
      architecture: structure?.summary?.architecture || null,
      graph_root: structure?.graph?.nodes?.find((node) => node.id === structure.graph.root_id)?.name || null,
      nodes: structure?.graph?.nodes?.length || 0,
      error: errors.join("; "),
    });
  } catch (error) {
    results.push({
      model_id: entry.model_id,
      ok: false,
      architecture: null,
      graph_root: null,
      nodes: 0,
      error: error.stack || error.message,
    });
  }
}

const failed = results.filter((result) => !result.ok);
console.log(JSON.stringify({ total: results.length, passed: results.length - failed.length, failed: failed.length }, null, 2));

if (failed.length > 0) {
  console.log(JSON.stringify(failed, null, 2));
  process.exit(1);
}

for (const result of results) {
  console.log(`${result.model_id}\t${result.architecture}\tnodes=${result.nodes}`);
}
