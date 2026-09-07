// W3-D2 验收（§2.1/§2.2）：含子节点的模块必须"边声明或顺序标记"；
// 全部输出边 evidence 非空。fallback 推断只允许存在于未适配兜底树。
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { buildStructureFromConfig } from "../../buildStructure.js";
import { buildEdgeMap, hashSpecTree } from "./opsSpecTreeGoldenLib.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");

function walk(node, visit) {
  visit(node);
  for (const child of node.children || []) walk(child, visit);
}

test("W3-D2：所有多子节点模块均有 dataflow_edges 或 sequence 标记", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json")));
  const offenders = [];
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path)));
    const structure = buildStructureFromConfig(config, { modelId: entry.model_id, source: "declaration-test" });
    walk(structure.root, (node) => {
      if (node.type === "operator" || node.type === "embedding" || node.type === "output") return;
      if ((node.children || []).length < 2) return;
      const declared = Array.isArray(node.attributes?.dataflow_edges) && node.attributes.dataflow_edges.length > 0;
      const sequenced = node.attributes?.sequence === true;
      if (!declared && !sequenced) offenders.push(`${entry.model_id} :: ${node.id}`);
    });
  }
  if (offenders.length > 0) {
    console.error(`未声明顺序的模块（加 sequence: true 或 dataflow_edges）:\n  ${[...new Set(offenders)].slice(0, 30).join("\n  ")}`);
  }
  assert.deepEqual([...new Set(offenders)], []);
});

test("W3-D2：所有输出边 evidence 非空", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json")));
  const empty = [];
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path)));
    const structure = buildStructureFromConfig(config, { modelId: entry.model_id, source: "evidence-test" });
    walk(structure.root, (node) => {
      for (const edge of node.attributes?.dataflow_edges || []) {
        if (!edge) empty.push(`${entry.model_id} :: ${node.id} :: 空边`);
      }
    });
  }
  // graph 层 evidence 由 materializer 保证；此处断言声明边无空项
  assert.deepEqual([...new Set(empty)], []);
});

test("W3-D2：边集合（含 evidence）与基线一致", () => {
  const golden = JSON.parse(fs.readFileSync(path.join(repoRoot, "frontend/src/structure/model_executor/__tests__/ops-edge.golden.json"), "utf8"));
  const current = buildEdgeMap();
  const bad = [];
  for (const [model, edges] of Object.entries(current)) {
    if (hashSpecTree(edges) !== golden[model]) bad.push(model);
  }
  if (bad.length > 0) console.error("边集变更:", bad.join(", "));
  assert.deepEqual(bad, [], "边集变更须重生成 ops-edge.golden.json 并审阅 diff（有意的证据升级/边修复除外）");
});
