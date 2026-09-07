// W1 第二批差分测试（T2）：attention 族（模块节点 + matmul/fused 算子叶），
// 旧 nodeMacs vs 新 extractor 镜像。已定性旧链问题（vision 类）单独归类。
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../../../structure/buildStructure.js";
import { normalizeConfig } from "../../../structure/config/normalize.js";
import { computeNodeCosts } from "../../../cost/compute.js";
import { countsForNode } from "../../../structure/formulas/extractor.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");
const IN_SCOPE_IDS = new Set(["matmul", "qsa_attention", "minimax_sparse_attention", "dsv4_swa_attention", "dsv4_compressed_attention"]);

test("attention 族差分：全部内置模型逐节点相等（vision 类为已知旧链修正）", async () => {
  const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const mismatches = [];
  const knownIssues = [];
  let checked = 0;

  for (const entry of catalog.models) {
    const config = JSON.parse(await fs.readFile(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const structure = buildStructureFromConfig(config, { modelId: entry.model_id, source: "differential-test" });
    const rows = computeNodeCosts(structure.root, normalized, { batch: 1, sequence: 128, phase: "prefill" });

    for (const row of rows) {
      const operatorId = String(row.node?.attributes?.operator_id || "").toLowerCase();
      const type = String(row.node?.type || "").toLowerCase();
      if (type !== "attention" && !IN_SCOPE_IDS.has(operatorId)) continue;
      const oldMatrix = row.macs == null ? null : row.macs / row.multiplier;
      const fresh = countsForNode(row.node, {
        config: normalized,
        options: { batch: 1, sequence: 128, phase: "prefill" },
        path: row.path,
        bytesPerElement: 2,
      });
      const newMatrix = fresh == null ? null : fresh.matrix;
      checked += 1;

      const bothZero = (oldMatrix ?? 0) === 0 && (newMatrix ?? 0) === 0;
      if (bothZero) continue;
      if (oldMatrix == null && newMatrix == null) continue;
      if (oldMatrix !== newMatrix) {
        if (row.node?.attributes?.modality === "vision") {
          knownIssues.push(`${entry.model_id} ${row.path} (${operatorId || type}): old=${oldMatrix} new=${newMatrix}`);
          continue;
        }
        mismatches.push(`${entry.model_id} ${row.path} (${operatorId || type}, kind=${row.node?.attributes?.attention_kind}): old=${oldMatrix} new=${newMatrix}`);
      }
    }
  }
  console.error(`checked=${checked} vision已知修正=${knownIssues.length}`);
  if (knownIssues.length > 0) console.error("vision 修正样例:\n" + knownIssues.slice(0, 4).join("\n"));
  if (mismatches.length > 0) console.error("未定性差分:\n" + mismatches.slice(0, 12).join("\n"));
  assert.ok(checked > 0, "attention 节点数为 0，测试无效");
  assert.deepEqual(mismatches, []);
});
