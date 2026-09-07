// W1 第二批差分测试（T1）：linear 子集，旧 nodeMacs vs 新 extractor+counts。
// 断言：非 vision 的 linear 节点逐节点相等；vision 类差异是已定性旧链 bug
// （旧 linearMacs 未传 vision/visionTokens），归入旧链问题清单不计失败。
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

test("linear 子集差分：全部内置模型逐节点相等（vision 类为已知旧链修正）", async () => {
  const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const mismatches = [];
  const knownIssues = [];
  let checked = 0;
  let sharedKnown = 0;
  let oldNullNewValue = 0;
  let oldValueNewNull = 0;

  for (const entry of catalog.models) {
    const config = JSON.parse(await fs.readFile(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const structure = buildStructureFromConfig(config, { modelId: entry.model_id, source: "differential-test" });
    const rows = computeNodeCosts(structure.root, normalized, { batch: 1, sequence: 128, phase: "prefill" });

    for (const row of rows) {
      if (String(row.node?.attributes?.operator_id || "").toLowerCase() !== "linear") continue;
      const oldMatrix = row.macs == null ? null : row.macs / row.multiplier; // 还原单实例
      const fresh = countsForNode(row.node, {
        config: normalized,
        options: { batch: 1, sequence: 128, phase: "prefill" },
        path: row.path,
        bytesPerElement: 2,
      });
      const newMatrix = fresh == null ? null : fresh.matrix;
      checked += 1;

      if (oldMatrix == null && newMatrix == null) continue;
      if (oldMatrix === 0 && newMatrix === 0) continue;
      if (oldMatrix == null && newMatrix != null) { oldNullNewValue += 1; continue; }
      if (oldMatrix != null && newMatrix == null) {
        mismatches.push(`${entry.model_id} ${row.path}: old=${oldMatrix} new=null`);
        continue;
      }
      if (oldMatrix !== newMatrix) {
        // 已定性旧链 bug（2026-09-07）：vision 投影按文本 sequence 计数——
        // 旧 linearMacs 未传 vision/visionTokens（而旧 matmul 分支传了，旧链自身不一致）。
        // 新链按 visionTokens 计为正确语义；归入旧链问题清单，不计失败。
        if (/shared_experts/.test(row.node?.id || "")) {
          // 双链已知 bug：shared_experts 被旧 ROUTED_EXPERT_RE 误按 k/E 缩放
          // （shared 每 token 全跑，不稀疏）。extractor 已修，旧链 W5 修。
          sharedKnown += 1;
          continue;
        }
        if (row.node?.attributes?.modality === "vision") {
          knownIssues.push(`${entry.model_id} ${row.path}: old=${oldMatrix} new=${newMatrix}`);
          continue;
        }
        mismatches.push(`${entry.model_id} ${row.path}: old=${oldMatrix} new=${newMatrix}`);
      }
    }
  }
  console.error(`shared双链修正=${sharedKnown} checked=${checked} vision已知修正=${knownIssues.length} oldNull→new=${oldNullNewValue} old→newNull=${oldValueNewNull}`);
  if (knownIssues.length > 0) console.error("vision 修正样例:\n" + knownIssues.slice(0, 4).join("\n"));
  if (mismatches.length > 0) console.error("未定性差分:\n" + mismatches.slice(0, 12).join("\n"));
  assert.ok(checked > 0, "linear 节点数为 0，测试无效");
  assert.deepEqual(mismatches, []);
});
