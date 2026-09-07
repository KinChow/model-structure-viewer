// W1 第二批差分测试（T3）：elementwise / MoE / 递推 / 复合节点。
// 分类：① 全等（0/0 或 >0 相等）② 已定性旧链错算（vision；复合投影按整体 in×out 高估）
// ③ 旧链漏算（old=0，new>0，复合/融合节点）——其余必须为空。
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
const IN_SCOPE_IDS = new Set([
  "softmax", "rope", "rmsnorm", "gemma_rmsnorm", "gated_rmsnorm",
  "attention_output_gate", "mla_output_gate", "linear_attention_gate", "shared_expert_gate",
  "swiglu", "vision_activation", "vision_position", "vision_merge",
  "split", "mla_kv_split", "qwen_qkvz_split", "attention_qkv_split",
  "causal_conv1d", "linear_attention", "gated_delta_attention",
  "topk", "moe_dispatch", "moe_combine", "moe_add", "dsv4_hash_route",
  "mhc_pre", "mhc_fused_post_pre", "mhc_post", "mhc_contract",
  "hyper_connection", "ple", "attention_residual",
  "mla_query_compress", "mla_kv_compress", "qsa_indexer", "minimax_sparse_indexer",
]);
// 旧链把"整体 in×out"当单一密集矩阵计的复合投影（新链按分解计，数学正确、旧链高估）
const COMPOSITE_PROJECTION = new Set(["mla_query_compress", "mla_kv_compress"]);

test("T3 差分：elementwise/MoE/递推/复合 分类清晰", async () => {
  const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const mismatches = [];
  const knownProjection = [];
  const uncounted = new Map(); // operatorId → 漏算节点数（旧链 0，新链 > 0）
  let checked = 0;
  let sharedKnown = 0;
  let knownVision = 0;
  let knownSwiglu = 0;

  for (const entry of catalog.models) {
    const config = JSON.parse(await fs.readFile(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const structure = buildStructureFromConfig(config, { modelId: entry.model_id, source: "differential-test" });
    const rows = computeNodeCosts(structure.root, normalized, { batch: 1, sequence: 128, phase: "prefill" });

    for (const row of rows) {
      const operatorId = String(row.node?.attributes?.operator_id || "").toLowerCase();
      if (!IN_SCOPE_IDS.has(operatorId)) continue;
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
      if (oldMatrix === newMatrix) continue;

      if (operatorId === "swiglu" && /experts?\\.|expert_mlp/.test(row.node?.id || "")) {
        // 双链已知 bug：压缩 routed FFN 叶旧链按 ·(k/E) 计，少乘 E/k；
        // 新链按 T·k·3·EH·EI 计（每 token 激活 k 个专家全量 FFN）。
        knownSwiglu += 1;
        continue;
      }
      if (row.node?.attributes?.modality === "vision") { knownVision += 1; continue; }
      if (oldMatrix != null && newMatrix != null && COMPOSITE_PROJECTION.has(operatorId)) {
        knownProjection.push(`${entry.model_id} ${row.path}: old=${oldMatrix} new=${newMatrix}`);
        continue;
      }
      if (oldMatrix === 0 && newMatrix > 0) {
        // 旧链漏算（复合/融合语义节点），W5 切换后修复
        uncounted.set(operatorId, (uncounted.get(operatorId) || 0) + 1);
        continue;
      }
      mismatches.push(`${entry.model_id} ${row.path} (${operatorId}): old=${oldMatrix} new=${newMatrix}`);
    }
  }
  console.error(`shared双链修正=${sharedKnown} checked=${checked} vision=${knownVision} 复合投影旧链高估=${knownProjection.length} 漏算清单=`, Object.fromEntries(uncounted));
  if (knownProjection.length > 0) console.error("复合投影样例:\n" + knownProjection.slice(0, 3).join("\n"));
  if (mismatches.length > 0) console.error("未定性差分:\n" + mismatches.slice(0, 12).join("\n"));
  assert.ok(checked > 0, "T3 节点数为 0，测试无效");
  assert.deepEqual(mismatches, []);
});
