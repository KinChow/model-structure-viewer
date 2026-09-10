// M11 bytes 完整性棘轮（§3.1c）：全部内置模型的全部 leaf 算子，其动作向量
// 的访存三分量不得全为零——除非该算子在 VIEW_OPS 显式豁免（strided view
// 无拷贝语义，零是精确陈述）。
// 背景：W5-1 手搓 switch 从旧 MACs 链镜像迁移时，内联 return 的 case 把
// bytes 写成占位零，而恒等式只锚 matrix、bytes 又无消费者，占位零存活
// 至 M11 才被四路审计发现。本测试保证"未建模的零"不再可能静默混入。
import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { normalizeConfig } from "../../structure/config/normalize.js";
import { resolveArchitecture } from "../../structure/registry/resolveArchitecture.js";
import { buildNetwork } from "../../structure/model_executor/models/index.js";
import { createStructureIr } from "../../structure/ir/createStructureIr.js";
import { materializeModelStructure } from "../../structure/materializers/modelStructure.js";
import { computeNodeCosts } from "../compute.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");

// view 语义（无拷贝）：显式豁免清单。新增豁免必须在此登记并写明理由。
const VIEW_OPS = new Set(["split", "mla_kv_split", "qwen_qkvz_split", "attention_qkv_split"]);

// 未建模登记（照 identity REGISTERED 惯例）：新算子接入时允许临时登记，
// 必须写明跟踪位置并尽快补齐。2026-09-08 方案 A 落地后此前登记的
// qsa_attention / dsv4_swa / dsv4_compressed 已全部补齐清空。
const PENDING_UNMODELED = new Set([]);

test("bytes 完整性：全部 leaf 算子的访存分量不得全零（view 豁免除外）", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const offenders = new Map();
  for (const entry of catalog.models) {
    const rawConfig = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(rawConfig);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    // P7（步骤 7）：computeNodeCosts 首参即 Graph IR，不再经 options.graph 透传。
    const rows = computeNodeCosts(structure.graph, normalized, { batch: 1, sequence: 128, phase: "prefill" });
    for (const row of rows) {
      if (!row.actions) continue;
      const op = String(row.node?.attributes?.operator_id || row.node?.type || "").toLowerCase();
      if (VIEW_OPS.has(op) || PENDING_UNMODELED.has(op)) continue;
      // W4：乘子为 0 的子树（MTP 在投机解码未启用时 repeat=0，见
      // model_executor/layers/mtp.js 口径声明）本来就该聚合成零，
      // 不是「未建模的零」。判据是显式的 multiplier===0，不是白名单。
      if (row.multiplier === 0) continue;
      const { weights, actIn, actOut } = row.actions.bytes;
      if (!(weights > 0 || actIn > 0 || actOut > 0)) {
        if (!offenders.has(op)) offenders.set(op, new Set());
        offenders.get(op).add(entry.model_id);
      }
    }
  }
  assert.deepEqual(
    [...offenders.keys()].sort(),
    [],
    `以下算子存在全零访存分量（未建模的零）：\n${[...offenders.entries()]
      .map(([op, models]) => `  ${op}: ${[...models].slice(0, 3).join(", ")}${models.size > 3 ? ` 等 ${models.size} 个模型` : ""}`)
      .join("\n")}`,
  );
});
