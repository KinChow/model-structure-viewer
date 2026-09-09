// W3-C 基线库：59 模型 normalizeConfig 输出哈希 + 方案字段快照。
// 哈希用于"删除字段后其余逐字节不变"的机械审阅；
// 方案字段快照是 plan.js 逐字搬迁的 parity oracle（搬迁前后值必须相等）。
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeConfig } from "../../config/normalize.js";
import { deriveBuildPlan } from "../plan.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");

// W3-C 迁移的方案类字段（决策逻辑 → config/plan.js）
export const PLAN_FIELDS = [
  "attentionSchedule",
  "layerSchedule",
  "indexerSchedule",
  "linearAttentionMode",
  "normMode",
  "sharedExpertsAreFused",
  "visionInternalMerger",
  "attentionOutputGate",
];

function sortDeep(value) {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).sort(([a], [b]) => (a < b ? -1 : 1)).map(([k, v]) => [k, sortDeep(v)]),
    );
  }
  return value;
}

function catalogEntries() {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  return catalog.models.map((entry) => ({
    model_id: entry.model_id,
    config: JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8")),
  }));
}

/** { model_id: 规范化输出（去 raw，raw 原样回填无意义且巨大） } */
export function buildNormalizeMap() {
  const map = {};
  for (const { model_id, config } of catalogEntries()) {
    const { raw, ...normalized } = normalizeConfig(config);
    map[model_id] = JSON.stringify(sortDeep(normalized));
  }
  return map;
}

/** { model_id: { 方案字段: 值 } } —— plan parity oracle */
export function buildPlanFixture() {
  const map = {};
  for (const { model_id, config } of catalogEntries()) {
    const normalized = normalizeConfig(config);
    map[model_id] = Object.fromEntries(PLAN_FIELDS.map((field) => [field, normalized[field] ?? null]));
  }
  return map;
}

export function hashJson(json) {
  return createHash("sha256").update(json).digest("hex").slice(0, 16);
}

/** { model_id: deriveBuildPlan(config) 的方案字段 } —— parity 对比的现算侧 */
export function buildPlanFromDerive() {
  const map = {};
  for (const { model_id, config } of catalogEntries()) {
    map[model_id] = Object.fromEntries(PLAN_FIELDS.map((field) => [field, deriveBuildPlan(config)[field] ?? null]));
  }
  return map;
}
