// W3-C 基线库：59 模型 normalizeConfig 输出哈希 + 方案字段快照。
// 哈希用于"删除字段后其余逐字节不变"的机械审阅；
// 方案字段快照是调度函数搬迁的 parity oracle（搬迁前后值必须相等）。
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeConfig } from "../../config/normalize.js";
import { attentionScheduleOf, indexerScheduleOf, layerScheduleOf } from "../../layers/schedule.js";
import {
  recipeAttentionOutputGate,
  recipeLinearAttentionMode,
  recipeNormMode,
  recipeSharedExpertsAreFused,
  recipeVisionInternalMerger,
} from "../../archs/index.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");

// 组网调度字段（决策逻辑 → layers/schedule.js）
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

/** { model_id: 调度函数 + recipe* 快照 } —— 与 plan-fixture 字段同名，便于逐值对。 */
export function buildPlanFromDerive() {
  const map = {};
  for (const { model_id, config } of catalogEntries()) {
    const snapshot = {
      attentionSchedule: attentionScheduleOf(config) ?? null,
      layerSchedule: layerScheduleOf(config) ?? null,
      indexerSchedule: indexerScheduleOf(config) ?? null,
      linearAttentionMode: recipeLinearAttentionMode(config) ?? null,
      normMode: recipeNormMode(config) ?? null,
      sharedExpertsAreFused: recipeSharedExpertsAreFused(config) ?? null,
      visionInternalMerger: recipeVisionInternalMerger(config) ?? null,
      attentionOutputGate: recipeAttentionOutputGate(config) ?? null,
    };
    map[model_id] = snapshot;
  }
  return map;
}
