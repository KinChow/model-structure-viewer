// W2 差分基线库：构建全部内置模型的 spec 树规范化 JSON 与哈希。
// 规范化 = 键递归排序 + minify；哈希用于紧凑存储，失败时由测试 dump 全文供 diff。
// 注意：仅剥离 FORMULAS 注册表注入的 explanation/inputs/outputs 大文本（由
// operatorId 决定，与 ops/index.js 重构无关），其余逐字节保真。
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../../buildStructure.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");

function sortDeep(value) {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).sort(([a], [b]) => (a < b ? -1 : 1)).map(([k, v]) => [k, sortDeep(v)]),
    );
  }
  return value;
}

function stripRegistryTexts(node) {
  if (Array.isArray(node)) return node.map(stripRegistryTexts);
  if (!node || typeof node !== "object") return node;
  const out = {};
  for (const [k, v] of Object.entries(node)) {
    if (k === "attributes" && v && typeof v === "object") {
      out[k] = Object.fromEntries(
        Object.entries(v).filter(([ak]) => ak !== "explanation" && ak !== "inputs" && ak !== "outputs"),
      );
    } else {
      out[k] = stripRegistryTexts(v);
    }
  }
  return out;
}

export function buildSpecTreeMap() {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const map = {};
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const structure = buildStructureFromConfig(config, { modelId: entry.model_id, source: "spec-golden" });
    map[entry.model_id] = JSON.stringify(stripRegistryTexts(sortDeep(structure.root)));
  }
  return map;
}

export function hashSpecTree(json) {
  return createHash("sha256").update(json).digest("hex").slice(0, 16);
}
