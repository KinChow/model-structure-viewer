import assert from "node:assert/strict";
import test from "node:test";
import { fetchHfConfigDirect, normalizeModelId, revisionForEndpoint, HF_ENDPOINTS } from "./hf.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

test("normalizes Hugging Face and ModelScope URLs to repo ids", () => {
  assert.equal(normalizeModelId("https://huggingface.co/Qwen/Qwen3.5-0.8B", "huggingface"), "Qwen/Qwen3.5-0.8B");
  assert.equal(normalizeModelId("https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B", "modelscope"), "Qwen/Qwen3.5-0.8B");
});

test("fetches config with normalized repo URL and encoded revision", async () => {
  let requested;
  await fetchHfConfigDirect({
    modelId: "https://huggingface.co/Qwen/Qwen3.5-0.8B",
    revision: "feature/test",
    fetchImpl: async (url) => {
      requested = url;
      return { ok: true, json: async () => ({ model_type: "qwen" }) };
    },
  });
  assert.equal(requested, "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/feature%2Ftest/config.json");
});

test("source contract: revision defaults and error classes match Python", () => {
  const contractPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../docs/details/models/source_contract.json");
  const contract = JSON.parse(readFileSync(contractPath, "utf8"));
  assert.deepEqual(contract.sources, ["auto", "builtin", "local", "hf", "config"]);
  assert.deepEqual(contract.auto_fallback, ["builtin", "local", "hf"]);
  assert.deepEqual(contract.frontend.sources, ["builtin", "hf", "config"]);
  assert.deepEqual(contract.frontend.legacy_auto_fallback, ["builtin", "hf"]);
  assert.equal(contract.endpoints.huggingface.default_revision, HF_ENDPOINTS.huggingface.defaultRevision);
  assert.equal(contract.endpoints.modelscope.default_revision, HF_ENDPOINTS.modelscope.defaultRevision);
  assert.equal(revisionForEndpoint("huggingface"), "main");
  assert.equal(revisionForEndpoint("modelscope", "main"), "master");
  assert.equal(revisionForEndpoint("modelscope", "v1"), "v1");
  assert.equal(contract.errors.config, 400);
  assert.equal(contract.errors.not_found, 404);
  assert.equal(contract.errors.remote, 502);
  assert.deepEqual(contract.cache_key, ["repo_id", "revision", "cache_dir"]);
  assert.equal(contract.graph_protocol.schema_version, 2);
});

test("HF search failures stay on the public Hub and retain a readable error", async () => {
  const { searchHfDirect } = await import("./hf.js");
  const calls = [];
  for (const fail of [async () => { throw new TypeError("Failed to fetch"); }, async () => ({ ok: false, status: 503 })]) {
    await assert.rejects(searchHfDirect("example", 10, "huggingface", async (url) => { calls.push(url); return fail(); }),
      (error) => error.issue.code === "model.searchFailed");
  }
  assert.equal(calls.length, 2);
  assert.ok(calls.every((url) => url.startsWith("https://huggingface.co/api/models?")));
});
