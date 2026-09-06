import assert from "node:assert/strict";
import test from "node:test";
import { fetchHfConfigDirect, normalizeModelId } from "./hf.js";

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
