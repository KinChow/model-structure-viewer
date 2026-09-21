import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { catalogPath, headerTruthPath, modelConfigPath, normalizeCatalog, staticAssetPath } from "./manifest.js";
import { sortModelsByReleaseTime } from "./modelOrdering.js";

test("all 60 built-in models have an ISO UTC release snapshot; V4.1 is newest DeepSeek", () => {
  const raw = JSON.parse(readFileSync(new URL("../../../../models/catalog.json", import.meta.url)));
  assert.equal(raw.models.length, 60);
  for (const model of raw.models) {
    assert.match(model.release_time || "", /^\d{4}-\d{2}-\d{2}T.*Z$/, model.model_id);
    assert.ok(Number.isFinite(Date.parse(model.release_time)), model.model_id);
  }
  const deepseek = normalizeCatalog(raw).models.filter((model) => model.modelId.startsWith("deepseek-ai/"));
  assert.equal(sortModelsByReleaseTime(deepseek)[0].modelId, "deepseek-ai/DeepSeek-V4.1-Flash");
});

test("normalizes catalog entries for verified built-in models", () => {
  const catalog = normalizeCatalog({
    models: [
      {
        model_id: "deepseek-ai/DeepSeek-V3.1",
        display_name: "DeepSeek V3.1",
        release_time: "2026-01-02T08:00:00Z",
        revision: "abc123",
        architectures: ["DeepseekV3ForCausalLM"],
        config_path: "deepseek-ai/DeepSeek-V3.1/config.json",
      },
    ],
  });

  assert.equal(catalog.models.length, 1);
  assert.equal(catalog.models[0].modelId, "deepseek-ai/DeepSeek-V3.1");
  assert.equal(catalog.models[0].displayName, "DeepSeek V3.1");
  assert.equal(catalog.models[0].releaseTime, "2026-01-02T08:00:00Z");
  assert.equal(modelConfigPath(catalog.models[0]), "/models/deepseek-ai/DeepSeek-V3.1/config.json");
});

test("uses Vite base path for static deployment assets", () => {
  assert.equal(staticAssetPath("models/catalog.json", "/model-structure-viewer/"), "/model-structure-viewer/models/catalog.json");
  assert.equal(catalogPath("/model-structure-viewer/"), "/model-structure-viewer/models/catalog.json");
  assert.equal(
    modelConfigPath(
      {
        configPath: "Qwen/Qwen3.5-0.8B/config.json",
      },
      "/model-structure-viewer/",
    ),
    "/model-structure-viewer/models/Qwen/Qwen3.5-0.8B/config.json",
  );
  assert.equal(
    headerTruthPath({ configPath: "Qwen/Qwen3.5-0.8B/config.json" }, "/model-structure-viewer/"),
    "/model-structure-viewer/models/Qwen/Qwen3.5-0.8B/header-truth.json",
  );
});

test("rejects catalog entries missing model id or config path", () => {
  assert.throws(
    () => normalizeCatalog({ models: [{ model_id: "bad/model" }] }),
    /config_path/,
  );
});
