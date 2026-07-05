import assert from "node:assert/strict";
import test from "node:test";
import { catalogPath, modelConfigPath, normalizeCatalog, staticAssetPath } from "./manifest.js";

test("normalizes catalog entries for verified built-in models", () => {
  const catalog = normalizeCatalog({
    models: [
      {
        model_id: "deepseek-ai/DeepSeek-V3.1",
        revision: "abc123",
        canonical_architecture: "mla-moe-decoder",
        config_path: "deepseek-ai/DeepSeek-V3.1/config.json",
        verified: true,
      },
    ],
  });

  assert.equal(catalog.models.length, 1);
  assert.equal(catalog.models[0].modelId, "deepseek-ai/DeepSeek-V3.1");
  assert.equal(catalog.models[0].verified, true);
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
});

test("rejects catalog entries missing model id or config path", () => {
  assert.throws(
    () => normalizeCatalog({ models: [{ model_id: "bad/model" }] }),
    /config_path/,
  );
});
