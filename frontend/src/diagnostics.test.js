import assert from "node:assert/strict";
import test from "node:test";

import { structureStatus } from "./diagnostics.js";

test("structureStatus explains successful meta introspection", () => {
  const status = structureStatus({
    summary: { strategy: "meta-introspect", confidence: "high" },
    source: {},
  });

  assert.equal(status.label, "Meta introspect");
  assert.equal(status.tone, "ok");
  assert.equal(status.detail, "Live module tree");
});

test("structureStatus explains frontend architecture template output", () => {
  const status = structureStatus({
    summary: { strategy: "frontend-architecture-template" },
    source: { diagnostics: { canonical_architecture: "mla-moe-decoder" } },
  });

  assert.equal(status.label, "Frontend template");
  assert.equal(status.tone, "ok");
  assert.equal(status.detail, "Config-driven frontend structure");
});

test("structureStatus explains repaired meta introspection", () => {
  const status = structureStatus({
    summary: {
      strategy: "repaired-meta-introspect",
      confidence: "high",
    },
    source: {
      diagnostics: {
        repair_strategy: "minimax_config_adapter",
      },
    },
  });

  assert.equal(status.label, "Meta introspect");
  assert.equal(status.tone, "ok");
  assert.equal(status.detail, "Repaired by minimax_config_adapter");
});
