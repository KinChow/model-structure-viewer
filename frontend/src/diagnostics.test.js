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

test("structureStatus 区分 checkpoint 骨架真值与模板合并真值", () => {
  assert.deepEqual(structureStatus({ summary: { strategy: "skeleton-truth" } }), {
    label: "Checkpoint 骨架真值",
    tone: "truth",
    detail: "Checkpoint-derived module tree",
  });
  assert.deepEqual(structureStatus({ summary: { strategy: "template+truth" } }), {
    label: "模板 + checkpoint 真值",
    tone: "truth",
    detail: "Template semantics with checkpoint values",
  });
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
