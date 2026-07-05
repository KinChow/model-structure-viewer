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

test("structureStatus explains budget config fallback with compact budget detail", () => {
  const status = structureStatus({
    summary: {
      strategy: "budget-config-fallback",
      fallback_reason: "resource budget exceeded for meta introspection",
      confidence: "low",
    },
    source: {
      diagnostics: {
        failure_kind: "resource_budget_exceeded",
        budget: {
          layers: 78,
          hidden_size: 6144,
          experts: 256,
          score: 1757184,
          score_limit: 400000,
        },
      },
    },
  });

  assert.equal(status.label, "Config fallback");
  assert.equal(status.tone, "warn");
  assert.equal(status.detail, "Budget exceeded: layers 78, hidden 6144, experts 256");
});
