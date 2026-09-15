import assert from "node:assert/strict";
import test from "node:test";
import { structureStatus } from "./diagnostics.js";

test("structureStatus explains successful meta introspection", () => {
  const status = structureStatus({
    summary: { strategy: "meta-introspect", confidence: "high" },
    source: {},
  }, "en");

  assert.equal(status.label, "Meta introspect");
  assert.equal(status.tone, "ok");
  assert.equal(status.detail, "Live module tree");
});

test("structureStatus explains frontend architecture template output", () => {
  const status = structureStatus({
    summary: { strategy: "frontend-architecture-template" },
    source: { diagnostics: { architecture: "DeepseekV3ForCausalLM" } },
  }, "en");

  assert.equal(status.label, "Frontend template");
  assert.equal(status.tone, "ok");
  assert.equal(status.detail, "Config-driven frontend structure");
});

test("structureStatus reports when checkpoint truth falls back to config", () => {
  const status = structureStatus({
    summary: { strategy: "frontend-architecture-template" },
    source: { checkpoint_truth: "unavailable" },
  }, "en");

  assert.equal(status.detail, "Config-driven frontend structure; checkpoint metadata unavailable, using config-only structure");
});

test("structureStatus 区分 checkpoint 骨架真值与模板合并真值", () => {
  assert.deepEqual(structureStatus({ summary: { strategy: "skeleton-truth" } }, "zh"), {
    label: "Checkpoint 骨架真值",
    tone: "truth",
    detail: "由 checkpoint 得到的模块树",
  });
  assert.deepEqual(structureStatus({ summary: { strategy: "template+truth" } }, "zh"), {
    label: "模板 + checkpoint 真值",
    tone: "truth",
    detail: "模板语义，值为 checkpoint",
  });
  assert.deepEqual(structureStatus({ summary: { strategy: "template+header-truth" } }, "zh"), {
    label: "模板 + header 总量",
    tone: "truth",
    detail: "模板语义，总量来自 safetensors header",
  });
});

test("structureStatus English labels contain no Han", () => {
  const status = structureStatus({ summary: { strategy: "skeleton-truth" } }, "en");
  assert.equal(status.label, "Checkpoint skeleton truth");
  assert.doesNotMatch(status.label, /\p{Script=Han}/u);
  assert.doesNotMatch(status.detail, /\p{Script=Han}/u);
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
  }, "en");

  assert.equal(status.label, "Meta introspect");
  assert.equal(status.tone, "ok");
  assert.equal(status.detail, "Repaired by minimax_config_adapter");
});
