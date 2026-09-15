import assert from "node:assert/strict";
import test from "node:test";
import { verifyStructureApi } from "../api/client.js";
import { formatIssue } from "../i18n/format.js";

test("verifyStructureApi 上行 /api/verify 并携带 msv_graph", async () => {
  const calls = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (path, options) => {
    calls.push({ path, body: JSON.parse(options.body) });
    return {
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        ok: true,
        status: "passed",
        evidence: { summary: { constructed: true, structurally_consistent: true }, diff: {}, modules: [] },
      }),
    };
  };
  try {
    const data = await verifyStructureApi({
      source: "builtin",
      model_id: "Qwen/Qwen3.5-0.8B",
      msv_graph: { nodes: [{ id: "root" }] },
    });
    assert.equal(calls[0].path, "/api/verify");
    assert.deepEqual(calls[0].body.msv_graph.nodes[0], { id: "root" });
    assert.equal(data.ok, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("verifyStructureApi 后端不可达时给出启动指引", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: false,
    status: 500,
    text: async () => "",
  });
  try {
    await assert.rejects(
      () => verifyStructureApi({ source: "builtin", model_id: "Qwen/Qwen3.5-0.8B" }),
      (error) => {
        assert.equal(error.issue.code, "http.verifyUnavailable");
        assert.match(formatIssue("zh", error.issue), /后端不可用/);
        assert.match(formatIssue("zh", error.issue), /msv serve/);
        assert.match(formatIssue("en", error.issue), /Backend unavailable/);
        assert.doesNotMatch(formatIssue("en", error.issue), /\p{Script=Han}/u);
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
