import assert from "node:assert/strict";
import test from "node:test";

import { readLocalSafetensorsHeaders, readSafetensorsHeaders } from "../safetensorsReader.js";

/** 构造一个假 safetensors 文件字节：8 字节小端 u64 长度 + JSON。 */
function fakeSafetensorsBytes(headerObj) {
  const json = JSON.stringify(headerObj);
  const bytes = new TextEncoder().encode(json);
  const len = new Uint8Array(8);
  let n = BigInt(bytes.length);
  for (let i = 0; i < 8; i++) {
    len[i] = Number(n & 0xffn);
    n >>= 8n;
  }
  const out = new Uint8Array(8 + bytes.length);
  out.set(len, 0);
  out.set(bytes, 8);
  return out;
}

/** 构造 mock fetch：按 URL 路由返回假文件/JSON。 */
function mockFetch({ files = {}, index = null, base }) {
  return async (url, opts = {}) => {
    const u = new URL(url);
    const path = u.pathname;
    const range = opts.headers?.Range;
    if (path.endsWith("model.safetensors.index.json")) {
      if (index === null) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0), json: async () => null };
      return { ok: true, status: 200, json: async () => index };
    }
    const name = path.split("/").pop();
    const file = files[name];
    if (!file) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
    if (range) {
      const [, spec] = range.split("=");
      const [start, end] = spec.split("-").map(Number);
      return {
        ok: true,
        status: 206,
        arrayBuffer: async () => file.slice(start, end + 1).buffer,
      };
    }
    return { ok: true, status: 200, arrayBuffer: async () => file.buffer };
  };
}

test("单文件：读 header 结构、逐 dtype 参数量、总参数量", async () => {
  const header = {
    "model.embed_tokens.weight": { dtype: "BF16", shape: [32000, 1024] },
    "model.layers.0.mlp.gate_proj.weight": { dtype: "BF16", shape: [4096, 1024] },
    "model.layers.0.mlp.gate_proj.qweight": { dtype: "I32", shape: [1024, 1024] },
    "__metadata__": { total_parameters: "1" },
  };
  const file = fakeSafetensorsBytes(header);
  const fetchImpl = mockFetch({ files: { "model.safetensors": file }, base: "" });

  const result = await readSafetensorsHeaders({
    modelId: "Qwen/Qwen2.5-0.5B",
    revision: "master",
    hubUrl: "https://www.modelscope.cn",
    resolvePrefix: "/models",
    fetchImpl,
  });

  // __metadata__ 被排除
  assert.equal(result.tensors.length, 3);
  assert.equal(result.parameterCount.BF16, 32000 * 1024 + 4096 * 1024);
  assert.equal(result.parameterCount.I32, 1024 * 1024);
  assert.equal(result.parameterTotal, 32000 * 1024 + 4096 * 1024 + 1024 * 1024);
  const gate = result.tensors.find((t) => t.name.endsWith("gate_proj.qweight"));
  assert.deepEqual(gate.shape, [1024, 1024]);
});

test("分片：按 index 读多个 shard 并合并", async () => {
  const shard0 = fakeSafetensorsBytes({
    "model.embed_tokens.weight": { dtype: "BF16", shape: [32000, 1024] },
  });
  const shard1 = fakeSafetensorsBytes({
    "model.layers.0.mlp.gate_proj.weight": { dtype: "BF16", shape: [4096, 1024] },
  });
  const index = {
    weight_map: {
      "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
      "model.layers.0.mlp.gate_proj.weight": "model-00002-of-00002.safetensors",
    },
  };
  const fetchImpl = mockFetch({
    files: {
      "model-00001-of-00002.safetensors": shard0,
      "model-00002-of-00002.safetensors": shard1,
    },
    index,
  });

  const result = await readSafetensorsHeaders({
    modelId: "x/y",
    revision: "main",
    hubUrl: "https://huggingface.co",
    resolvePrefix: "",
    fetchImpl,
  });
  assert.equal(result.tensors.length, 2);
  assert.equal(result.parameterTotal, 32000 * 1024 + 4096 * 1024);
});

test("header 长度字段不足 8 字节时报错", async () => {
  const fetchImpl = async () => ({
    ok: true,
    status: 206,
    arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
  });
  await assert.rejects(
    () =>
      readSafetensorsHeaders({
        modelId: "x/y",
        revision: "main",
        hubUrl: "https://huggingface.co",
        resolvePrefix: "",
        fetchImpl,
      }),
    /range response is truncated/,
  );
});

test("Range 不被支持时拒绝读取完整权重响应", async () => {
  let arrayBufferCalled = false;
  let bodyCancelled = false;
  const fetchImpl = async (url, opts = {}) => {
    const range = opts.headers?.Range;
    if (range) return {
      ok: true,
      status: 200,
      body: { cancel: async () => { bodyCancelled = true; } },
      arrayBuffer: async () => { arrayBufferCalled = true; return new ArrayBuffer(0); },
    };
    return { ok: false, status: 404 };
  };
  await assert.rejects(
    () => readSafetensorsHeaders({ modelId: "x/y", fetchImpl }),
    /does not support byte ranges/,
  );
  assert.equal(arrayBufferCalled, false);
  assert.equal(bodyCancelled, true);
});

test("拒绝超过成熟库上限的 safetensors header", async () => {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setBigUint64(0, 25_000_001n, true);
  const fetchImpl = async (url, opts = {}) => {
    if (!opts.headers?.Range) return { ok: false, status: 404 };
    return { ok: true, status: 206, arrayBuffer: async () => bytes.buffer };
  };
  await assert.rejects(
    () => readSafetensorsHeaders({ modelId: "x/y", fetchImpl }),
    /exceeds 25000000 bytes/,
  );
});

test("浏览器本地目录：按 safetensors index 读取分片 header", async () => {
  const shard0 = fakeSafetensorsBytes({ "model.embed.weight": { dtype: "BF16", shape: [4, 8] } });
  const shard1 = fakeSafetensorsBytes({ "model.lm_head.weight": { dtype: "F32", shape: [8, 2] } });
  const makeFile = (name, bytes) => ({ name, text: async () => "", slice: (start, end) => ({ arrayBuffer: async () => bytes.slice(start, end).buffer }) });
  const index = { weight_map: { "model.embed.weight": "model-00001-of-00002.safetensors", "model.lm_head.weight": "model-00002-of-00002.safetensors" } };
  const indexFile = { name: "model.safetensors.index.json", text: async () => JSON.stringify(index) };
  const result = await readLocalSafetensorsHeaders([
    indexFile,
    makeFile("model-00001-of-00002.safetensors", shard0),
    makeFile("model-00002-of-00002.safetensors", shard1),
  ]);
  assert.equal(result.tensors.length, 2);
  assert.equal(result.parameterCount.BF16, 32);
  assert.equal(result.parameterCount.F32, 16);
  assert.equal(result.parameterTotal, 48);
});
