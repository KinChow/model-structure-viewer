// DeepSeek-V4.1-Flash 官方 CED 呈现回归：20 层因果编码器 + 20 层解码器、
// CSA2(ratio, mode) 折叠命名、候选池标注、以及编码器→解码器的全局 KV 关系边。
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { buildStructureFromConfig } from "../buildStructure.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const config = JSON.parse(
  fs.readFileSync(path.join(repoRoot, "models/deepseek-ai/DeepSeek-V4.1-Flash/config.json"), "utf8"),
);

function build() {
  return buildStructureFromConfig(config, { modelId: "deepseek-ai/DeepSeek-V4.1-Flash" }).graph;
}

function childrenOf(graph, id) {
  return graph.nodes.filter((node) => {
    const p = node.path || node.id;
    return p !== id && p.split(".").slice(0, -1).join(".") === id;
  });
}

test("V4.1-Flash 呈现为 CED 两段：20 层编码器 + 20 层解码器", () => {
  const graph = build();
  const top = childrenOf(graph, graph.root_id || "root");
  const encoder = top.find((n) => n.type === "encoder");
  const decoder = top.find((n) => n.type === "decoder");
  assert.ok(encoder, "存在 Causal Encoder 段");
  assert.ok(decoder, "存在 Decoder 段");
  assert.equal(encoder.name, "Causal Encoder");
  assert.equal(decoder.name, "Decoder");
  assert.equal(encoder.attributes?.num_hidden_layers, 20);
  assert.equal(decoder.attributes?.num_hidden_layers, 20);
  assert.equal(top.filter((n) => n.type === "decoder").length, 1);
});

test("V4.1-Flash 折叠组按 CSA2(ratio, mode) 命名，且层号 0..39 不重不漏", () => {
  const graph = build();
  const encoderId = graph.nodes.find((n) => n.type === "encoder").id;
  const decoderId = graph.nodes.find((n) => n.type === "decoder").id;
  const groups = [...childrenOf(graph, encoderId), ...childrenOf(graph, decoderId)];
  const names = groups.map((g) => g.name);
  assert.ok(names.some((n) => n.includes("(SWA)")), "编码器含纯滑窗层");
  assert.ok(names.some((n) => n.includes("CSA2(2, Full)")), "编码器含 CSA2(2, Full)");
  assert.ok(names.some((n) => n.includes("CSA2(2, Reuse)")), "编码器含 CSA2(2, Reuse)");
  assert.ok(names.some((n) => n.includes("CSA2(1, Full)")), "解码器含 CSA2(1, Full)");
  assert.ok(names.some((n) => n.includes("CSA2(1, Reindex)")), "解码器含 CSA2(1, Reindex)");
  const covered = [];
  for (const g of groups) {
    const [a, b] = String(g.attributes?.range || "").split("..").map(Number);
    for (let i = a; i <= b; i += 1) covered.push(i);
  }
  covered.sort((x, y) => x - y);
  assert.deepEqual(covered, Array.from({ length: 40 }, (_, i) => i));
});

test("V4.1-Flash 候选池：首个解码 Full 层建池，其后 Reindex 层受约束", () => {
  const graph = build();
  const decoderId = graph.nodes.find((n) => n.type === "decoder").id;
  const groups = childrenOf(graph, decoderId);
  const source = groups.find((g) => g.attributes?.candidate_pool_source);
  assert.ok(source, "存在候选池源层");
  assert.equal(source.attributes?.range, "20..20");
  assert.equal(source.attributes?.csa2_mode, "full");
  assert.ok(groups.some((g) => g.attributes?.candidate_constrained), "存在受候选池约束的 Reindex 层");
});

test("V4.1-Flash 存在编码器→解码器的全局 KV 关系边", () => {
  const graph = build();
  const encoderId = graph.nodes.find((n) => n.type === "encoder").id;
  const decoderId = graph.nodes.find((n) => n.type === "decoder").id;
  const cedEdge = (graph.edges || []).find(
    (e) => e.source === encoderId && e.target === decoderId && e.relation === "kv-projection",
  );
  assert.ok(cedEdge, "存在 kv-projection 关系边");
  assert.equal(cedEdge.kind, "dataflow");
});

test("V4.1-Flash 顶层 dataflow_edges 为纯 [from,to] 二元组（不泄漏关系对象，避免 [object Object]）", () => {
  const graph = build();
  const root = graph.nodes.find((n) => n.id === (graph.root_id || "root"));
  const dfe = root?.attributes?.dataflow_edges || [];
  assert.ok(dfe.length > 0, "存在顶层 dataflow_edges");
  for (const pair of dfe) {
    assert.ok(Array.isArray(pair) && pair.length === 2, "每条声明是二元组");
    assert.ok(pair.every((ref) => typeof ref === "string"), "端点均为字符串引用");
  }
  // 关系标注走单独属性，不进 dataflow_edges。
  const rels = root?.attributes?.dataflow_edge_relations || [];
  assert.ok(rels.some((r) => r.relation === "kv-projection"), "关系标注在 dataflow_edge_relations 中");
});

test("V4.1-Flash 每个模块的子节点都至少参与一条兄弟连线（无悬空瓷砖，含 compressor/indexer）", () => {
  const graph = build();
  const parentOf = (id) => { const i = id.lastIndexOf("."); return i < 0 ? null : id.slice(0, i); };
  const kidsByParent = new Map();
  for (const n of graph.nodes) {
    const par = parentOf(n.id);
    if (!par) continue;
    if (!kidsByParent.has(par)) kidsByParent.set(par, []);
    kidsByParent.get(par).push(n);
  }
  const orphans = [];
  for (const [par, kids] of kidsByParent) {
    if (kids.length < 2) continue;
    const sib = (graph.edges || []).filter((e) => parentOf(e.source) === par && parentOf(e.target) === par);
    const touched = new Set();
    sib.forEach((e) => { touched.add(e.source); touched.add(e.target); });
    for (const k of kids) if (!touched.has(k.id)) orphans.push(`${graph.nodes.find((n) => n.id === par)?.name} :: ${k.name}`);
  }
  assert.deepEqual(orphans, [], `存在无连线的悬空子节点：\n  ${orphans.join("\n  ")}`);
});

test("无 CED 分界证据的 dsv4 模型仍走单一解码栈", () => {
  const v4 = {
    architectures: ["DeepseekV4ForCausalLM"],
    model_type: "deepseek_v4",
    text_config: {
      vocab_size: 1000, hidden_size: 128, num_hidden_layers: 4, num_attention_heads: 8,
      num_key_value_heads: 1, head_dim: 64, moe_intermediate_size: 64,
      compress_ratios: [0, 4, 4, 128], sliding_window: 128,
    },
  };
  const graph = buildStructureFromConfig(v4, { modelId: "deepseek-ai/DeepSeek-V4-toy" }).graph;
  const top = childrenOf(graph, graph.root_id || "root");
  assert.equal(top.filter((n) => n.type === "encoder").length, 0);
  assert.equal(top.filter((n) => n.type === "decoder").length, 1);
});
