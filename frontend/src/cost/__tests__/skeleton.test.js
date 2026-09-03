import assert from "node:assert/strict";
import test from "node:test";

import { buildSkeleton } from "../skeleton.js";

const T = (name, dtype = "BF16", shape) => ({ name, dtype, shape });

function findNode(root, id) {
  if (root.id === id) return root;
  for (const c of root.children || []) {
    const found = findNode(c, id);
    if (found) return found;
  }
  return null;
}

test("数字路径段折叠为 repeat 节点（ModuleList）", () => {
  const tensors = [
    T("model.layers.0.self_attn.q_proj.weight", "BF16", [4096, 4096]),
    T("model.layers.0.self_attn.k_proj.weight", "BF16", [1024, 4096]),
    T("model.layers.0.mlp.gate_proj.weight", "BF16", [12288, 4096]),
    T("model.layers.1.self_attn.q_proj.weight", "BF16", [4096, 4096]),
    T("model.layers.1.self_attn.k_proj.weight", "BF16", [1024, 4096]),
    T("model.layers.1.mlp.gate_proj.weight", "BF16", [12288, 4096]),
    T("model.layers.2.self_attn.q_proj.weight", "BF16", [4096, 4096]),
    T("model.layers.2.self_attn.k_proj.weight", "BF16", [1024, 4096]),
    T("model.layers.2.mlp.gate_proj.weight", "BF16", [12288, 4096]),
  ];
  const root = buildSkeleton(tensors);
  assert.equal(root.id, "model");
  const layers = findNode(root, "model.layers");
  assert.equal(layers.type, "list");
  assert.equal(layers.repeat, 3);
  // 折叠后只保留 index 0 作为代表
  assert.equal(layers.children.length, 1);
  assert.equal(layers.children[0].id, "model.layers.0");
  // 代表层的子节点：mlp / self_attn（字典序）
  assert.deepEqual(layers.children[0].children.map((c) => c.name), ["mlp", "self_attn"]);
});

test("量化三张量聚合到同一模块节点", () => {
  const tensors = [
    T("model.layers.0.self_attn.k_proj.qweight", "I32", [4096, 1024]),
    T("model.layers.0.self_attn.k_proj.qzeros", "I32", [512, 1024]),
    T("model.layers.0.self_attn.k_proj.scales", "F16", [512, 1024]),
    T("model.layers.0.self_attn.k_norm.weight", "BF16", [4096]),
  ];
  const root = buildSkeleton(tensors);
  const kProj = findNode(root, "model.layers.0.self_attn.k_proj");
  assert.ok(kProj);
  assert.deepEqual(Object.keys(kProj.weight_shapes).sort(), ["qweight", "qzeros", "scales"]);
  assert.equal(kProj.tensor_names.length, 3);
  assert.equal(kProj.dtype, "I32");
  // 元素计数 = 4096*1024 + 512*1024 + 512*1024
  assert.equal(kProj.params, 4096 * 1024 + 512 * 1024 * 2);
  // k_norm 是独立兄弟节点
  const kNorm = findNode(root, "model.layers.0.self_attn.k_norm");
  assert.deepEqual(kNorm.weight_shapes.weight, [4096]);
});

test("MoE 专家编号折叠为 repeat", () => {
  const tensors = [];
  for (let i = 0; i < 8; i++) {
    tensors.push(T(`model.layers.0.mlp.experts.${i}.gate_proj.weight`, "BF16", [512, 4096]));
    tensors.push(T(`model.layers.0.mlp.experts.${i}.down_proj.weight`, "BF16", [4096, 512]));
  }
  const root = buildSkeleton(tensors);
  const experts = findNode(root, "model.layers.0.mlp.experts");
  assert.equal(experts.type, "list");
  assert.equal(experts.repeat, 8);
  assert.deepEqual(experts.children[0].children.map((c) => c.name), ["down_proj", "gate_proj"]);
});

test("params：叶节点为自有元素计数，非叶节点为 0", () => {
  const tensors = [
    T("model.embed_tokens.weight", "BF16", [32000, 4096]),
    T("model.layers.0.self_attn.q_proj.weight", "BF16", [4096, 4096]),
  ];
  const root = buildSkeleton(tensors);
  const embed = findNode(root, "model.embed_tokens");
  assert.equal(embed.params, 32000 * 4096);
  assert.equal(findNode(root, "model.layers").params, 0);
  assert.equal(findNode(root, "model").params, 0);
});

test("数字段子树异构时不折叠，保留展开", () => {
  const tensors = [
    T("model.layers.0.mlp.gate_proj.weight", "BF16", [12288, 4096]),
    T("model.layers.1.mlp.gate_proj.weight", "BF16", [4096, 4096]), // 形状不同
  ];
  const root = buildSkeleton(tensors);
  const layers = findNode(root, "model.layers");
  assert.equal(layers.type, "module");
  assert.equal(layers.repeat, undefined);
  assert.deepEqual(layers.children.map((c) => c.name), ["0", "1"]);
});

test("非连续数字段不折叠", () => {
  const tensors = [
    T("model.layers.0.mlp.gate_proj.weight", "BF16", [12288, 4096]),
    T("model.layers.2.mlp.gate_proj.weight", "BF16", [12288, 4096]), // 缺 1
  ];
  const root = buildSkeleton(tensors);
  const layers = findNode(root, "model.layers");
  assert.equal(layers.repeat, undefined);
  assert.deepEqual(layers.children.map((c) => c.name), ["0", "2"]);
});

test("空输入返回空树", () => {
  const root = buildSkeleton([]);
  assert.equal(root.name, "root");
  assert.equal(root.children.length, 0);
});
