import assert from "node:assert/strict";
import test from "node:test";
import { mtpTensorCount } from "../weights.js";

test("mtpTensorCount：mtp.{i} 前缀（V4 / MiniMax / Qwen）", () => {
  const tensors = [
    { name: "model.layers.0.mlp.down_proj.weight" },
    { name: "mtp.0.eh_proj.weight" },
    { name: "mtp.0.layer.self_attn.q_proj.weight" },
    { name: "lm_head.weight" },
  ];
  assert.equal(mtpTensorCount(tensors, { hiddenLayers: 43 }), 2);
});

test("mtpTensorCount：越界 layers.{n}（V3 把 MTP 编进层号）", () => {
  const tensors = [
    { name: "model.layers.60.mlp.down_proj.weight" },
    { name: "model.layers.61.eh_proj.weight" },
    { name: "model.layers.61.shared_head.norm.weight" },
  ];
  assert.equal(mtpTensorCount(tensors, { hiddenLayers: 61 }), 2);
  assert.equal(mtpTensorCount(tensors, { hiddenLayers: 62 }), 0);
});

test("mtpTensorCount：DSpark 的 mtp.{0,1,2} 都计入，主干 layers.40 不计", () => {
  const tensors = [
    { name: "layers.40.attn.wq_a.weight" },
    { name: "mtp.0.main_proj.weight" },
    { name: "mtp.1.attn.wq_a.weight" },
    { name: "mtp.2.markov_head.markov_w1.weight" },
  ];
  assert.equal(mtpTensorCount(tensors, { hiddenLayers: 43 }), 3);
});
test("mtpTensorCount：空声明 / 无张量 → 0", () => {
  assert.equal(mtpTensorCount([], { hiddenLayers: 62 }), 0);
  assert.equal(mtpTensorCount([{ name: "model.layers.0.weight" }], { hiddenLayers: 62 }), 0);
  assert.equal(mtpTensorCount(null), 0);
});
