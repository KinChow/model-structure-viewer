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

test("mtpTensorCount：空声明 / 无张量 → 0", () => {
  assert.equal(mtpTensorCount([], { hiddenLayers: 62 }), 0);
  assert.equal(mtpTensorCount([{ name: "model.layers.0.weight" }], { hiddenLayers: 62 }), 0);
  assert.equal(mtpTensorCount(null), 0);
});
