import assert from "node:assert/strict";
import test from "node:test";
import { classifyRoofline } from "../cost/roofline.js";
import { buildNodeLens } from "./lens.js";

test("roofline 按动作向量分类 bound（M11-P0-4 迁移到 actions 形状）", () => {
  const result = classifyRoofline({
    actions: { matrix: 10, vector: 0, sfu: 0, bytes: { weights: 100, actIn: 0, actOut: 0 } },
  }, {
    memory_bandwidth: 100,
    peak_flops: { bf16: 1000 },
    interconnect: { intra_node: { bandwidth: 100 } },
  });
  assert.equal(result.bound, "memory");
});

test("节点 lens 使用校验后的逐卡并行投影", () => {
  const structure = {
    extra_config: { hidden_size: 4, num_attention_heads: 1, num_hidden_layers: 1 },
    graph: {
      root_id: "root",
      nodes: [
        { id: "root", canonical_id: "model", type: "model", parent_id: null, order: 0 },
        { id: "root.0", canonical_id: "model.layers.0.self_attn.o_proj", type: "linear", parent_id: "root", order: 0, dtype: "BF16", weight_shapes: { weight: [4, 4] }, input_shape: [-1, -1, 4], output_shape: [-1, -1, 4] },
      ],
    },
  };
  const chip = {
    memory_bandwidth: 100,
    peak_flops: { bf16: 1000 },
    interconnect: { intra_node: { bandwidth: 100 } },
  };
  const result = buildNodeLens(structure, chip, { phase: "decode", sequence: 8, plan: { tp: 2 } });
  assert.equal(result.ok, true);
  assert.ok(result.nodes["root.0"].times.comm > 0);
  assert.equal(result.nodes["root.0"].bytesMoved, 24);
  assert.equal(buildNodeLens(structure, chip, { plan: { tp: 0 } }).ok, false);
});
