import assert from "node:assert/strict";
import test from "node:test";
import { classifyRoofline } from "../cost/roofline.js";

test("图 lens 的 bound 结果可按节点成本计算", () => {
  const result = classifyRoofline({ macs: 10, weightBytes: 100 }, {
    memory_bandwidth: 100,
    peak_flops: { bf16: 1000 },
    interconnect: { intra_node: { bandwidth: 100 } },
  });
  assert.equal(result.bound, "memory");
});
