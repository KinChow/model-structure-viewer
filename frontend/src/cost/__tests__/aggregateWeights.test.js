import assert from "node:assert/strict";
import test from "node:test";
import { materializeStructureGraph } from "../../structure/graph/materializeStructureGraph.js";
import { aggregateCost } from "../aggregate.js";

// P7（步骤 7）：夹具 tree root 经 materializeStructureGraph 转 Graph IR。
const toGraph = (root) => materializeStructureGraph(root);

test("无 checkpoint 和节点权重时不编造闭式容量", () => {
  const result = aggregateCost({ graph: toGraph({ children: [] }), config: { hiddenSize: 4, vocabSize: 10, tieWordEmbeddings: true }, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 0);
  assert.equal(result.weightSource, "empty");
});

test("权重 what-if 只在显式指定时覆盖默认字节数", () => {
  const graph = toGraph({
    id: "embed",
    attributes: { weightMatrices: [{ class: "vocab", out: 10, in: 4, count: 1, matrices: 1 }] },
    children: [],
  });
  const result = aggregateCost({ graph, config: {}, weightBytesPerParameter: 1, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 40);
  assert.equal(result.weightSource, "what-if");
});

test("量化配置进入图声明容量并明确标记来源", () => {
  const group = { class: "tp", out: 8, in: 4, count: 1, matrices: 1 };
  const graph = toGraph({
    id: "linear",
    type: "operator",
    attributes: { operator_id: "linear", weightMatrices: [group] },
    children: [],
  });
  const result = aggregateCost({
    graph,
    config: {
      quantization_config: { quant_method: "gptq", bits: 4, group_size: 128 },
      quantizationBytesPerParameter: 0.5,
      quantizationMethod: "gptq",
    },
    activationPeak: 0,
    runtimeConst: 0,
  });
  assert.equal(result.memory.weightBytes, 8 * 4 * 0.5 + 8 * 1 * (2 + 0.5));
  assert.equal(result.weightSource, "derived-quantized");
  assert.equal(result.assumptions.weightBytesPerParameter, 0.5);
});
