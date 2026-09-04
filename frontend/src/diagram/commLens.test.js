import assert from "node:assert/strict";
import test from "node:test";
import { nodeCommunicationBytes } from "../cost/comm.js";

test("lens 可将 TP 通信量传入节点 bound 计算", () => {
  const bytes = nodeCommunicationBytes(
    { id: "decoder.0.self_attn.o_proj" },
    { hiddenSize: 4096 },
    { tp: 8 },
    { batch: 1, tokens: 1, bytesPerElement: 2 },
  );
  assert.equal(bytes, 14336);
});
