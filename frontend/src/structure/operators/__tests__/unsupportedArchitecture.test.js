import assert from "node:assert/strict";
import test from "node:test";
import { buildNetwork, SUPPORTED_MODEL_ARCHITECTURES } from "../../models/index.js";
import { buildStructureFromConfig } from "../../buildStructure.js";
import { normalizeConfig } from "../../config/normalize.js";
import { resolveArchitecture } from "../../registry/resolveArchitecture.js";
import { createStructureIr } from "../../ir/createStructureIr.js";

// P0-2 回归：未知架构兜底路径曾因 buildGenericConfigNetwork 调用未导入的
// networkSpec 而必崩（ReferenceError）。修复后演进（执行路线步骤 2）：
// 未知架构统一 unsupported——空网络走完管线 + unsupported-architecture 诊断
// 枚举支持项（vLLM _raise_for_unsupported 模式），不再伪造通用结构。

test("unsupported builds an explicit empty network instead of a guessed structure", () => {
  const network = buildNetwork(
    { architecture: "MysteryForCausalLM", resolution: "unsupported" },
    { modelType: "mystery" },
  );
  assert.equal(network.kind, "network");
  assert.equal(network.architecture, "MysteryForCausalLM");
  assert.deepEqual(network.children, []);
});

test("unsupported diagnostic enumerates supported architectures", () => {
  const normalized = normalizeConfig({ model_type: "mystery", architectures: ["MysteryForCausalLM"] });
  const resolved = resolveArchitecture(normalized);
  assert.equal(resolved.resolution, "unsupported");
  const ir = createStructureIr({ network: buildNetwork(resolved, normalized), normalized, resolved });
  const unsupported = ir.diagnostics.unsupported.filter((entry) => entry.code === "unsupported-architecture");
  assert.equal(unsupported.length, 1);
  for (const architecture of SUPPORTED_MODEL_ARCHITECTURES) {
    assert.ok(
      unsupported[0].message.includes(architecture),
      `message should enumerate ${architecture}`,
    );
  }
});

test("buildStructureFromConfig completes for a config with no known architecture", () => {
  // 修复前：此处抛 ReferenceError networkSpec is not defined。
  const structure = buildStructureFromConfig({ model_type: "mystery", architectures: ["MysteryForCausalLM"] });
  assert.ok(structure, "pipeline should complete");
});
