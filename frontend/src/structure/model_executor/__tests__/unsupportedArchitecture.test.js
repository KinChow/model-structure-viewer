import assert from "node:assert/strict";
import test from "node:test";
import { buildNetwork, SUPPORTED_MODEL_ARCHITECTURES } from "../models/index.js";
import { buildStructureFromConfig } from "../../buildStructure.js";
import { normalizeConfig } from "../../config/normalize.js";
import { resolveArchitecture } from "../../registry/resolveArchitecture.js";
import { createStructureIr } from "../../ir/createStructureIr.js";

// P0-2 回归：generic-config 兜底路径曾因 buildGenericConfigNetwork 调用未导入的
// networkSpec 而必崩（ReferenceError）。修复后契约：空网络走完管线 +
// unsupported 诊断枚举支持项（vLLM _raise_for_unsupported 模式）。

test("generic-config builds an explicit empty network instead of crashing", () => {
  const network = buildNetwork(
    { canonicalArchitecture: "generic-config", architecture: "MysteryForCausalLM" },
    { modelType: "mystery" },
  );
  assert.equal(network.kind, "network");
  assert.equal(network.canonicalArchitecture, "generic-config");
  assert.deepEqual(network.children, []);
});

test("generic-config diagnostic enumerates supported architectures", () => {
  const normalized = normalizeConfig({ model_type: "mystery", architectures: ["MysteryForCausalLM"] });
  const resolved = resolveArchitecture(normalized);
  assert.equal(resolved.canonicalArchitecture, "generic-config");
  const ir = createStructureIr({ network: buildNetwork(resolved, normalized), normalized, resolved });
  const unsupported = ir.diagnostics.unsupported.filter((entry) => entry.code === "generic-config");
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
