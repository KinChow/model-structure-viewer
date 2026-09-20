import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { MODELS, SGLANG_REUSED_ARCHITECTURES } from "./models/index.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const registry = JSON.parse(
  fs.readFileSync(path.join(here, "__tests__/sglang_arch_registry.json"), "utf8"),
);

// P4 守卫：MSV 支持的 architectures[0] 必须是**真实 SGLang 架构**（EntryClass 类名），
// 防止命名漂移/自造名。快照由 scripts/gen-sglang-arch-registry.mjs 从本地 SGLang 生成。
// 例外表：MSV 领先于 SGLang 已注册的架构（附原因）。新增例外必须显式登记，否则测试红。
const KNOWN_MSV_AHEAD = {
  // DeepSeek-V4.1-Flash 非 transformers-native、需 reference 栈；SGLang 尚无 deepseek_v41 EntryClass。
  DeepseekV41ForCausalLM: "SGLang 未注册 deepseek_v41（V4.1 需 remote reference 栈）；MSV 领先建模。",
};

test("MSV architecture keys align with SGLang EntryClass registry", () => {
  const missing = Object.keys(MODELS).filter(
    (arch) => !registry.architectures[arch] && !KNOWN_MSV_AHEAD[arch],
  );
  assert.deepEqual(
    missing,
    [],
    `以下 MSV 架构键不在 SGLang EntryClass 快照中，也未登记为已知领先例外：${missing.join(", ")}。`
      + ` 若为拼写/命名漂移请修正；若确为 MSV 领先请在 KNOWN_MSV_AHEAD 登记原因并重生成快照。`,
  );
  // 例外表不能腐烂：登记为"领先"的架构若已被 SGLang 注册，应移出例外表（改由快照覆盖）。
  for (const arch of Object.keys(KNOWN_MSV_AHEAD)) {
    assert.ok(
      !registry.architectures[arch],
      `${arch} 已在 SGLang EntryClass 快照中，请从 KNOWN_MSV_AHEAD 例外表移除。`,
    );
  }
});

// 守卫：注册表里多个 architectures[0] 指向同一装配器（复用上游同款类）时，除属主外的
// 复用键必须在 SGLANG_REUSED_ARCHITECTURES 附 SGLang 继承/复用依据；否则可能把独立架构
// 错当子类（deepseek_v41→V4 曾如此）。依据来自 SGLang 源码（class X(Base) 或 self.language_model = Y()）。
test("shared-builder reuse is justified by SGLang inheritance/reuse", () => {
  // 装配器函数 → 引用它的 architectures[0]。
  const builderToArchs = new Map();
  for (const [arch, builder] of Object.entries(MODELS)) {
    if (!builderToArchs.has(builder)) builderToArchs.set(builder, []);
    builderToArchs.get(builder).push(arch);
  }
  // 每个「被 ≥2 键共享」的装配器：去掉已登记的复用键后应恰剩 1 个属主。
  const violations = [];
  for (const [builder, archs] of builderToArchs.entries()) {
    if (archs.length < 2) continue;
    const owners = archs.filter((a) => !SGLANG_REUSED_ARCHITECTURES[a]);
    if (owners.length > 1) {
      violations.push(`${builder.name} 被 ${archs.join(", ")} 共享，但 ${owners.join(", ")} 均未登记复用依据（应仅属主 1 个不登记）`);
    } else if (owners.length === 0) {
      violations.push(`${builder.name} 被 ${archs.join(", ")} 共享，却无属主键（复用登记应保留 1 个属主不登记）`);
    }
  }
  assert.deepEqual(
    violations,
    [],
    `以下架构复用他名装配器却无 SGLang 依据：\n${violations.join("\n")}\n`
      + `请查 SGLang（class X(Base) 或 language_model 复用）确认是否为同款——若非（如 V4.1≠V4）应改独立装配器；`
      + ` 若确为上游复用请在 SGLANG_REUSED_ARCHITECTURES 登记依据。`,
  );
  // 登记表不腐烂：登记的 arch 必须确实与他键共享装配器（若已改独立装配器应移除）。
  for (const arch of Object.keys(SGLANG_REUSED_ARCHITECTURES)) {
    const builder = MODELS[arch];
    const shared = Boolean(builder) && (builderToArchs.get(builder) || []).length >= 2;
    assert.ok(shared, `${arch} 已不与他键共享装配器（或不在注册表），请从 SGLANG_REUSED_ARCHITECTURES 移除。`);
  }
});
