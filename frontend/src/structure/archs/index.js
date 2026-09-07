// archs/ —— 每架构 checkpoint 映射（W3-B2，§4.3 层 2，§8.1 的"一处"）。
//
// 形态对标 llama.cpp tensor_mapping（候选模式 + {bid} 占位）+ transformers
// WeightConverter，但按自家 §8.1 收敛为"每家族一份覆盖文件"：
// - index.js：canonical 解析（HF 标准命名约定），任何家族开箱即用；
// - <family>.js：仅当某家族的 checkpoint 命名偏离 canonical 时才创建，
//   以覆盖规则数组登记（首个匹配生效）。可逆校验测试（§4.6）发现新偏离
//   → 在对应家族文件加规则。
//
// 解析结果 = { role, bid, domain }，与模板节点（role + 模板 id 内的层号）做
// 连接键匹配；role 见 model_executor/roles.js。

import { SUFFIX_ROLES } from "../model_executor/roles.js";

const WRAPPERS = new Set(["model", "language_model", "model_tower"]);
const BID_MARKERS = new Set(["layers", "blocks", "h", "layer"]);
const EXPERT_MARKERS = new Set(["experts", "expert"]);
const VISION_MARKERS = new Set(["visual", "vision", "vision_tower", "mm"]);

// 家族覆盖注册表：modelType → 规则数组（首个匹配生效）。
// 规则形态：{ when: (segments) => boolean, scope?: string, role?: string, bid?: number|null }
// 当前所有内置家族均符合 canonical 命名；首个偏离出现时新建 <family>.js 并在此登记。
export const FAMILY_OVERRIDES = {};

function applyOverrides(segments, modelType) {
  const rules = FAMILY_OVERRIDES[modelType];
  if (!rules) return {};
  for (const rule of rules) {
    if (rule.when(segments)) return rule;
  }
  return {};
}

/**
 * checkpoint 模块路径 → { role, bid, domain }（无法定位 role 时 role=undefined，
 * 绑定退化为路径匹配兜底）。
 * @param {string} moduleId checkpoint 模块路径（如 "model.layers.0.self_attn.q_proj"）
 * @param {string} modelType 用于家族覆盖分派
 */
export function resolveCheckpointModule(moduleId, modelType) {
  const segments = String(moduleId || "").split(".").filter(Boolean);
  while (segments.length > 0 && WRAPPERS.has(segments[0])) segments.shift();

  const override = applyOverrides(segments, modelType);
  const domain = segments.some((s) => VISION_MARKERS.has(s)) ? "vision" : "text";

  let bid = null;
  for (let i = 1; i < segments.length; i++) {
    if (BID_MARKERS.has(segments[i - 1]) && /^\d+$/.test(segments[i])) {
      bid = Number(segments[i]);
      break;
    }
  }
  const hasExpertIndex = segments.some((s, i) => EXPERT_MARKERS.has(s) && /^\d+$/.test(segments[i + 1] || ""));

  const suffix = segments.at(-1);
  let role = override.role ?? SUFFIX_ROLES[suffix];
  // checkpoint 顶层 final norm（model.norm）：通用后缀 "norm" 不入表（vision
  // 的 merger.norm 会误标），仅在剥完 wrapper 后深度为 1 时特判。
  if (!override.role && segments.length === 1 && suffix === "norm") {
    role = "output_norm";
  }
  if (role && !override.role) {
    const scoped = override.scope
      ?? (segments.includes("shared_experts") ? "shexp" : undefined);
    // shexp 作用域改写由 roles.js 的 ROLE_SCOPES 承担
    if (scoped === "shexp") {
      role = { ffn_gate: "ffn_gate_shexp", ffn_up: "ffn_up_shexp", ffn_down: "ffn_down_shexp" }[role] ?? role;
    }
  }
  return { role, bid, domain, hasExpertIndex };
}

/** §4.6 可逆校验用：role → 合法 checkpoint 后缀集合（SUFFIX_ROLES 的逆像）。 */
export function suffixesForRole(role) {
  return Object.entries(SUFFIX_ROLES)
    .filter(([, r]) => r === role)
    .map(([suffix]) => suffix);
}

/** 绑定连接键：domain|bid|role。role 未知的两侧都走路径兜底，不经此键。 */
export function bindingKey({ role, bid, domain }) {
  return `${domain ?? "text"}|${bid ?? "-"}|${role ?? "-"}`;
}

/** 模板节点 → 绑定键（role 已在 B1 落树；bid/域从模板 id 提取）。 */
export function templateBindingKey(node) {
  const id = String(node?.canonical_id || node?.module_id || node?.id || "");
  const segments = id.split(".").filter(Boolean);
  const domain = segments.some((s) => VISION_MARKERS.has(s)) ? "vision" : "text";
  let bid = null;
  for (const segment of segments) {
    if (/^\d+$/.test(segment)) {
      bid = Number(segment);
      break;
    }
  }
  return bindingKey({ role: node.role, bid, domain });
}
