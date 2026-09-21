// Pure runtime semantics, not a serving runtime or benchmark model.
// Source revisions and unresolved backend choices: docs/details/framework_accounting.md.
import { recipeStateKind } from "./structure/archs/index.js";

export const FRAMEWORK_PROFILES = ["neutral", "sglang", "vllm"];

export function frameworkProfileOf(profile) {
  return FRAMEWORK_PROFILES.includes(profile) ? profile : "neutral";
}

export function ssmDtypeForFramework(profile, { kind = "gdn", modelDtype = "bfloat16" } = {}) {
  if (frameworkProfileOf(profile) === "neutral") return null;
  // vLLM MambaStateDtypeCalculator: GDN auto follows model dtype, KDA auto is FP32.
  return profile === "vllm" && kind !== "kda" ? modelDtype : "float32";
}

function resolvePlan(plan, config, framework) {
  if (framework !== "vllm" || !(config?.experts > 0)) return plan;
  const epEnabled = (plan.ep ?? plan.EP ?? 1) > 1 || (plan.moeEp ?? plan.moe_ep ?? 1) > 1;
  if (!epEnabled) return plan; // TP-only experts inherit TP; do not inject moeTp=1.
  // vLLM EP group spans TP x DP, not the generic hybrid ETP domain.
  const ep = (plan.tp ?? plan.TP ?? 1) * (plan.dp ?? plan.DP ?? 1);
  return { ...plan, ep, moeEp: undefined, moe_ep: undefined, moeTp: 1, moe_tp: 1 };
}

function resolveCacheAllocation(node, { draft = false, dspark = false, config = {} } = {}, framework = "neutral") {
  const a = node.attributes || {};
  const owner = a.cache_pool_shared || a.cache_pool_owner === "shared"
    ? "shared" : a.cache_pool_owner || (draft ? "draft" : "main");
  return {
    id: a.cache_pool_id != null ? String(a.cache_pool_id) : node.id,
    owner,
    shared: owner === "shared",
    // Config-faithful keeps the historical full-context draft upper bound.
    // Runtime SWA has a bounded private window. BF16 is the uncompressed
    // logical row upper bound; packed pages/backend padding remain unknown.
    boundedDraft: dspark && framework !== "neutral" && a.sliding_window > 0,
    windowElements: framework !== "neutral" && String(a.attention_kind || "").startsWith("dsv4_")
      ? (config.headDim || (dspark ? a.cache_kv_elements : 0)) : 0,
    windowSize: a.sliding_window || config.slidingWindow || 0,
    windowDtype: "BF16",
    // Sharing an allocator/full-to-SWA mapping does NOT share the storage.
    // Current vLLM DSpark has per-layer caches; SGLang allocates a draft ring.
    rule: dspark ? "dspark-private-swa-unless-explicit-pool-alias" : "graph-cache-ownership",
  };
}

const PROFILES = Object.fromEntries(FRAMEWORK_PROFILES.map((id) => [id, Object.freeze({
  id,
  resolvePlan: (plan = {}, config = {}) => resolvePlan(plan, config, id),
  resolveCacheAllocation: (node, context) => resolveCacheAllocation(node, context, id),
  resolveStateDtype: (attrs, config = {}) => {
    if (config.mambaSsmDtype && config.mambaSsmDtype !== "auto") return config.mambaSsmDtype;
    const kind = recipeStateKind(config, attrs.model_kind);
    const modelDtype = config.textConfig?.dtype || config.textConfig?.torch_dtype
      || config.raw?.dtype || config.raw?.torch_dtype || "bfloat16";
    return ssmDtypeForFramework(id, { kind, modelDtype }) || attrs.state_recurrent_dtype;
  },
  resolveCommunication: (plan, option) => ({
    sharedExpertsFusion: id === "sglang" && (option
      ?? plan.enforceSharedExpertsFusion ?? plan.enforce_shared_experts_fusion ?? false) === true,
  }),
})]));

export function getFrameworkRuntimeProfile(profile) {
  return PROFILES[frameworkProfileOf(profile)];
}
