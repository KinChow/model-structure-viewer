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

function resolveStateDtype(framework, attrs, config = {}) {
  if (config.mambaSsmDtype && config.mambaSsmDtype !== "auto") return config.mambaSsmDtype;
  const kind = recipeStateKind(config, attrs.model_kind);
  const modelDtype = config.textConfig?.dtype || config.textConfig?.torch_dtype
    || config.raw?.dtype || config.raw?.torch_dtype || "bfloat16";
  return ssmDtypeForFramework(framework, { kind, modelDtype }) || attrs.state_recurrent_dtype;
}

/**
 * SGLang target verify 会在持久 Mamba state 旁分配 request-slot 级投机 state。
 * 这里必须显式传入投机负载；没有 draft token 数和并发 request 容量时，
 * 不能擅自编造运行时参数。
 * 来源：MambaPool.SpeculativeState / conv_window_dedup_enabled；
 * kv_cache_configurator._build_hybrid_req_pool 跳过 PD prefill/draft worker。
 */
function speculativeStateForSglang(attributes = {}, { config = {}, speculative = {}, draft = false } = {}) {
  if (!speculative?.enabled) return null;
  const recurrentElements = Number(attributes.state_recurrent_elements || 0);
  const convElements = Number(attributes.state_conv_elements || 0);
  if (!recurrentElements && !convElements) return null;
  if (draft || speculative.disaggregationMode === "prefill") {
    return { intermediateSsmBytes: 0, intermediateConvBytes: 0, rule: "sglang-no-target-verify-scratch" };
  }
  const draftTokens = Number(speculative.draftTokens ?? speculative.numDraftTokens);
  const maxRequests = Number(speculative.maxRunningRequests);
  const attentionDpSize = Number(speculative.attentionDpSize ?? 1);
  const eagleTopk = Number(speculative.eagleTopk ?? 1);
  if (![draftTokens, attentionDpSize, eagleTopk].every((v) => Number.isSafeInteger(v) && v > 0)) return null;
  // stateSlots 是约束后的每 attention worker 有效容量（不含 sentinel），
  // 不能把 CLI max-running-requests 直接当实际池容量；可选约束由调用方声明。
  let requests = speculative.stateSlots == null ? Math.floor(maxRequests / attentionDpSize) : Number(speculative.stateSlots);
  if (speculative.stateSlots == null && (!Number.isSafeInteger(maxRequests) || maxRequests <= 0)) return null;
  if (speculative.maxMambaCacheSize != null) {
    const size = Number(speculative.maxMambaCacheSize);
    const ratio = Number(speculative.mambaSlotsPerRequest);
    if (![size, ratio].every((v) => Number.isSafeInteger(v) && v > 0)) return null;
    requests = Math.min(requests, Math.floor(size / ratio));
  }
  if (!Number.isSafeInteger(requests) || requests <= 0 || speculative.enableLinearReplaySsmSpec) return null;
  const slots = requests + 1;
  const convKernel = Number(config.linearConvKernelSize) - 1;
  if (convElements > 0 && (!Number.isSafeInteger(convKernel) || convKernel <= 0 || convElements % convKernel !== 0)) return null;
  const channels = convKernel > 0 ? convElements / convKernel : 0;
  const recurrentDtype = String(resolveStateDtype("sglang", attributes, config))
    .replace(/^torch\./i, "").toUpperCase();
  const recurrentBytes = { BF16: 2, BFLOAT16: 2, F16: 2, FP16: 2, FLOAT16: 2, F32: 4, FP32: 4, FLOAT32: 4 }[recurrentDtype];
  if (recurrentBytes == null) return null;
  // CUDA GDN 线性链使用 [K-1 + D-1] 的唯一 backing storage。
  // KDA 转置、树状 verify、CPU/NPU 和显式禁用去重时必须保留 dense 窗口。
  const denseConv = recipeStateKind(config, attributes.model_kind) === "kda"
    || eagleTopk > 1 || ["cpu", "npu"].includes(speculative.platform)
    || speculative.disableConvWindowDedup === true;
  const convElementsPerSlot = denseConv
    ? convElements * draftTokens : channels * (convKernel + draftTokens - 1) || 0;
  return {
    intermediateSsmBytes: slots * draftTokens * recurrentElements * recurrentBytes,
    intermediateConvBytes: slots * convElementsPerSlot * 2,
    rule: denseConv
      ? "sglang-speculative-state-dense-conv-window"
      : "sglang-speculative-state-dedup-conv-window",
  };
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
  const kpool = Number(a.index_kpool ?? config.dsaIndexKpool ?? 1);
  const isDsaKpool = kpool > 1
    && (a.index_kpool != null
      || String(a.attention_kind || "").startsWith("dsa_")
      || String(a.operator_id || "").includes("dsa"));
  return {
    id: a.cache_pool_id != null ? String(a.cache_pool_id) : node.id,
    owner,
    shared: owner === "shared",
    // vLLM 每个 k-pool 只存一个压缩 index 条目；已验证的 SGLang DSA
    // pool 仍按 token 粒度保留容量。该差异只放在 runtime profile 中，
    // 不改写 neutral Graph IR，也不使用全局乘数。
    indexGrowthDivisor: framework === "vllm" && isDsaKpool ? kpool : 1,
    indexKpool: isDsaKpool ? kpool : 1,
    // config-faithful 保留历史上的全上下文草稿上界；runtime SWA 是有界的
    // 私有窗口。BF16 只表示未压缩逻辑行上界，页打包和 backend padding
    // 仍然是 unknown。
    boundedDraft: dspark && framework !== "neutral" && a.sliding_window > 0,
    windowElements: framework !== "neutral" && String(a.attention_kind || "").startsWith("dsv4_")
      ? (config.headDim || (dspark ? a.cache_kv_elements : 0)) : 0,
    windowSize: a.sliding_window || config.slidingWindow || 0,
    windowDtype: "BF16",
    // 共享 allocator 或 full-to-SWA 映射不等于共享 storage。
    // 当前 vLLM DSpark 是逐层 cache，SGLang 会分配 draft ring。
    rule: dspark ? "dspark-private-swa-unless-explicit-pool-alias" : "graph-cache-ownership",
  };
}

const PROFILES = Object.fromEntries(FRAMEWORK_PROFILES.map((id) => [id, Object.freeze({
  id,
  resolvePlan: (plan = {}, config = {}) => resolvePlan(plan, config, id),
  resolveCacheAllocation: (node, context) => resolveCacheAllocation(node, context, id),
  resolveSpeculativeState: (attrs, context = {}) => (
    id === "sglang" ? speculativeStateForSglang(attrs, context) : null
  ),
  resolveStateDtype: (attrs, config = {}) => resolveStateDtype(id, attrs, config),
  resolveCommunication: (plan, option) => ({
    sharedExpertsFusion: id === "sglang" && (option
      ?? plan.enforceSharedExpertsFusion ?? plan.enforce_shared_experts_fusion ?? false) === true,
  }),
})]));

export function getFrameworkRuntimeProfile(profile) {
  return PROFILES[frameworkProfileOf(profile)];
}
