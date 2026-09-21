// 推理场景的一阶显存核算；结果是理论估算，不是运行时实测。
// 容量只 walk 图上的声明（原则 §3.8）：KV/KDA 来自叶 attributes，不是 config 闭式。

import { paramBytes } from "../structure/operators/formulas/paramDtypes.js";
import { walkStructure } from "./traverse.js";
import { getFrameworkRuntimeProfile } from "../frameworkProfiles.js";

const BYTES_PER_DTYPE = {
  BF16: 2, F16: 2, FP16: 2, F32: 4, FP32: 4, F8_E4M3: 1, F8_E5M2: 1, F8_E8M0: 1, I8: 1,
  BFLOAT16: 2, FLOAT16: 2, FLOAT32: 4,
  U8: 1, I16: 2, I32: 4, I64: 8,
  // fp4 (float4_e2m1fn_x2)：2 值/字节 = 0.5 B/elem。含 scale 摊销的有效字节：
  //   F4_E4M3S16 = 压缩 KV（E4M3 scale/16）= 0.5+1/16 = 0.5625；F4_E8M0S32 = index（E8M0 scale/32）= 0.53125。
  F4: 0.5, FP4: 0.5, F4_E2M1: 0.5, F4_E4M3S16: 0.5625, F4_E8M0S32: 0.53125,
  // DSA（deepseek_v32/glm5_next）index 键缓存：fp8(uint8,1B) + E8M0 尺度(4B/128) = 1 + 4/128 = 1.03125 B/elem。
  // SGLang 硬编码 uint8（DSATokenToKVPool.index_k_with_scale_buffer_dtype），不受 --kv-cache-dtype 影响。
  F8_E8M0S128: 1.03125,
};

export function bytesPerDtype(dtype, fallback = 2) {
  return BYTES_PER_DTYPE[String(dtype || "").replace(/^torch\./i, "").toUpperCase()] ?? fallback;
}

function product(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return 0;
  return shape.reduce((total, value) => total * (Number.isFinite(value) && value >= 0 ? value : 0), 1);
}

/** 把 IR 数值 shape 中的动态维解析为当前推理负载的元素数。 */
export function tensorElements(shape, { batch = 1, sequence = 1, phase = "prefill", attentionHeads = 1, vision = false, visionTokens = 1 } = {}) {
  if (!Array.isArray(shape) || shape.length === 0 || shape.some((value) => value == null)) return 0;
  // Image/video dimensions need an explicit workload shape; never reinterpret
  // unknown spatial dimensions as text sequence length.
  if (shape.length === 5 && shape.every((value) => value === -1)) return 0;
  if (shape.length === 4 && shape[0] === -1 && shape[2] === -1 && shape[3] === -1) {
    return batch * (shape[1] > 0 ? shape[1] : attentionHeads) * (phase === "decode" ? 1 : sequence) * sequence;
  }
  let dynamicIndex = 0;
  return shape.reduce((total, value) => {
    if (value !== -1) return total * value;
    const replacement = dynamicIndex++ === 0
      ? batch
      : vision
        ? visionTokens
        : phase === "decode" ? 1 : sequence;
    return total * replacement;
  }, 1);
}

export function activationTensorBytes(shape, options = {}, bytesPerElement = 2) {
  return tensorElements(shape, options) * bytesPerElement;
}

export function nodeWeightBytes(node) {
  if (!node?.weight_shapes) return 0;
  const dtypes = node.attributes?.weight_dtypes || {};
  const fallback = bytesPerDtype(node.dtype);
  return Object.entries(node.weight_shapes).reduce(
    (total, [name, shape]) => total + product(shape) * bytesPerDtype(dtypes[name] || node.dtype, fallback),
    0,
  );
}

function groupElements(group) {
  return (group.count ?? 1) * (group.matrices ?? 1) * (group.out || 0) * (group.in || 0);
}

function groupBytes(group, fallbackBytes = 2) {
  return groupElements(group) * (group.param_dtype ? paramBytes(group.param_dtype) : fallbackBytes);
}

/** 单节点驻留权重：checkpoint shape 优先，否则 weightMatrices（shared 跳过）。 */
export function nodeWeightCapacityBytes(node, { fallbackBytes = 2 } = {}) {
  const shaped = nodeWeightBytes(node);
  if (shaped > 0) return shaped;
  const declaration = node?.attributes?.weightMatrices;
  if (!Array.isArray(declaration) || declaration.length === 0) return 0;
  return declaration.reduce((total, group) => (
    total + (group.shared ? 0 : groupBytes(group, fallbackBytes))
  ), 0);
}

function isMtpPath(node) {
  const id = String(node?.id || "");
  return node?.type === "mtp" || node?.type === "dspark" || /(^|\.)mtp(\.|$)/.test(id);
}

function cachePathSets(graph) {
  const rawById = new Map((graph?.nodes || []).map((node) => [node.id, node]));
  const canonOf = (node) => node?.canonical_id || node?.module_id || node?.id;
  const draft = new Set();
  const dspark = new Set();
  for (const node of graph?.nodes || []) {
    let current = node;
    let draftPath = false;
    let dsparkPath = false;
    while (current) {
      if (current.type === "dspark") {
        draftPath = true;
        dsparkPath = true;
        break;
      }
      if (current.type === "mtp" || /(^|\.)mtp(\.|$)/.test(String(current.id || ""))) {
        draftPath = true;
      }
      current = current.parent_id != null ? rawById.get(current.parent_id) : null;
    }
    if (draftPath) draft.add(canonOf(node));
    if (dsparkPath) dspark.add(canonOf(node));
  }
  return { draft, dspark };
}

function cacheBytesForNode(a = {}, fallback = 2, { fallbackToCapacity = false } = {}) {
  // An explicit zero growth is meaningful (bounded SWA), not "missing".
  if (a.cache_kv_dtype != null) {
    const kv = fallbackToCapacity
      ? (a.cache_kv_growth_elements > 0 ? a.cache_kv_growth_elements : (a.cache_kv_elements || 0))
      : (a.cache_kv_growth_elements ?? a.cache_kv_elements ?? 0);
    const index = fallbackToCapacity
      ? (a.cache_index_growth_elements > 0 ? a.cache_index_growth_elements : (a.cache_index_elements || 0))
      : (a.cache_index_growth_elements ?? a.cache_index_elements ?? 0);
    return kv * bytesPerDtype(a.cache_kv_dtype, fallback)
      + index * bytesPerDtype(a.cache_index_dtype || a.cache_kv_dtype, fallback);
  }
  const kv = a.cache_kv_growth_elements ?? a.cache_kv_elements ?? 0;
  const index = a.cache_index_growth_elements ?? a.cache_index_elements ?? 0;
  return kv * fallback + index * bytesPerDtype(a.cache_index_dtype, fallback);
}

export function stateBytesForNode(attributes = {}, kvBytes = 2, { frameworkProfile = "neutral", config = {} } = {}) {
  if (attributes.state_recurrent_elements != null || attributes.state_conv_elements != null) {
    const dtype = getFrameworkRuntimeProfile(frameworkProfile).resolveStateDtype(attributes, config);
    const recB = bytesPerDtype(dtype, 4);
    return (attributes.state_conv_elements || 0) * 2
      + (attributes.state_recurrent_elements || 0) * recB;
  }
  return (attributes.state_elements || 0) * kvBytes;
}

/**
 * Map graph leaves to logical cache pools before multiplying by the workload.
 * Explicit pool attributes are forward-compatible Graph IR metadata. If an
 * implementation does not provide them, the profile supplies only the
 * ownership rule it can prove. Same pool id means storage alias, not merely a
 * shared allocator; differing declarations are kept as a conservative envelope
 * and exposed as an evidence gap.
 */
export function cacheAccountingFromGraph(graph, {
  kvBytes = 2,
  batch = 1,
  tokens = 1,
  frameworkProfile = "neutral",
  config = {},
} = {}) {
  const profile = getFrameworkRuntimeProfile(frameworkProfile);
  const { draft: draftPaths, dspark: dsparkPaths } = cachePathSets(graph);
  const pools = new Map();
  const buffers = [];
  const rules = new Set();
  const unknownFields = new Set(["runtimeWorkspace", "allocatorPadding", "backendCacheLayout"]);
  const rawById = new Map((graph?.nodes || []).map((node) => [node.id, node]));
  function layerScope(path, draft) {
    if (draft) return { placement: "last" };
    let cur = rawById.get(path);
    let scope = { placement: "first" };
    while (cur) {
      const id = cur.canonical_id || cur.module_id || cur.id;
      const match = String(id).match(/(?:^|\.)(?:layers|language_model|decoder)\.(\d+)(?:\.|$)/);
      const range = String(cur.attributes?.range || "").match(/^(\d+)\.\.(\d+)$/);
      if (range) return { start: Number(range[1]), end: Number(range[2]) };
      if (match) {
        const start = Number(match[1]);
        scope = { start, end: start + Math.max(1, cur.repeat || 1) - 1 };
      }
      cur = rawById.get(cur.parent_id);
    }
    return scope;
  }

  if (graph?.nodes?.length) {
    walkStructure(graph, ({ node, resident, path }) => {
      const attrs = node?.attributes || {};
      const id = node?.id;
      if (attrs.modality === "vision") return;
      const isDraft = draftPaths.has(id);
      const isDspark = dsparkPaths.has(id);
      const descriptor = profile.resolveCacheAllocation(node, { draft: isDraft, dspark: isDspark, config });
      const scope = layerScope(path, isDraft);
      if (attrs.buffer_elements) buffers.push({ bytes: attrs.buffer_elements * 4 * resident, scope, owner: descriptor.owner });
      const growth = descriptor.boundedDraft ? 0
        : cacheBytesForNode(attrs, kvBytes, { fallbackToCapacity: isDraft }) * resident;
      const state = stateBytesForNode(attrs, kvBytes, { frameworkProfile, config }) * resident;
      const window = descriptor.windowElements > 0 && descriptor.windowSize > 0
        ? descriptor.windowElements * descriptor.windowSize * bytesPerDtype(descriptor.windowDtype) * resident
        : (attrs.cache_window_elements || 0) * (attrs.cache_window_size || 0)
          * bytesPerDtype(attrs.cache_window_dtype, kvBytes) * resident;
      if (isDspark) unknownFields.add("dsparkBackendPackingAndPageHeadroom");
      if (String(attrs.attention_kind || "").startsWith("dsv4_")) {
        unknownFields.add("compressedStateAndBackendWindowAllocation");
      }
      if (!growth && !state && !window && attrs.cache_pool_id == null) return;
      rules.add(descriptor.rule);
      const pool = { ...descriptor, kvBytesPerToken: growth, stateBytesPerSequence: state,
        boundedKvBytesPerSequence: window, scope };
      const previous = pools.get(descriptor.id);
      if (previous) {
        const shared = previous.shared || pool.shared || previous.owner !== pool.owner;
        for (const key of ["kvBytesPerToken", "stateBytesPerSequence", "boundedKvBytesPerSequence"]) {
          if (previous[key] !== pool[key]) unknownFields.add(`pool:${descriptor.id}:inconsistent-${key}`);
          previous[key] = Math.max(previous[key], pool[key]);
        }
        previous.shared = shared;
        previous.owner = shared ? "shared" : previous.owner;
        // Keep the physical target's scope even when the draft alias is visited first.
        if (!isDraft) previous.scope = scope;
      } else {
        pools.set(descriptor.id, pool);
      }
    });
  }

  const bucket = () => ({ kvBytes: 0, kvBytesPerToken: 0, boundedKvBytes: 0, stateBytes: 0, stateBytesPerSequence: 0 });
  const buckets = { main: bucket(), draft: bucket(), shared: bucket() };
  for (const pool of pools.values()) {
    pool.boundedKvBytes = pool.boundedKvBytesPerSequence * batch;
    pool.kvBytes = pool.kvBytesPerToken * batch * tokens + pool.boundedKvBytes;
    pool.stateBytes = pool.stateBytesPerSequence * batch;
    const target = buckets[pool.owner] || buckets.main;
    for (const key of Object.keys(target)) target[key] += pool[key] || 0;
  }
  const { main, draft, shared } = buckets;
  return {
    framework: profile.id,
    ...buckets,
    mainKvBytes: main.kvBytes, draftKvBytes: draft.kvBytes, sharedKvBytes: shared.kvBytes,
    totalKvBytes: main.kvBytes + draft.kvBytes + shared.kvBytes,
    mainKvBytesPerToken: main.kvBytesPerToken,
    draftKvBytesPerToken: draft.kvBytesPerToken,
    sharedKvBytesPerToken: shared.kvBytesPerToken,
    totalKvBytesPerToken: main.kvBytesPerToken + draft.kvBytesPerToken + shared.kvBytesPerToken,
    boundedKvBytes: main.boundedKvBytes + draft.boundedKvBytes + shared.boundedKvBytes,
    mainStateBytes: main.stateBytes, draftStateBytes: draft.stateBytes, sharedStateBytes: shared.stateBytes,
    totalStateBytes: main.stateBytes + draft.stateBytes + shared.stateBytes,
    totalStateBytesPerSequence: main.stateBytesPerSequence + draft.stateBytesPerSequence + shared.stateBytesPerSequence,
    pools: [...pools.values()],
    buffers,
    evidence: {
      formulaSources: ["Graph IR cache attributes", `${profile.id} runtime profile; docs/details/framework_accounting.md`],
      profileRules: [
        ...rules,
        "totalKv=unique cache pools",
        "totalVram=weights+buffers+kv+state",
      ],
      unknownFields: [...unknownFields],
    },
  };
}

/**
 * 权重驻留容量：Σ weightMatrices × residentRepeat（MTP repeat=0 仍计入）。
 * 有 weight_shapes（checkpoint 绑定）的叶优先用形状字节，避免与声明双计。
 * includeMtp=false 只用于身份对账：config 声明的投机头不是 checkpoint 实际。
 * ref: vLLM named_parameters；原则 §3.8。
 */
export function graphWeightCapacity(graph, { fallbackBytes = 2, includeMtp = true } = {}) {
  let elements = 0;
  let bytes = 0;
  if (!graph?.nodes?.length) return { elements: 0, bytes: 0 };
  walkStructure(graph, ({ node, resident }) => {
    if (!includeMtp && isMtpPath(node)) return;
    const shaped = nodeWeightBytes(node);
    if (shaped > 0) {
      bytes += shaped * resident;
      return;
    }
    const declaration = node?.attributes?.weightMatrices;
    if (!Array.isArray(declaration) || declaration.length === 0) return;
    for (const group of declaration) {
      if (group.shared) continue;
      const n = groupElements(group);
      elements += n * resident;
      bytes += groupBytes(group, fallbackBytes) * resident;
    }
  });
  return { elements, bytes };
}

/**
 * 身份测试：checkpoint header 的张量名是实际（vLLM load_weights）。
 * mtp_tensor_count>0 → 计入投机头；=0 → 主干。
 * 字段缺席（sidecar 尚未刷新）回退近邻，避免把「还没扫到」当成空声明。
 */
export function declaredElementsForHeader(graph, header) {
  const withMtp = graphWeightCapacity(graph).elements;
  const withoutMtp = graphWeightCapacity(graph, { includeMtp: false }).elements;
  if (Number.isFinite(header?.mtp_tensor_count)) {
    const includeMtp = header.mtp_tensor_count > 0;
    return {
      declared: includeMtp ? withMtp : withoutMtp,
      withMtp,
      withoutMtp,
      includeMtp,
    };
  }
  const headerElements = Number(header?.parameterTotal);
  if (!Number.isFinite(headerElements) || headerElements <= 0) {
    return { declared: withMtp, withMtp, withoutMtp, includeMtp: true };
  }
  const closerWithout = Math.abs(withoutMtp - headerElements) < Math.abs(withMtp - headerElements);
  return {
    declared: closerWithout ? withoutMtp : withMtp,
    withMtp,
    withoutMtp,
    includeMtp: !closerWithout,
  };
}

export function graphShapedWeightBytes(graph) {
  let total = 0;
  if (!graph?.nodes?.length) return 0;
  walkStructure(graph, ({ node, resident }) => {
    total += nodeWeightBytes(node) * resident;
  });
  return total;
}

/** tid2eid 等 buffer：叶 `buffer_elements` × 4B int32。ref: Megatron-Bridge。 */
export function bufferBytesFromGraph(graph, bytesPerElement = 4) {
  let elements = 0;
  if (graph?.nodes?.length) {
    walkStructure(graph, ({ node, resident }) => {
      elements += (node?.attributes?.buffer_elements || 0) * resident;
    });
  }
  return elements * bytesPerElement;
}

/**
 * 从图上叶声明汇总 KV / KDA 容量。
 * cache_kv_elements / cache_index_elements = 每 token 驻留元素（vLLM AttentionSpec）。
 * state_elements = 每 sequence 的 request state（vLLM kda_state_shape）。
 * 容量 ≠ counts.bytes.kvRead。
 */
export function residentMemoryFromGraph(graph, options = {}) {
  const accounting = cacheAccountingFromGraph(graph, options);
  return {
    kvBytesPerToken: accounting.totalKvBytesPerToken,
    kvBytes: accounting.totalKvBytes,
    stateBytesPerSequence: accounting.totalStateBytesPerSequence,
    stateBytes: accounting.totalStateBytes,
  };
}

export function linearStateBytesPerSequence(graph, bytesPerElement = 2) {
  return residentMemoryFromGraph(graph, { kvBytes: bytesPerElement }).stateBytesPerSequence;
}

export function kvBytesPerToken(graph, kvBytes = 2) {
  return residentMemoryFromGraph(graph, { kvBytes }).kvBytesPerToken;
}

/**
 * Compatibility accessor for draft KV growth. Bounded SWA is NOT per-token
 * growth. Production consumers use buildCostAccounting with actual workload
 * tokens; the deprecated draftTokens multiplier is retained only for callers
 * of this legacy accessor.
 */
export function draftKvBytesPerToken(graph, config = {}, kvBytes = 2, { draftTokens = 0, frameworkProfile = "neutral" } = {}) {
  const accounting = cacheAccountingFromGraph(graph, {
    kvBytes, config, frameworkProfile,
    batch: 1,
    tokens: 1,
  });
  // Keep the historical accessor contract for callers that explicitly ask for
  // a verify window. Production accounting receives the actual workload.
  return accounting.draftKvBytesPerToken * (1 + Math.max(0, draftTokens));
}

/**
 * 草稿模型（MTP/DSpark 子树）的常驻**权重**字节。与 `draftKvBytesPerToken` 同源——按 parent
 * 祖先链判定草稿子树，`nodeWeightCapacityBytes` 逐叶累加。给了实际 `weightBytesTotal`
 * （checkpoint/override 口径的总权重）时，按草稿子树在**图权重**里的占比摊到该总量，避免
 * 「图声明字节」与「checkpoint 总量」两种口径直接相减产生的双计；不给则回退图绝对字节。
 * 无 mtp/dspark 子树时返回 0（主干模型逐字节不变）。
 */
export function draftWeightBytes(graph, weightBytesTotal = null) {
  if (!graph?.nodes?.length) return 0;
  const { draft: draftCanon } = cachePathSets(graph);
  if (draftCanon.size === 0) return 0;
  let draftGraphBytes = 0; let totalGraphBytes = 0;
  walkStructure(graph, ({ node, resident }) => {
    const b = nodeWeightCapacityBytes(node) * resident;
    totalGraphBytes += b;
    if (draftCanon.has(node?.id)) draftGraphBytes += b;
  });
  if (!(totalGraphBytes > 0)) return 0;
  if (typeof weightBytesTotal === "number" && weightBytesTotal > 0) {
    return weightBytesTotal * (draftGraphBytes / totalGraphBytes);
  }
  return draftGraphBytes;
}

/** Single resident-memory ledger; measurements never enter this function. */
export function buildCostAccounting({
  weightBytes = 0,
  bufferBytes,
  graph,
  batch = 1,
  tokens = 1,
  kvBytes = 2,
  frameworkProfile = "neutral",
  config = {},
} = {}) {
  const allocation = cacheAccountingFromGraph(graph, { kvBytes, batch, tokens, frameworkProfile, config });
  const buffers = bufferBytes ?? bufferBytesFromGraph(graph);
  const draftWeight = weightBytes === 0 ? 0 : draftWeightBytes(graph, weightBytes);
  const graphBuffers = allocation.buffers.reduce((sum, entry) => sum + entry.bytes, 0);
  const bufferScale = graphBuffers > 0 ? buffers / graphBuffers : 0;
  const draftBuffers = allocation.buffers.filter((entry) => entry.owner === "draft")
    .reduce((sum, entry) => sum + entry.bytes * bufferScale, 0);
  const main = { ...allocation.main, weightBytes: weightBytes - draftWeight, bufferBytes: buffers - draftBuffers };
  const draft = { ...allocation.draft, weightBytes: draftWeight, bufferBytes: draftBuffers };
  const shared = { ...allocation.shared, weightBytes: 0, bufferBytes: 0 };
  for (const bucket of [main, draft, shared]) {
    bucket.vramBytes = bucket.weightBytes + bucket.bufferBytes + bucket.kvBytes + bucket.stateBytes;
  }
  return {
    ...allocation, main, draft, shared,
    buffers: graphBuffers > 0 ? allocation.buffers.map((entry) => ({ ...entry, bytes: entry.bytes * bufferScale }))
      : [{ bytes: buffers, scope: { placement: "first" }, owner: "main" }],
    total: {
      weightBytes, bufferBytes: buffers,
      kvBytes: allocation.totalKvBytes,
      kvBytesPerToken: allocation.totalKvBytesPerToken,
      boundedKvBytes: allocation.boundedKvBytes,
      stateBytes: allocation.totalStateBytes,
      stateBytesPerSequence: allocation.totalStateBytesPerSequence,
      vramBytes: weightBytes + buffers + allocation.totalKvBytes + allocation.totalStateBytes,
    },
  };
}

/** Legacy flat fields are projections of the ledger, never a second formula. */
export function memoryBreakdown(options = {}) {
  const accounting = buildCostAccounting(options);
  const { total, main, draft, shared } = accounting;
  return {
    ...total,
    mainKvBytes: main.kvBytes, draftKvBytes: draft.kvBytes, sharedKvBytes: shared.kvBytes,
    mainStateBytes: main.stateBytes, draftStateBytes: draft.stateBytes, sharedStateBytes: shared.stateBytes,
    accounting,
    totalBytes: total.vramBytes,
  };
}
