// parallelPlan.js —— 并行计划（parallel plan）的归一化与校验单源（协议 Q8）。
//
// 命名边界：**parallel plan**（本文件，用户输入的逻辑并行轴）。组网逐层调度
// 在 structure/layers/schedule.js，不是产品对象，也不叫 plan。
//
// 校验等式的出处：details/parallel_protocol.md §二（九项裁决 Q1-Q4）——
//   world_size == tp×pp×dp
//   ep ≤ experts；moe_ep ≤ experts；experts % moe_ep == 0（整除）
//   EP 启用 + attnMode=dp + 未显式 moe_ep：ep == tp×dp（vLLM EP_SIZE = TP_SIZE×DP_SIZE）
//   EP 启用：moe_ep × moe_tp == ep × tp（专家域闭合，TRT-LLM Hybrid ETP）
//   无 EP：moe_tp == tp
// 缺省纪律：moe_tp/moe_ep 缺省 undefined、不伪造数值（组合语义由 sharding.js
// expertShardDivisor 按 EP 状态取缺省）。
//
// 错误只产 {code, params}（LSP Diagnostic.code）；展示层 catalog 格式化。

function positiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function issue(code, params) {
  return params ? { code, params } : { code };
}

/** 归一化 + 校验。返回 { ok, errors, plan }；plan 为补齐 worldSize 的完整计划。 */
export function normalizeParallelPlan(plan = {}, config = {}) {
  const normalized = {
    tp: plan.tp ?? plan.TP ?? 1,
    pp: plan.pp ?? plan.PP ?? 1,
    ep: plan.ep ?? plan.EP ?? 1,
    dp: plan.dp ?? plan.DP ?? 1,
    moeTp: plan.moeTp ?? plan.moe_tp,
    moeEp: plan.moeEp ?? plan.moe_ep,
    worldSize: plan.worldSize ?? plan.world_size,
    attnMode: plan.attnMode ?? plan.attn_mode ?? "tp",
    vocabParallel: plan.vocabParallel ?? plan.vocab_parallel ?? true,
    enforceSharedExpertsFusion: plan.enforceSharedExpertsFusion ?? plan.enforce_shared_experts_fusion ?? false,
    // P10（协议 Q7③）：KV keep-ratio —— decode 侧实际驻留的 KV 比例
    // （streaming/滑窗/逐出）。1 = 全保留（缺省）；<1 时 fit 估算按比例折减，
    // 输出标注"估算口径，非运行时行为"。
    kvKeepRatio: plan.kvKeepRatio ?? plan.kv_keep_ratio ?? 1,
  };
  const errors = [];
  for (const key of ["tp", "pp", "ep", "dp"]) {
    if (!positiveInteger(normalized[key])) errors.push(issue("plan.positiveInteger", { key }));
  }
  for (const key of ["moeTp", "moeEp"]) {
    if (normalized[key] != null && !positiveInteger(normalized[key])) {
      errors.push(issue("plan.positiveInteger", { key }));
    }
  }
  const expectedWorld = normalized.tp * normalized.pp * normalized.dp;
  if (normalized.worldSize != null && normalized.worldSize !== expectedWorld) {
    errors.push(issue("plan.worldSize", { expected: expectedWorld }));
  }
  if (!["tp", "dp"].includes(normalized.attnMode)) errors.push(issue("plan.attnMode"));
  if (!(normalized.kvKeepRatio > 0) || normalized.kvKeepRatio > 1) errors.push(issue("plan.kvKeepRatio"));
  if (config?.experts && normalized.ep > config.experts) errors.push(issue("plan.epExceedsExperts"));
  if (config?.experts && normalized.moeEp > config.experts) errors.push(issue("plan.moeEpExceedsExperts"));

  // 协议 Q2/Q4（P6 新增，此前散落在文档未执法）：
  if (normalized.moeEp != null && normalized.moeEp > normalized.ep) {
    errors.push(issue("plan.moeEpExceedsEp", { moeEp: normalized.moeEp, ep: normalized.ep }));
  }
  const epEnabled = normalized.ep > 1;
  if (epEnabled && normalized.moeEp != null) {
    if (config?.experts && config.experts % normalized.moeEp !== 0) {
      errors.push(issue("plan.expertsNotDivisible", { experts: config.experts, moeEp: normalized.moeEp }));
    }
    if (normalized.moeTp != null && normalized.moeEp * normalized.moeTp !== normalized.ep * normalized.tp) {
      errors.push(issue("plan.expertDomainOpen", {
        moeEp: normalized.moeEp,
        moeTp: normalized.moeTp,
        ep: normalized.ep,
        tp: normalized.tp,
      }));
    }
  }
  if (!epEnabled && normalized.moeTp != null && normalized.moeTp !== normalized.tp) {
    errors.push(issue("plan.moeTpEqualsTp", { moeTp: normalized.moeTp, tp: normalized.tp }));
  }
  // vLLM 组合语义：EP 启用时 ep_size = tp × dp（DP attention + EP 是 DeepSeek 系
  // 标准部署）。仅在依赖 ep 缺省（未显式声明 moe_ep 的混合 ETP 不受此约束）且
  // DP attention 时硬校验。
  if (epEnabled && normalized.attnMode === "dp" && normalized.moeEp == null
    && normalized.ep !== normalized.tp * normalized.dp) {
    errors.push(issue("plan.epEqualsTpDp", { expected: normalized.tp * normalized.dp }));
  }
  return { ok: errors.length === 0, errors, plan: { ...normalized, worldSize: normalized.worldSize ?? expectedWorld } };
}
