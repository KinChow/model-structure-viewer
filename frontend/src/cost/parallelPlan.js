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

function positiveInteger(value) {
  return Number.isInteger(value) && value > 0;
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
    // P10（协议 Q7③）：KV keep-ratio —— decode 侧实际驻留的 KV 比例
    // （streaming/滑窗/逐出）。1 = 全保留（缺省）；<1 时 fit 估算按比例折减，
    // 输出标注"估算口径，非运行时行为"。
    kvKeepRatio: plan.kvKeepRatio ?? plan.kv_keep_ratio ?? 1,
  };
  const errors = [];
  for (const key of ["tp", "pp", "ep", "dp"]) if (!positiveInteger(normalized[key])) errors.push(`${key} 必须是正整数`);
  for (const key of ["moeTp", "moeEp"]) {
    if (normalized[key] != null && !positiveInteger(normalized[key])) errors.push(`${key} 必须是正整数`);
  }
  const expectedWorld = normalized.tp * normalized.pp * normalized.dp;
  if (normalized.worldSize != null && normalized.worldSize !== expectedWorld) {
    errors.push(`world_size 应为 TP×PP×DP=${expectedWorld}`);
  }
  if (!["tp", "dp"].includes(normalized.attnMode)) errors.push("attn_mode 只能是 tp 或 dp");
  if (!(normalized.kvKeepRatio > 0) || normalized.kvKeepRatio > 1) errors.push("kv_keep_ratio 必须在 (0, 1] 区间");
  if (config?.experts && normalized.ep > config.experts) errors.push("EP 不能大于专家总数");
  if (config?.experts && normalized.moeEp > config.experts) errors.push("moe_ep 不能大于专家总数");

  // 协议 Q2/Q4（P6 新增，此前散落在文档未执法）：
  if (normalized.moeEp != null && normalized.moeEp > normalized.ep) {
    errors.push(`moe_ep(${normalized.moeEp}) 不能大于 ep(${normalized.ep})：专家 ownership 轴是 ep 的细化（协议 Q2）`);
  }
  const epEnabled = normalized.ep > 1;
  if (epEnabled && normalized.moeEp != null) {
    if (config?.experts && config.experts % normalized.moeEp !== 0) {
      errors.push(`experts(${config.experts}) 必须能被 moe_ep(${normalized.moeEp}) 整除，否则不均衡区间无意义`);
    }
    if (normalized.moeTp != null && normalized.moeEp * normalized.moeTp !== normalized.ep * normalized.tp) {
      errors.push(`专家域不闭合：moe_ep(${normalized.moeEp}) × moe_tp(${normalized.moeTp}) 应为 ep(${normalized.ep}) × tp(${normalized.tp})（TRT-LLM Hybrid ETP）`);
    }
  }
  if (!epEnabled && normalized.moeTp != null && normalized.moeTp !== normalized.tp) {
    errors.push(`无 EP 时 moe_tp(${normalized.moeTp}) 应等于 tp(${normalized.tp})：DP 切专家由组合语义承担（协议 Q4）`);
  }
  // vLLM 组合语义：EP 启用时 ep_size = tp × dp（DP attention + EP 是 DeepSeek 系
  // 标准部署）。仅在依赖 ep 缺省（未显式声明 moe_ep 的混合 ETP 不受此约束）且
  // DP attention 时硬校验。
  if (epEnabled && normalized.attnMode === "dp" && normalized.moeEp == null
    && normalized.ep !== normalized.tp * normalized.dp) {
    errors.push(`EP 启用时 ep 应为 TP×DP=${normalized.tp * normalized.dp}（vLLM：EP_SIZE = TP_SIZE × DP_SIZE）`);
  }
  return { ok: errors.length === 0, errors, plan: { ...normalized, worldSize: normalized.worldSize ?? expectedWorld } };
}
