// 并行计划轴的组合语义与声明组投影（N2-4 层 2+3，docs/details/sharding_matrix.md）。
//
// 为什么存在：EP/TP/DP 的组合语义此前散在 weightBytesPerCard 的路径规则表里
// （专家 ÷ep、其余 ÷tp、DP 恒复制），四条已核实的组合全部缺失。这里把它们
// 收敛成纯函数——出处逐条联网核实（2026-09-09）：
// - vLLM：EP_SIZE = TP_SIZE × DP_SIZE（DP attention + EP 是 DeepSeek 系标准
//   部署）；无 EP 时 DP 也把路由专家 ÷dp 切（DP-shards-experts，
//   「DP=复制」只对 attention 成立）；
// - TRT-LLM：混合 ETP（moe_ep × moe_tp）——每卡持 E/moe_ep 个**完整**专家、
//   专家权重再 ÷moe_tp；
// - MLA/MQA 的 KV ×tp 复制与 KDA state 的 attnMode 响应不在此处
//   （parallel.js kvBytesPerCard / stateBytesPerCard 既有实现，KV 是数据类）。
//
// 层 3 内存类响应（每类对轴的响应不同）：
//   weights(tp 组)   → ÷tp；attnMode=dp 且 attention 叶 → 复制（协议 attnMode 行）
//                      MLP tp 组仍 ÷tp。
//   weights(ep 组) → ÷moe_ep（无 EP 时 ÷moe_tp×dp）；
//   weights(vocab 组) → vocabParallel ? ÷tp : 复制；weights(replicated) → 复制。
//   KV / KDA state / activations 的响应见 parallel.js 与 §七范围外登记。

/** 声明组的全部元素数（驻留容量口径；与叶 counts.bytes.weights 的触达口径区分——
 *  触达 = min(k·T, E)，容量 = E，两者在 prefill 大 T 工作点相等，见锚 1）。 */
export function declaredWeightElements(groups) {
  if (!Array.isArray(groups)) return 0;
  return groups.reduce(
    (sum, group) => group.shared ? sum : sum + (group.count ?? 1) * (group.matrices ?? 1) * group.out * group.in,
    0,
  );
}

/**
 * 路由专家组（ep 亲和）的单卡除数与组合语义。
 * - EP 启用（ep>1 或显式 moe_ep>1）：epSize = moe_ep ?? ep；vLLM 组合语义要求
 *   ep = tp×dp（validatePlan 在 attnMode=dp 时校验）。每卡持 epSize 个完整
 *   专家、专家不再 TP 内切（moe_tp 缺省 1）；TRT-LLM 混合 ETP 经显式 moe_tp
 *   把每个专家再 ÷moe_tp。除数 = epSize × moeTp。
 * - EP 未启用：专家跟 TP 切分（moe_tp 缺省 = tp——专家 TP 切分语义），且 DP
 *   也把专家集合 ÷dp 切。除数 = moeTp × dp。
 * setDegree = 专家集合的切分度（expertWeightRange 的 ep 实参）：
 *   EP 启用为 epSize（完整专家不均衡），未启用为 dp（集合按 DP 分、矩阵按 TP 切）。
 */
export function expertShardDivisor(plan = {}) {
  const tp = plan.tp ?? plan.TP ?? 1;
  const dp = plan.dp ?? plan.DP ?? 1;
  const ep = plan.ep ?? plan.EP ?? 1;
  const moeEp = plan.moeEp ?? plan.moe_ep ?? null;
  const moeTp = plan.moeTp ?? plan.moe_tp ?? null;
  const epOn = ep > 1 || (moeEp ?? 1) > 1;
  if (epOn) {
    const epSize = moeEp ?? ep;
    const hybridTp = moeTp ?? 1;
    return { axis: "ep", epOn: true, epSize, moeTp: hybridTp, dpFactor: 1, divisor: epSize * hybridTp, setDegree: epSize };
  }
  const expertTp = moeTp ?? tp;
  return { axis: "ep", epOn: false, epSize: 1, moeTp: expertTp, dpFactor: dp, divisor: expertTp * dp, setDegree: dp };
}

/**
 * 声明组 class → 单卡除数。
 *   ep → expertShardDivisor；tp → ÷tp（attnMode=dp 的 attention 叶除外，复制）；
 *   vocab → vocabParallel ? ÷tp : 复制；replicated → 复制。
 * node 只用于 attnMode=dp 时区分 attention / MLP，缺省按 ÷tp（与旧行为一致）。
 */
export function declaredClassDivisor(klass, plan = {}, node) {
  switch (klass) {
    case "ep":
      return expertShardDivisor(plan).divisor;
    case "tp":
      return attentionReplicatedUnderDp(plan, node) ? 1 : (plan.tp ?? plan.TP ?? 1);
    case "vocab":
      return (plan.vocabParallel ?? plan.vocab_parallel ?? true) ? (plan.tp ?? plan.TP ?? 1) : 1;
    case "replicated":
      return 1;
    default:
      return 1;
  }
}

/** 协议 attnMode=dp：attention 权重复制、KV 按 rank 分区。MLP 的 tp 组仍 ÷tp。 */
function attentionReplicatedUnderDp(plan = {}, node) {
  const attnMode = plan.attnMode ?? plan.attn_mode ?? "tp";
  if (attnMode !== "dp") return false;
  const role = node?.attributes?.communication_role;
  if (role === "tp_mlp_output" || role === "ep_dispatch" || role === "ep_combine") return false;
  if (role === "tp_attention_output") return true;
  const path = String(node?.id || "").toLowerCase();
  if (!path) return false;
  if (/(^|\.)(mlp|moe|experts|expert_mlp|shared_expert)(\.|$)/.test(path)) return false;
  return /(^|\.)(self_attn|attn|attention|q_proj|k_proj|v_proj|qkv|o_proj|out_proj)(\.|$)/.test(path);
}

/**
 * 声明叶子在单卡上的权重字节（组级投影）。
 * @param totalBytes 该叶的（已乘 repeat/scale 的）驻留权重字节；组内按元素数占比分摊，
 *        使 Σ 组投影 == totalBytes（caller 传入的 total 来自声明本身或 checkpoint 校准）。
 * @returns { bytes, axis, divisor } —— axis/divisor 取字节占比最大的组（nodeCostPerCard
 *          用它分摊 compute；混合 class 的叶 compute 归属跟随主导组）。
 */
export function declaredWeightBytesPerCard(totalBytes, groups, plan = {}, node) {
  if (!Array.isArray(groups) || groups.length === 0) {
    return { bytes: totalBytes, axis: "replicated", divisor: 1 };
  }
  const shares = groups.map((group) => ({
    klass: group.class,
    divisor: declaredClassDivisor(group.class, plan, node),
    elements: (group.count ?? 1) * (group.matrices ?? 1) * group.out * group.in,
  }));
  const totalElements = shares.reduce((sum, share) => sum + share.elements, 0);
  if (!(totalElements > 0)) return { bytes: totalBytes, axis: "replicated", divisor: 1 };
  let bytes = 0;
  let dominant = shares[0];
  for (const share of shares) {
    const groupBytes = totalBytes * (share.elements / totalElements);
    bytes += groupBytes / share.divisor;
    if (share.elements > dominant.elements) dominant = share;
  }
  return { bytes, axis: dominant.klass, divisor: dominant.divisor };
}
