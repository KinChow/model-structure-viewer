// counts：动作向量注册（docs/details/cost_counts.md F1-F9；principles §3.1/§3.7）。
// 纯 shape 参数，禁止 node/显示名（§3.2）；单位与假设见 cost_counts.md。
//
// 来源标注三级体系（principles §3.5 / MAINTENANCE 3c，M11-P2-4 全量补齐）：
// - 一等：aten/算子库锚点（注明 FLOPs↔MACs 2× 换算）；
// - 二等：modeling 源码对照，引用 models/<org>/<id>/ 入库证据（或其缺失声明）；
// - 三等：分解声明（复合条目，写明由哪几个 F 函数组合）。
// 每条注明单位换算与 A1-A7 全局假设引用（docs/details/cost_counts.md）。
import {
  linearCounts, attentionCounts, softmaxCounts, rmsnormCounts, gateCounts, swigluCounts,
  ropeCounts, causalConvCounts, linearAttentionStateCounts, topkCounts, moeDispatchCounts,
  moeCombineCounts, addCounts, hashRouteCounts, rearrangeCounts,
} from "./counts.js";

const sumCounts = (...parts) => parts.reduce((total, part) => ({
  matrix: total.matrix + part.matrix,
  vector: total.vector + part.vector,
  sfu: total.sfu + part.sfu,
  bytes: {
    weights: total.bytes.weights + part.bytes.weights,
    actIn: total.bytes.actIn + part.bytes.actIn,
    actOut: total.bytes.actOut + part.bytes.actOut,
  },
}));

export const FORMULAS = {
  linear: {
    title: "Linear",
    // ref: 一等 aten::mm；torch mm_flop = m·n·2k FLOPs → matrix 存 MACs（2× 已换算）；
    //      bias 逐 flop 计 vector（A5）。A7：融合实现按语义分解计数。
    formula: "Y = XW^T + b",
    explanation: "线性投影，用于生成 q/k/v、MLP 中间状态或输出投影。",
    inputs: ["X", "W", "b"],
    outputs: ["Y"],
    counts: linearCounts,
  },
  matmul: {
    title: "MatMul",
    // ref: 一等 aten::bmm ×2（scores + context，各含 2× → MACs 已换算）；
    //      A2 softmax 融合单遍（bytes 含 scores/probs 读写各一次）。
    formula: "Y = A B",
    explanation: "矩阵乘法，用于 attention score 或加权 value 聚合。",
    inputs: ["A", "B"],
    outputs: ["Y"],
    counts: attentionCounts,
  },
  softmax: {
    title: "Softmax",
    // ref: 一等 aten::_softmax；torch flop_counter 明确不数 softmax——本仓有意超越
    //      计 vector/sfu/bytes（principles §3.1）；A2 融合单遍，logits 读 1 遍。
    formula: "softmax(x_i) = exp(x_i) / sum_j exp(x_j)",
    explanation: "将 attention score 转成概率分布。",
    inputs: ["scores"],
    outputs: ["probabilities"],
    counts: softmaxCounts,
  },
  split: {
    title: "Fused Projection Split",
    // ref: 一等 aten::split（视图语义，flop_counter 无成本条目）；A1 拆分零计算零流量。
    formula: "[y_1, ..., y_n] = split(z; split_sizes)",
    explanation: "按当前模型声明的 split_sizes 将 fused projection 拆成语义分支；不同模型的分支数量和宽度由节点属性给出。",
    inputs: ["z", "split sizes"],
    outputs: ["semantic branches"],
    counts: rearrangeCounts,
  },
  causal_conv1d: {
    title: "Causal Short Convolution",
    // ref: 一等 aten::conv1d（C_out·C_in·k·T FLOPs 含 2× → MACs 已换算）；
    //      SiLU 2 SFU/元素（A5：sigmoid = exp + rcp）。
    formula: "x'_t = SiLU(Conv1D(x_{t-w+1:t}; w))",
    explanation: "对模型声明的 q/k/v 分支执行 causal short convolution，并维护卷积历史状态。",
    inputs: ["x history", "conv weight", "conv state"],
    outputs: ["x'", "conv state"],
    counts: causalConvCounts,
  },
  rope: {
    title: "RoPE",
    // ref: 三等分解声明（每维对 4 乘 2 加 = 3 flop/元素，无单一 aten 对应）；
    //      A3 sin/cos 查表，sfu ≈ 0。
    formula: "q', k' = rotate(q, k, position)",
    explanation: "对 q/k 注入旋转位置编码。",
    inputs: ["q", "k", "position"],
    outputs: ["q'", "k'"],
    counts: ropeCounts,
  },
  vision_position: {
    title: "Vision Position Embedding",
    // ref: 三等分解声明（逐元素加法，vector = T·H_v）；bytes 读 2 写 1 的一阶访存约定。
    formula: "x' = x + position(image_or_video)",
    explanation: "将空间或时空位置编码加入视觉 patch token；具体实现由视觉塔配置决定。",
    inputs: ["patch tokens", "position"],
    outputs: ["position-aware tokens"],
    counts: addCounts,
  },
  vision_merge: {
    title: "Vision Patch Merge",
    // ref: 三等分解声明（rearrange copy=true）；A1 豁免不适用——patch merge 的
    //      permute 是真拷贝，bytes 按读写各一遍计。
    formula: "y_{i,j} = concat(x_{mi+a,mj+b})_{a,b=0}^{r-1}",
    explanation: "将相邻空间 patch token 按 merge size 重排并拼接，为视觉投影层提供合并后的 token。",
    inputs: ["patch tokens", "merge size"],
    outputs: ["merged visual tokens"],
    counts: (ctx) => rearrangeCounts({ copy: true, ...ctx }),
  },
  vision_activation: {
    title: "Vision Activation",
    // ref: 一等 aten::gelu / aten::silu 家族（φ 由视觉配置 hidden_act 决定；
    //      gelu 含 exp → SFU）；F5 同构，A5 SFU 口径。
    formula: "y = phi(x)",
    explanation: "视觉前馈层的逐元素激活；具体函数由视觉配置中的 hidden_act 决定。",
    inputs: ["x"],
    outputs: ["y"],
    counts: swigluCounts,
  },
  rmsnorm: {
    title: "RMSNorm",
    // ref: 三等分解声明 mul / reduce / rsqrt / mul（无单一 aten 对应）；
    //      A5：rsqrt = 1、div = 1 SFU，vector 逐 flop。
    formula: "y = x / sqrt(mean(x^2) + eps) * weight",
    explanation: "按均方根归一化，不减去均值。",
    inputs: ["x", "weight", "eps"],
    outputs: ["y"],
    counts: rmsnormCounts,
  },
  gemma_rmsnorm: {
    title: "Gemma RMSNorm",
    // ref: 三等分解声明（F3 + (1+w) 逐元素加法；Qwen3.5 Gemma 风格 checkpoint 语义）；A5。
    formula: "y = x / sqrt(mean(x^2) + eps) * (1 + weight)",
    explanation: "Qwen3.5 使用 Gemma 风格 RMSNorm；checkpoint 权重在归一化缩放时先加 1。",
    inputs: ["x", "weight", "eps"],
    outputs: ["y"],
    counts: (ctx) => rmsnormCounts({ ...ctx, weightOne: true }),
  },
  swiglu: {
    title: "SwiGLU",
    // ref: 一等 aten::silu + aten::mul（silu = x·sigmoid(x)：2 SFU + 1 mul，A5）；
    //      A7：fused gate+up 按语义分解计数，融合收益记 implementation。
    formula: "y = SiLU(xW_gate) * (xW_up)",
    explanation: "门控前馈激活，常用于 LLM 的 MLP。",
    inputs: ["x", "W_gate", "W_up"],
    outputs: ["y"],
    counts: swigluCounts,
  },
  topk: {
    title: "TopK Routing",
    // ref: 一等 aten::topk（flop_counter 不数比较选择——vector 按 T·E 诚实计）；
    //      norm_topk_prob 除法 SFU（A5）。
    formula: "experts = topk(router_logits, k)",
    explanation: "为 token 选择得分最高的专家。",
    inputs: ["router_logits", "k"],
    outputs: ["expert_ids", "expert_weights"],
    counts: topkCounts,
  },
  moe_dispatch: {
    title: "MoE Dispatch",
    // ref: 一等 aten::index_select（gather 纯搬运，零计算）。
    formula: "x_e = dispatch(x, expert_ids)",
    explanation: "按路由结果把 token 分发给专家。",
    inputs: ["x", "expert_ids"],
    outputs: ["expert_inputs"],
    counts: moeDispatchCounts,
  },
  moe_combine: {
    title: "MoE Combine",
    // ref: 三等分解声明（scatter + 加权合并 y = Σ w_e·y_e，vector = 2·TkH 诚实数学；
    //      2026-09-07 规格修正）。
    formula: "y = sum_e weight_e * expert_e(x_e)",
    explanation: "按路由权重合并专家输出。",
    inputs: ["expert_outputs", "expert_weights"],
    outputs: ["y"],
    counts: moeCombineCounts,
  },
  moe_add: {
    title: "MoE Branch Add",
    // ref: 一等 aten::add（routed/shared 分支合并，vector = TH）。
    formula: "y = y_routed + y_shared",
    explanation: "将 routed expert 输出与可选的 shared expert 分支合并。",
    inputs: ["y_routed", "y_shared"],
    outputs: ["y"],
    counts: addCounts,
  },
  linear_attention: {
    title: "Gated Linear Attention",
    // ref: 二等 modeling 对照（Gated DeltaNet arXiv 2412.06464 递推语义，generic
    //      plain 变体；目录 0 模型发射此叶——通用槽位保留）；A6 per-token 递推下界。
    formula: "S_t = decay_t * S_{t-1} + k_t^T v_t; y_t = q_t S_t",
    explanation: "按 token 递推更新线性 attention 状态，避免构造完整的 query-key score 矩阵。",
    inputs: ["q", "k", "v", "decay", "state"],
    outputs: ["state", "y"],
    counts: linearAttentionStateCounts,
  },
  linear_attention_gate: {
    title: "Linear Attention Output Gate",
    // ref: 二等 modeling 对照（KDA 输出门 z·y 路：models/moonshotai/Kimi-K3/
    //      modeling_kimi_linear.py FusedRMSNormGated；目录 0 模型发射此叶）；
    //      A5 sigmoid = 2 SFU。
    formula: "y_t = gate(z_t) * y_t",
    explanation: "使用门控向量调制线性 attention 输出。",
    inputs: ["z", "y"],
    outputs: ["y"],
    counts: gateCounts,
  },
  gated_delta_attention: {
    title: "Gated Delta Attention",
    // ref: 二等 modeling 对照 models/moonshotai/Kimi-K3/modeling_kimi_linear.py
    //      KimiDeltaAttention（:477：beta sigmoid、safe decay exp(g)、S_{t-1}k matvec）；
    //      Qwen3.5/GLM-5 系同族（delta=true）；A6。2026-09-07 数学修正：delta matvec
    //      属矩阵 MACs（matrix = 3T·dk·dv）。
    formula: "beta_t=sigmoid(beta_raw_t); v'_t=beta_t(v_t-S_{t-1}k_t); S_t=exp(g_t)S_{t-1}+v'_tk_t^T; o_t=S_tq_t",
    explanation: "KDA 的统一 q/k L2 normalization、beta sigmoid、safe decay 和 gated-delta recurrent state 更新；模型差异记录在投影属性中。",
    inputs: ["q", "k", "v", "beta_raw", "A_log", "dt_bias", "state"],
    outputs: ["state", "o"],
    counts: (ctx) => linearAttentionStateCounts({ ...ctx, delta: true }),
  },
  gated_rmsnorm: {
    title: "Gated RMSNorm",
    // ref: 二等 modeling 对照 models/moonshotai/Kimi-K3/modeling_kimi_linear.py
    //      FusedRMSNormGated（:539，逐头门控）；三等分解 = F3(gated)：多一路
    //      sigmoid 门乘（A5）。
    formula: "y = RMSNorm(o, weight) * phi(g_2)",
    explanation: "KDA recurrent attention 输出使用输入相关 gate 执行 gated RMSNorm；phi 由模型配置决定（例如 sigmoid 或 SiLU）。",
    inputs: ["o", "g_2", "weight"],
    outputs: ["y"],
    counts: (ctx) => rmsnormCounts({ ...ctx, gated: true }),
  },
  mhc_pre: {
    title: "mHC Pre",
    // ref: 三等分解声明 = F4(post mix) + F1(小矩阵) + softmax(comb mix) + add；
    //      vLLM MHCPreOp（layers/hybrid.js implementation 指针）；A7 融合按语义分解。
    formula: "p=sigmoid(M_a s_a+b_a)+eps; C=Sinkhorn(softmax(M_c s_c+b_c)+eps); x=sum_i p_i H_i",
    explanation: "vLLM MHCPreOp：从多 residual streams 计算 post mix、comb mix，并合成为 attention 输入。",
    inputs: ["residual streams", "hc function", "hc scale", "hc base"],
    outputs: ["post mix", "comb mix", "layer input"],
    counts: (ctx) => sumCounts(gateCounts(ctx.mix), linearCounts(ctx.matrix), addCounts(ctx.merge)),
  },
  mhc_fused_post_pre: {
    title: "mHC Fused Post + Pre",
    // ref: 三等分解声明 = mhc_post + mhc_pre 层间融合（vLLM MHCFusedPostPreOp）；
    //      A7：融合收益记 attributes.implementation，不折算流量。
    formula: "(H',post',C',x') = MHCPre(MHCPost(x,H,post,C); F,scale,base)",
    explanation: "vLLM 在相邻 decoder layer 间融合上一层 post 与当前层 pre，并可同时执行 RMSNorm。",
    inputs: ["block output", "residual streams", "post mix", "comb mix", "hc function"],
    outputs: ["residual streams", "post mix", "comb mix", "layer input"],
    counts: (ctx) => sumCounts(gateCounts(ctx.post), addCounts(ctx.inject), gateCounts(ctx.pre), linearCounts(ctx.matrix)),
  },
  mhc_post: {
    title: "mHC Post",
    // ref: 三等分解声明 = F1(combine 小矩阵) + add(inject)；vLLM MHCPostOp
    //      （H'_j = post_j·x + Σ C_ij H_i）。
    formula: "H'_j = post_j * x + sum_i C_{ij} H_i",
    explanation: "vLLM MHCPostOp：把 attention 或 MLP 输出注入 residual streams。",
    inputs: ["block output", "residual streams", "post mix", "comb mix"],
    outputs: ["residual streams"],
    counts: (ctx) => sumCounts(linearCounts(ctx.combine), addCounts(ctx.inject)),
  },
  mhc_contract: {
    title: "mHC Contract",
    // ref: 三等分解声明（n 流平均收缩，addCounts）；GLM-5.3-Flash 末层 HCContract 语义。
    formula: "h = (1 / n) * sum_i H_i",
    explanation: "GLM-5.3-Flash 最后一层将 n 个 residual streams 平均收缩回普通 hidden state。",
    inputs: ["residual streams"],
    outputs: ["hidden state"],
    counts: (ctx) => addCounts(ctx.contract),
  },
  mla_query_compress: {
    title: "MLA Query Compression",
    // ref: 三等分解声明 = F1(q_a 投影) + F3(norm)；对照 models/deepseek-ai/
    //      DeepSeek-V3.1/modeling_deepseek.py q_a_proj（:661）/ q_a_layernorm（:664）/
    //      q_b_proj（:769 调用点）；q_b 由独立 q_b_proj 叶计——2026-09-07 审计防双计。
    formula: "c^q_t = W_{qa} x_t; q_t = W_{qb} RMSNorm(c^q_t)",
    explanation: "将 query 压缩到低秩 latent 后恢复多头 query。",
    inputs: ["x", "W_qa", "W_qb"],
    outputs: ["q"],
    // q_b 由独立 q_b_proj 叶计；组合含 qb 会与叶双计（2026-09-07 审计）
    counts: (ctx) => sumCounts(linearCounts(ctx.qa), rmsnormCounts(ctx.norm)),
  },
  mla_kv_compress: {
    title: "MLA KV Compression",
    // ref: 三等分解声明 = F1(kv_a 投影) + F9(split view)；对照 models/deepseek-ai/
    //      DeepSeek-V3.1/modeling_deepseek.py kv_a_proj_with_mqa（:669）；
    //      latent cache 写由本叶 actOut 计。
    formula: "[c^{KV}_t, k^R_t] = W_{kv} x_t",
    explanation: "将 KV 压缩为共享 latent 与旋转位置分量，供 MLA attention 使用。",
    inputs: ["x", "W_kv"],
    outputs: ["c_KV", "k_R"],
    counts: (ctx) => sumCounts(linearCounts(ctx.proj), rearrangeCounts()),
  },
  mla_kv_split: {
    title: "MLA KV Latent Split",
    // ref: 一等 aten::split（视图语义）；A1：latent/rope 拆分零流量。
    formula: "[c^{KV}, k^R] = split(z; kv\_lora\_rank, rope\_dim)",
    explanation: "DeepSeek MLA 将 KV 投影结果拆成可缓存的 latent 和独立 rotary 分量；两者后续路径不同。",
    inputs: ["z", "split sizes"],
    outputs: ["c_KV", "k_R"],
    counts: rearrangeCounts,
  },
  mla_output_gate: {
    title: "MLA Output Gate",
    // ref: 二等 modeling 对照 models/moonshotai/Kimi-K3/modeling_kimi_linear.py
    //      （:398 mla_use_output_gate、:470 门乘；目录仅 Kimi-K3 发射此叶）；
    //      A5 sigmoid = 2 SFU。
    formula: "O' = sigmoid(W_g x) * O",
    explanation: "使用输入相关的门控向量调制 MLA 输出。",
    inputs: ["x", "W_g", "O"],
    outputs: ["O'"],
    counts: gateCounts,
  },
  attention_residual: {
    title: "Attention Residual",
    // ref: 二等 modeling 对照 models/moonshotai/Kimi-K3/modeling_kimi_linear.py
    //      （:907 use_attn_residuals、:931 _forward_attn_residual；config
    //      attn_res_block_size=12）；三等分解 = F3(norms) ×2 + F1(score 小投影)
    //      + F2(流数维 softmax) + add(mix)。
    formula: "s_i = <RMSNorm(x_i), w>; p = softmax(s); y = RMSNorm(sum_i p_i x_i)",
    explanation: "Kimi-K3 在 attention 前和 MLP 前从 snapshot bank 与当前 prefix 中按 RMSNorm 后的投影分数聚合 residual stream；block 写层额外保存新的 snapshot。",
    inputs: ["residual_states", "score_projection", "score_norm", "output_norm"],
    outputs: ["y"],
    counts: (ctx) => sumCounts(rmsnormCounts(ctx.norms), linearCounts(ctx.scoreProj), softmaxCounts(ctx.aggregate), addCounts(ctx.mix)),
  },
  hyper_connection: {
    title: "Hyper Connection",
    // ref: 三等分解声明 = F3(grouped) + F5(mix silu) + F1(W_down/W_up) + F4(gate)
    //      + add(combine)；Qwen4Exp delayed HyperConnection（layers/hybrid.js；
    //      Qwen modeling 未入库，离线取证）；A7。
    formula: "x_n=GroupedRMSNorm(H); l=SiLU(W_down x_n); gate=W_up l; block_input=GateMix(x_n,gate); H'=Combine(H,block_output,injection)",
    explanation: "Qwen4Exp 的 delayed HyperConnection：attention 前执行 mix，下一边界先 combine 上一层输出再 mix；最终 mixer 只 materialize 多流状态并输出单流 hidden。",
    inputs: ["hidden_streams", "block_output", "injection", "W_down", "W_up"],
    outputs: ["hidden_streams", "block_input", "injection"],
    counts: (ctx) => sumCounts(rmsnormCounts(ctx.grouped), swigluCounts(ctx.mix), linearCounts(ctx.mixers), gateCounts(ctx.gate), addCounts(ctx.combine)),
  },
  ple: {
    title: "Position Learning Enhancement",
    // ref: 三等分解声明 = hash 查表 + F1(W_kv) + F3(norm) + F7a(short conv) + add；
    //      Qwen4Exp PLE（layers/hybrid.js；Qwen modeling 未入库，离线取证）。
    formula: "e=HashNGram(input_ids,context); [k,v]=W_{kv}e; y=ShortConv(GatedNorm(k,v,RMSNorm(H)))",
    explanation: "Qwen4Exp 指定层的 PLE：根据 input_ids/context 生成 ngram embedding，经 KV projection、grouped norm、gated output 和 dilated short-conv 后加到多流 hidden state。",
    inputs: ["hidden_state", "input_ids", "ngram_context", "ngram_embedding", "W_kv"],
    outputs: ["hidden_state"],
    counts: (ctx) => sumCounts(hashRouteCounts(ctx.embed), linearCounts(ctx.kv), rmsnormCounts(ctx.norm), causalConvCounts(ctx.conv), addCounts(ctx.add)),
  },
  shared_expert_gate: {
    title: "Shared Expert Gate",
    // ref: 二等 modeling 对照（Qwen3.5/3.6 MoE shared expert sigmoid gate；目录 16
    //      模型发射此叶；Qwen modeling 未入库——normalize.js sharedExpertGate
    //      字段驱动）；A5 sigmoid = 2 SFU。
    formula: "y = y_routed + sigmoid(W_g x) * y_shared",
    explanation: "用输入相关 gate 调制 shared expert 输出后与 routed MoE 合并。",
    inputs: ["x", "y_routed", "y_shared", "W_g"],
    outputs: ["y"],
    counts: gateCounts,
  },
  qsa_indexer: {
    title: "QSA Indexer",
    // ref: 二等 modeling 对照（DSA/QSA indexer：vLLM.SparseAttnIndexer /
    //      DeepseekV4Indexer，ops/index.js:393-405；V3.2/V4/Qwen3.8 modeling 未
    //      入库——离线取证）；三等分解 = F2(indexer 打分) + F8(topk)；
    //      bytes 结论 /tmp/m11-formulas/qsa.md §2.3-2.4。
    formula: "I = topk((W_q x) (W_k K)^T / sqrt(d_i), budget)",
    explanation: "用独立 indexer 对历史 token 打分并选择 sparse attention 的候选位置。",
    inputs: ["x", "K_cache", "W_q", "W_k", "budget"],
    outputs: ["selected_indices"],
    counts: (ctx) => sumCounts(attentionCounts(ctx.score), topkCounts(ctx.topk)),
  },
  qsa_attention: {
    title: "QSA Sparse Attention",
    // ref: 二等 modeling 对照（qsa / dsa_sparse_mla / dsv4_sparse_mla 三 kind 共用：
    //      FlashMLA-sparse 吸收式核按 latent 读宽、逐头变体按 kvHeads——变体矩阵见
    //      cost_counts.md F2 表）；F2 整体访存 S = indexerBudget；kvWrite 取舍与
    //      公式 /tmp/m11-formulas/qsa.md §2.4/§4.3（A2 的 4·scores 记本节点）。
    formula: "O = softmax(Q K_I^T / sqrt(d)) V_I",
    explanation: "只在 QSA indexer 选择的候选位置上执行 paged sparse attention。",
    inputs: ["Q", "K_selected", "V_selected", "selected_indices"],
    outputs: ["O"],
    counts: attentionCounts,
  },
  qwen_qkvz_split: {
    title: "Qwen GDN QKVZ Split",
    // ref: 二等 modeling 对照（Qwen3.5 GDN fused qkvz：vLLM.Qwen3NextAttention.qkv_proj
    //      / SGLang.Qwen3_5Attention.qkv_proj，ops/index.js:275）；A1 view 零流量。
    formula: "[q,k,v,z] = split(W_{qkvz}x; q,k,v,z)",
    explanation: "Qwen3.5/Qwen4Exp 将 q、k、v 和 gated RMSNorm 输入 z 打包投影；只有 q/k/v 进入 short convolution，z 旁路到输出归一化。",
    inputs: ["x", "W_{qkvz}"],
    outputs: ["q", "k", "v", "z"],
    counts: rearrangeCounts,
  },
  attention_qkv_split: {
    title: "Attention QKV Split",
    // ref: 二等 modeling 对照 models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py
    //      （fused QKV 拆 q/k/v 语义分支）；A1 view 零流量。
    formula: "[q,k,v] = split(W_{qkv}x; q,k,v)",
    explanation: "MiniMax-M2/3 的 fused QKV 投影在进入归一化与 RoPE 前拆成 q、k、v 语义分支。",
    inputs: ["x", "W_{qkv}"],
    outputs: ["q", "k", "v"],
    counts: rearrangeCounts,
  },
  attention_output_gate: {
    title: "Attention Output Gate",
    // ref: 二等 modeling 对照（Qwen3.5 full attention 输出门：vLLM/SGLang
    //      fused_sigmoid_mul，ops/index.js:290；目录 29 模型发射此叶）；
    //      A5 sigmoid = 2 SFU。
    formula: "O' = sigmoid(G) * O",
    explanation: "Qwen3.5 full attention 在 attention 聚合后，用 qkv projection 中的 gate 经 sigmoid 调制输出。",
    inputs: ["O", "G"],
    outputs: ["O'"],
    counts: gateCounts,
  },
  minimax_sparse_indexer: {
    title: "MiniMax M3 Block Indexer",
    // ref: 二等 modeling 对照 models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py
    //      MiniMaxM3VLIndexer（:492：index_block_size=128 池化打分 +
    //      topk_blocks=16 选块；index-value 路径 checkpoint 显式关闭）；
    //      三等分解 = F2(块打分) + F8(topk blocks)。
    formula: "B = topk_blocks(score_type((Q_i K_i^T) / sqrt(d_i)), k)",
    explanation: "MiniMax M3 的稀疏层用独立 index q/k 分支按 block 打分，选出 sparse_topk_blocks 个 KV blocks，并保留 init/local blocks。",
    inputs: ["index_Q", "index_K", "index weights", "topk blocks"],
    outputs: ["selected block ids"],
    counts: (ctx) => sumCounts(attentionCounts(ctx.score), topkCounts(ctx.topk)),
  },
  minimax_sparse_attention: {
    title: "MiniMax M3 Block-Sparse GQA",
    // ref: 二等 modeling 对照 models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py
    //      MiniMaxM3VLAttention（:408）+ eager_attention_forward（:340）：主 GQA 只读
    //      选中 KV blocks；paged cache 写回计 kvWrite（模板内无叶承担，与
    //      /tmp/m11-formulas/qsa.md §4.3 同规则）；A2 的 4·scores 记本节点。
    formula: "O = softmax(Q K_B^T / sqrt(d)) V_B",
    explanation: "主 GQA 只读取 indexer 选择的 KV blocks；index value/output 分支由配置显式关闭时不参与主输出。",
    inputs: ["Q", "K_selected blocks", "V_selected blocks", "block ids"],
    outputs: ["O"],
    counts: attentionCounts,
  },
  dsv4_hash_route: {
    title: "DeepSeek V4 Hash MoE Routing",
    // ref: 二等 modeling 对照（DeepSeek V4 hash MoE：input_ids 查表固定专家集合，
    //      vLLM tid2eid；actIn/actOut 已核实生效、weights 表条目接线待办——
    //      /tmp/m11-formulas/dsv4.md §(d)）。
    formula: "expert_ids = hash_table[input_ids]",
    explanation: "DeepSeek V4 前 num_hash_layers 层按 input_ids 查表得到固定的专家集合，不执行普通 router logits + top-k。",
    inputs: ["input_ids", "hash_table"],
    outputs: ["expert_ids", "expert_weights"],
    counts: hashRouteCounts,
  },
  dsv4_swa_attention: {
    title: "DeepSeek V4 Sliding-Window MQA",
    // ref: 二等 modeling 对照（DeepSeek V4 sliding-window MQA：vLLM.DeepseekV4SWACache
    //      / MQALayer，ops/index.js:419-427；swa 缓存每 token 一份 headDim 宽 KV
    //      latent，K/V 共享——权重 index 无 V 扩展投影实证）；
    //      /tmp/m11-formulas/dsv4.md §(b)5。
    formula: "O = softmax(Q K_{t-w:t}^T / sqrt(d)) V_{t-w:t}",
    explanation: "compress_ratio=0 层不建立压缩 KV 状态，只在 sliding_window 范围内使用单 KV 头执行 MQA。",
    inputs: ["Q", "K_window", "V_window"],
    outputs: ["O"],
    counts: attentionCounts,
  },
  dsv4_compressed_attention: {
    title: "DeepSeek V4 Compressed MLA",
    // ref: 二等 modeling 对照（DeepSeek V4 compressed MLA：compress_ratio=128，
    //      每压缩位 K/V 态各 headDim（2·headDim）——compressor 输出宽实证；
    //      压缩态写归 compressor 叶计费，本叶无 kvWrite 防双计）；
    //      /tmp/m11-formulas/dsv4.md §(b)6。
    formula: "O = softmax(Q C_{KV}^T / sqrt(d)) C_{KV}",
    explanation: "compress_ratio=128 层把历史 KV 压缩到更短的 cache 序列后执行 MLA；compressor 与 attention 是同一语义链上的两个步骤。",
    inputs: ["Q", "compressed_KV"],
    outputs: ["O"],
    counts: attentionCounts,
  },
};

export function formulaForOperator(operatorId) {
  return FORMULAS[operatorId] || null;
}
