// roles.js —— canonical 节点角色词表（W3-B，§4.3 层 1）。
// 对标 llama.cpp 的 MODEL_TENSOR（LLM_TENSOR_ATTN_Q 等），命名取其 snake_case 形式；
// role 与 operator_id 是两个正交维度：operator_id 描述"执行什么计算"，
// role 描述"在配方里承担什么参数位"（checkpoint 映射表的连接键，见 structure/archs/）。
//
// 赋值规则：
// - 算子叶（operatorSpec）：按 id 末段查 SUFFIX_ROLES，且 operatorId 必须具备承载
//   参数的能力（无参算子即使后缀同名也不赋 role）；mlpOperatorSpecs 的
//   roleScope（shexp = shared expert）把 ffn_* 映射到 llama.cpp 的 *_SHEXP。
// - 模块（moduleSpec）：由创建点显式赋（只有 ~6 处：embed/lm_head/三类 norm），
//   不做后缀推断——vision 的 norm 模块与最终输出 norm 同为 "norm" 后缀，推断必错。

// 具备承载参数能力的算子（checkpoint 里存在对应 weight 模块）。
const PARAM_CAPABLE_OPS = new Set([
  "linear",
  "rmsnorm",
  "gemma_rmsnorm",
  "gated_rmsnorm",
  "mla_query_compress",
  "mla_kv_compress",
  "dsv4_hash_route",
  "linear_attention",
  "causal_conv1d",
  // vision（M8-V1）：position embedding 与 merger conv 均为含参模块
  "vision_position",
  "vision_merge",
]);

// id 末段 → role。仅含 checkpoint 侧真实存在的参数模块。
export const SUFFIX_ROLES = {
  // attention 投影
  q_proj: "attn_q",
  k_proj: "attn_k",
  v_proj: "attn_v",
  o_proj: "attn_out",
  out_proj: "attn_out",
  qkv_proj: "attn_qkv",
  qkv_gate_proj: "attn_qkv",
  qkv_index_proj: "attn_qkv",
  qkv_projection: "attn_qkv",
  qkv: "attn_qkv",
  q_a_proj: "attn_q_a",
  q_b_proj: "attn_q_b",
  kv_a_proj: "attn_kv_a",
  kv_b_proj: "attn_kv_b",
  kv_a_norm: "attn_kv_a_norm",
  q_norm: "attn_q_norm",
  k_norm: "attn_k_norm",
  // KDA / linear attention
  beta_projection: "attn_beta",
  decay_projection: "attn_decay",
  in_proj_b: "attn_decay",
  in_proj_z: "attn_gate",
  short_conv: "attn_conv",
  // FFN / MoE（router = llama.cpp FFN_GATE_INP）
  gate_proj: "ffn_gate",
  up_proj: "ffn_up",
  down_proj: "ffn_down",
  router: "ffn_gate_inp",
  hash_router: "ffn_gate_inp",
  // kimi_k3 latent MoE 的共享压缩/扩展投影（llama.cpp 无对应，扩展词表）
  routed_expert_down_proj: "ffn_latent_down",
  routed_expert_up_proj: "ffn_latent_up",
  // vision（M8-V1）：llama.cpp mm.* 风格；qkv_proj/out_proj/gate_proj 等通用后缀
  // 复用既有 role（绑定键含 domain=vision，与文本侧天然隔离）
  patch_embed: "vision_patch_embd",
  position: "vision_position",
  fc1: "vision_ffn_up",
  fc2: "vision_ffn_down",
  input_norm: "vision_attn_norm",
  post_norm: "vision_block_norm",
  patch_merge: "vision_patch_merge",
  // norm / embed / lm_head 模块后缀（checkpoint 与模板双侧同名）
  input_layernorm: "attn_norm",
  post_attention_layernorm: "ffn_norm",
  embed_tokens: "token_embd",
  lm_head: "output",
  // 注意：通用后缀 "norm" 不入表——vision 的 merger.norm 等算子叶会误标
  // output_norm；checkpoint 顶层 model.norm 由 archs/index.js 深度规则特判。
};

// roleScope → { 原 role: 作用域内 role }。shexp 对标 llama.cpp FFN_*_SHEXP。
const ROLE_SCOPES = {
  shexp: { ffn_gate: "ffn_gate_shexp", ffn_up: "ffn_up_shexp", ffn_down: "ffn_down_shexp" },
};

function lastSegment(id) {
  return String(id || "").split(".").pop();
}

/** 算子叶 role：末段查表 + 参数能力过滤 + 作用域改写。 */
export function resolveOperatorRole(id, operatorId, roleScope) {
  if (!PARAM_CAPABLE_OPS.has(operatorId)) return undefined;
  const base = SUFFIX_ROLES[lastSegment(id)];
  if (!base) return undefined;
  return roleScope ? ROLE_SCOPES[roleScope]?.[base] ?? base : base;
}
