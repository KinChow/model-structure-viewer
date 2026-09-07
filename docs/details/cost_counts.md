# 算子动作向量注册表（W1 实现规格）

本文是 `frontend/src/structure/formulas/index.js` 每个条目 `counts()` 的实现规格：
分类、公式、共享实现与假设。实现以本文为准；修改公式先改本文。

**逐条公式来源与单位约定见 [`principles.md`](../principles.md) §3.1 / §3.7**：
`matrix` 存 MACs（aten 公式含 2×，抄时换算）、`vector` 存 flop、`sfu` 存操作次数、
`bytes` 为每次前向 compulsory traffic（权重读一遍 + 输入 + 输出，无 phase 分支）。

## 全局假设（每条的 counts 注释须引用）

| 编号 | 假设 | 依据 |
|---|---|---|
| A1 | split / view 类重排**零流量**（fused projection 拆分是视图，不发生拷贝） | 2026-09-07 拍板 |
| A2 | softmax 按**融合单遍**实现，logits 读 1 遍；多遍未融合读放大不建模 | 2026-09-07 拍板 |
| A3 | rope 的 sin/cos **查表**，SFU ≈ 0 | 常规实现 |
| A4 | 复合节点的分解假设（F9）逐条标注 | §3.1 分解声明 |
| A5 | SFU 计数约定：sigmoid = 2（exp + rcp）、exp = 1、rsqrt = 1、div = 1；elementwise/vector 操作逐 flop 计 | 2026-09-07 统一口径 |
| A6 | 线性注意力按 per-token 递推语义计（下界）；chunked kernel 的 chunk 内展开会多做 QK^T 项，不建模 | Gated DeltaNet arXiv 2412.06464 |

记号：`T`=tokens（phase 决定），`H`=hidden，`D`=head_dim，`I`=intermediate，
`S`=可见 key tokens，`E`=专家数，`k`=topk，`b`=每元素字节。
`bpe(dtype)` 来自 `cost/memory.js` 的 dtype 字节表。

## 已删除的死条目（2026-09-07 盘点确认，零引用）

`sigmoid`（仅作为属性值）、`kda_decay`（已内含于 gated_delta_attention）、
`kimi_kda`（与 gated_delta_attention 公式逐字相同，路径实际用后者）、
`kimi_kda_output_gate`（实际用 gated_rmsnorm）、`kimi_fused_qkvg_split`（实际用 split）、
`dsv4_output_projection`（wo_a/wo_b 实际用 linear，`ops/index.js:423,430`）。

## 九个共享 counts 实现

### F1 线性

matrix = T·out·in；vector = T·out（bias 加）；sfu = 0；
bytes = { out·in·b, T·in·b, T·out·b }。
aten: `aten.mm`（torch `mm_flop` = m·n·2k FLOPs → 换算 MACs）。

### F2 选择集注意力

matrix = heads·T·S·D ×2（scores + context）；vector ≈ 3·heads·T·S；sfu = 2·heads·T·S；
bytes：Q+K_S+V_S 读一遍（decode 时 K/V 读即读 KV cache），scores 写+读（A2 单遍），
probs 写+读，O 写，**新算 K/V 写回 cache（T·(D+dv)·heads：prefill 全量、decode 1 token）**。
prefill：T=S=seq → O(seq²)；decode：T=1、S=上下文全长 → O(S)（2026-09-07 补 KV cache 写）。
aten: `aten.bmm` ×2 + `aten._softmax`。
S 的取法与 kvHeads 由条目/提取器决定：
| 变体 | id | S | kvHeads | headDim | valueDim |
|---|---|---|---|---|---|
| MHA | gqa | seq/上下文 | =heads | D | D |
| GQA | gqa | 同上 | 实际 KV 头数 | D | D |
| MQA / SWA | dsv4_swa 等 | window | 1 | D | D |
| MLA | mla、dsv4_compressed | 上下文/压缩长 | 1（共享 latent） | kv_lora_rank+rope | kv_lora_rank |
| QSA/DSA | qsa | indexerBudget | 按模型 | D | D |
| 块稀疏 | sparse | blocks×blockSize | 按模型 | D | D |
matrix 不随 kvHeads 变（每个 query head 做完整点积），只有 K/V 流量随 kvHeads 缩小。
（打分式变体覆盖矩阵 2026-09-07 补；linear attention 家族不经 F2，见下表。）

### F7 家族的 linear attention 变体覆盖（2026-09-07 补）

| 变体 | linearAttentionMode | 状态更新 | F7b 参数 |
|---|---|---|---|
| generic gated LA | generic | decay⊙S + k^Tv（plain） | delta=false |
| Qwen3.5 / Qwen4Exp GDN | qwen3_5 / qwen4_exp | gated delta rule | delta=true |
| Kimi / Kimi-K3（KDA） | kimi / kimi_k3 | gated delta rule | delta=true |
| GLM-5.3-Flash | glm5_next | gated delta rule | delta=true |

配套算子：short conv = F7a、output gate = F4、gated RMSNorm = F3(gated)。
投影打包差异（fused qkvz vs 分离 beta/decay）是 linear 节点（F1），与打包无关；
decay 参数化（safe gate / lower_bound / A_log / dt_bias）是属性级，零计数影响。
执行形态假设：按 per-token 递推计，chunked 实现总量等价。
keyDim/valueDim 为每头维度，heads 显式（state = heads·dk·dv）。

### F3 归一化

rmsnorm：matrix=0；vector = 4TH（x²、mean-reduce、×rsqrt、×w）；sfu = T（rsqrt）；
bytes = { H·b, TH·b, TH·b }。
gemma_rmsnorm：同上 + TH（(1+w) 加法）。gated_rmsnorm：rmsnorm + 门乘（sfu += 2TH、vector += TH、bytes + gate 输入）。
分解声明：`mul / reduce / rsqrt / mul`，无单一 aten 对应。

### F4 门控乘

matrix=0；vector = TW；sfu = TW（sigmoid）；
bytes = { W_g·b（输入相关 gate 时）, (G+W)·b, W·b }。

### F5 逐元素激活

swiglu：vector = 2TI；sfu = TI（silu = x·sigmoid(x)）；bytes = { 0, 2TI·b, TI·b }。
vision_activation：同构，φ 由 hidden_act 决定（gelu → sfu 含 exp）。

### F6 旋转位置

rope：matrix=0；vector = 3·T·D_rope（每维对 4 乘 2 加）；sfu ≈ 0（A3 查表）；
bytes = { 0, 2TD·b, 2TD·b }。分解声明，无 aten 对应。

### F7 线性注意力状态与卷积

causal_conv1d：matrix = T·C_dim·w；vector = T·C_dim（silu → sfu 同数）；
bytes = { C_dim·w·b, T·C_dim·b, T·C_dim·b }。
linear_attention / gated_delta_attention：
matrix = 2·T·D_k·D_v（k^Tv 外积 + qS）；vector = T·D_k·D_v（decay 乘）；
sfu = T（exp decay；gated_delta 另加 sigmoid(beta)）；
**bytes = state 读+写 2·T·D_k·D_v·b（递推状态访存主导）+ q/k/v/decay 流**。
gated_delta 比 generic 多 delta matvec（**修正：属矩阵 MACs，matrix = 3T·dk·dv**）与 beta 门。

### F8 MoE 路由与分发

topk：matrix=0；vector ≈ TE（比较/选择）；sfu = Tk（norm_topk_prob）；
bytes = { 0, TE·b, Tk·b }。aten: `aten.topk`。
moe_dispatch：bytes = { 0, TH·b, TkH·b }（gather）。moe_combine：bytes = { 0, (TkH+Tk)·b, TH·b }（scatter + 加权）。
moe_add：vector = TH；bytes = { 0, 2TH·b, TH·b }。
dsv4_hash_route：纯查表，bytes = { 表·b, T·b, Tk·b }。

### F9 重排与视觉

split 家族（全零计算）：bytes = **0**（A1 view 语义）。
vision_merge：bytes = { 0, in·b, out·b }（permute 是真拷贝）。
vision_position：vector = T·H_v（加法）；bytes = { 0, 2TH·b, TH·b }。

## 复合节点（分解声明）

以下条目的 counts = 组合已知实现，组合假设逐条标注；
它们都是叶子 operatorSpec，子节点无独立算子，无双计。

| 条目 | 分解 |
|---|---|
| mla_query_compress | F1 ×2 + F3 |
| mla_kv_compress | F1 + F9(split) |
| attention_residual | F3 ×2 + F1(小投影) + F2(流数维 softmax) + 加权和 |
| hyper_connection | F3(grouped) + F5(silu) + F1(W_down/W_up 小矩阵) + 门控混合 + combine |
| mhc_pre / mhc_post / mhc_contract | sigmoid/softmax 混合 + F1(小矩阵) + 加权求和；mhc_fused_post_pre = post + pre |
| ple | hash ngram 查表 + F1 + F3 + F7(conv) + add |
| qsa_indexer / minimax_sparse_indexer | F2(打分，S=cache/blocks) + F8(topk) |

## 逐条清单（42 条）

| 条目 | 分类 | 实现 | 备注 |
|---|---|---|---|
| linear | 计算+访存 | F1 | aten: mm |
| matmul | 计算+访存 | F2 | scores/context 等 |
| softmax | 仅访存 | F2 内核 | A2 |
| split | 仅搬运 | F9 | A1 |
| causal_conv1d | 计算+访存 | F7 | |
| rope | 仅访存 | F6 | A3 |
| vision_position | 仅访存 | F9 | |
| vision_merge | 仅搬运 | F9 | 真拷贝 |
| vision_activation | 仅访存 | F5 | φ 由 hidden_act |
| rmsnorm | 仅访存 | F3 | |
| gemma_rmsnorm | 仅访存 | F3 | (1+w) |
| swiglu | 仅访存 | F5 | |
| topk | 仅访存 | F8 | aten: topk |
| moe_dispatch | 仅搬运 | F8 | gather |
| moe_combine | 仅搬运 | F8 | scatter |
| moe_add | 仅访存 | F8 | |
| linear_attention | 计算+访存 | F7 | state 主导 |
| linear_attention_gate | 仅访存 | F4 | |
| gated_delta_attention | 计算+访存 | F7 | +beta 门 |
| gated_rmsnorm | 仅访存 | F3 | φ 由配置 |
| mhc_pre | 分解 | F9 表 | |
| mhc_fused_post_pre | 分解 | post+pre | |
| mhc_post | 分解 | F9 表 | |
| mhc_contract | 分解 | F9 表 | |
| mla_query_compress | 分解 | F1+F3+F1 | |
| mla_kv_compress | 分解 | F1+F9 | |
| mla_kv_split | 仅搬运 | F9 | A1 |
| mla_output_gate | 仅访存 | F4 | |
| attention_residual | 分解 | F3+F1+F2 | |
| hyper_connection | 分解 | F3+F5+F1+F4 | |
| ple | 分解 | F8+F1+F3+F7 | |
| shared_expert_gate | 仅访存 | F4 | |
| qsa_indexer | 分解 | F2+F8 | S=cache |
| qsa_attention | 计算+访存 | F2 | S=budget |
| qwen_qkvz_split | 仅搬运 | F9 | A1 |
| attention_qkv_split | 仅搬运 | F9 | A1 |
| attention_output_gate | 仅访存 | F4 | |
| minimax_sparse_indexer | 分解 | F2+F8 | S=blocks |
| minimax_sparse_attention | 计算+访存 | F2 | S=blocks |
| dsv4_hash_route | 仅搬运 | F8 | |
| dsv4_swa_attention | 计算+访存 | F2 | S=window |
| dsv4_compressed_attention | 计算+访存 | F2 | S=压缩长 |

## 结构级缺口（登记于 principles §10，不在 W1 修）

- **embedding gather 无算子节点**：查表流量 T·H·b 不可见（`embedding.js` 无 operator 子节点）。
- **残差加法无算子节点**：decoder 层 `+x` 隐含在顺序边，每次 2TH·b 读 + TH·b 写不可见。
