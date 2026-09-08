# Kimi-K3 文本解码器（KimiLinear）建模源码取证报告

> 取证日期：2026-09-08。只读取证，未改动任何模型代码。
>
> 本地入库文件：
> - `/tmp/m8v2/modeling_kimi_linear.py`（1314 行，来自 HF `moonshotai/Kimi-K3` 仓库 raw 抓取，51506 字节）
> - `/tmp/m8v2/kimi_k3_README.md`、`/tmp/m8v2/kimi_k3_config.json`、`/tmp/m8v2/configuration_kimi_k3_hf.py`（辅助取证）
> - 本地 `/tmp/m8v2/configuration_kimi_k3.py` 与 HF 版本 `diff` 结果为 **完全一致**（已验证）。

## 0. 源码定位与仓库结构

| 仓库 | 文本建模文件 | 备注 |
|---|---|---|
| `moonshotai/Kimi-K3` | **`modeling_kimi_linear.py`**（51,506 字节） | 文件头声明："The multi-head latent attention, MoE gating and sparse MoE block in this file are adapted from DeepSeek-V3 … extensively modified and extended for the Kimi-Linear architecture"（`modeling_kimi_linear.py:1-6`） |
| `moonshotai/Kimi-K2-Thinking` | `modeling_deepseek.py`（75,769 字节）/ `configuration_deepseek.py` | K2-Thinking 仍用 DeepSeek 命名，未迁移到 KimiLinear 命名 |

来源：HF API `https://huggingface.co/api/models/moonshotai/Kimi-K3/tree/main` 与 `.../Kimi-K2-Thinking/tree/main`（文件名+字节数直接取自 API 返回）。confidence：**高**。

引用链确认：本地 `/tmp/m8v2/modeling_kimi_k3.py:919` `self.language_model = KimiLinearForCausalLM(config.text_config)`；`config.json` 的 `text_config.auto_map` 指向 `modeling_kimi_linear.KimiLinearForCausalLM`（`/tmp/m8v2/kimi_k3_config.json:27-31`）。confidence：**高**。

依赖：`fla`（fla-core）提供 `FusedRMSNormGated`、`ShortConvolution`、`chunk_kda`、`fused_recurrent_kda`（`modeling_kimi_linear.py:46-53`）。

---

## 1. KDA（Kimi Delta Attention）解码层权重张量构成

实现类 `KimiDeltaAttention`（`modeling_kimi_linear.py:477-663`）。设 `H = hidden_size = 7168`，`n_h = num_heads = 96`，`d_h = head_dim = 128`，`P = n_h × d_h = 12288`（来源：config `linear_attn_config.num_heads=96, head_dim=128`，`/tmp/m8v2/kimi_k3_config.json:94,166`）。

### 1.1 qkv 投影 + 短卷积

| 权重 | 形状 | 说明 | 行号 |
|---|---|---|---|
| `q_proj` | Linear(7168 → 12288, no bias) | `projection_k_size = head_k_dim × num_k_heads` | 498-499, 495 |
| `k_proj` | Linear(7168 → 12288, no bias) | 同上 | 500-501 |
| `v_proj` | Linear(7168 → 12288, no bias) | `projection_size = head_dim × num_heads` | 502, 496 |
| `q_conv1d` / `k_conv1d` | ShortConvolution(12288, kernel_size=4, activation='silu') | 短卷积核宽 `short_conv_kernel_size=4`（config:167） | 504-513 |
| `v_conv1d` | ShortConvolution(12288, kernel_size=4, activation='silu') | | 514-518 |

confidence：**高**（源码+config 双重印证）。

### 1.2 衰减（decay/g）与 beta

| 权重 | 形状 | 说明 | 行号 |
|---|---|---|---|
| `A_log` | Parameter(96,) | 每头一个，`log(uniform(1,16))` 初始化，传入 `chunk_kda` 内核 | 520-521 |
| `f_a_proj` | Linear(7168 → 128, no bias) | 低秩瓶颈（rank = head_dim） | 523 |
| `f_b_proj` | Linear(128 → 12288, no bias) | 升回每头 128，输出作为 **decay 门控 g** 传入 `chunk_kda(..., g=g)`（601-602, 610-614） | 524 |
| `dt_bias` | Parameter(12288,) | 传入内核 | 526-527 |
| `b_proj` | Linear(7168 → 96, no bias) | **beta**（每头一个标量），内核内做 sigmoid（`use_beta_sigmoid_in_kernel=True`，622, 641） | 529, 603 |

即：KDA 的细粒度 decay 是 `7168 → 128 → 12288` 的低秩两段（f_a/f_b），与"输出门"是两套独立投影。confidence：**高**。

### 1.3 输出门（gate/z）—— `use_full_rank_gate=true` 时的宽度

`modeling_kimi_linear.py:531-537`：

```python
self.use_full_rank_gate = config.linear_attn_config.get("use_full_rank_gate", False)  # K3: true
if self.use_full_rank_gate:
    self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)   # 7168 → 12288（全秩）
else:
    self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)   # 7168 → 128（低秩路径）
    self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)    # 128 → 12288
```

- **K3 配置 `use_full_rank_gate: true`（config:168）→ 只存在单个 `g_proj: 7168 → 12288`（权重 7168×12288 ≈ 88.1M/层），低秩路径 `g_a_proj/g_b_proj` 不实例化。**
- forward 中 `g = self.g_proj(hidden_states)` 后 `rearrange('... (h d) -> ... h d')`（651-655），与 decay 的 g 是两个不同张量。
- `gate_lower_bound = -5.0`（config:93）传入内核作 `safe_gate/lower_bound`（623-624）。

confidence：**高**。

### 1.4 gated RMSNorm（z 的消费方式）

`modeling_kimi_linear.py:539-540, 651-659`：

```python
self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
...
g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
o = self.o_norm(o, g)          # 656
o = rearrange(o, 'b t h d -> b t (h d)')
o = self.o_proj(o)             # o_proj: Linear(12288 → 7168, no bias)，541
```

语义：这里的"z"即输出门 g，消费方式为**逐头逐通道的 sigmoid 门控 RMSNorm**——`FusedRMSNormGated`（fla 提供）先对 o 做 RMSNorm，再乘 `sigmoid(g)`，作用在每头 `head_dim=128` 维上，然后 `o_proj` 投回 7168。注意它不是 qkv 之前的 z-norm，而是注意力输出之后的 out-gate norm。confidence：**高**。

### 1.5 MoE 层 expert 宽度与 shared expert

实现类 `KimiSparseMoeBlock`（762-874）+ `KimiBlockSparseMLP`（242-270）。

**Latent MoE 结构**（`use_latent_moe = routed_expert_hidden_size is not None`，776；K3 `routed_expert_hidden_size=3584`，config:246）：

| 权重 | 形状 | 行号 |
|---|---|---|
| `routed_expert_down_proj` | Linear(7168 → 3584)（token 先压到 expert 隐空间） | 804-806, 821-822 |
| 每个 expert（×896）`w1`(gate)/`w3`(up) | Linear(**3584 → 3072**) | 249, 251（hidden_size=3584, ffn_dim=3072 传入） |
| 每个 expert `w2`(down) | Linear(**3072 → 3584**) | 250 |
| `routed_expert_norm` | KimiRMSNorm(3584)（`latent_moe_use_norm=true`，config:64） | 810-813 |
| `routed_expert_up_proj` | Linear(3584 → 7168) | 807-809, 829-832 |

- **expert intermediate = `moe_intermediate_size = 3072`（config:176）；`routed_expert_hidden_size = 3584` 是 expert 的输入/输出（latent）宽度，不是 intermediate**（见 §4d）。
- expert 激活：`hidden_act="situ"` → `SituAndMul`：`beta·tanh(gate/beta)·sigmoid(gate)·up`，K3 取 `activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`（64-91, 253-258；config:20-21）。
- **Router**（`KimiMoEGate`，666-759）：`weight` (896 × 7168)，`e_score_correction_bias` (896,)，sigmoid 打分（`moe_router_activation_func="sigmoid"`），top-16（`num_experts_per_token=16`），`routed_scaling_factor=1.0`，renormalize=true。
- **Shared expert**（797-801, 836-837）：`num_shared_experts=2` → `KimiMLP` with `intermediate_size = mooe_intermediate_size × 2 = 6144`，即 gate/up 7168→6144、down 6144→7168，输出直接相加（`y = y + shared_experts(identity)`）。
- Layer 0 的 dense MLP（`KimiMLP`）：intermediate_size = **33792**（config:56），gate/up 7168→33792、down 33792→7168。

confidence：**高**。

### 1.6 全注意力层（MLA）宽度（对照组）

实现类 `KimiMLAAttention`（335-474），K3 24 个 MLA 层用：

| 权重 | 形状 | 依据 |
|---|---|---|
| `q_a_proj` | 7168 → 1536（q_lora_rank） | 365-367；config:199 |
| `q_a_layernorm` | RMSNorm(1536) | 368 |
| `q_b_proj` | 1536 → 96×192=18432（q_head_dim = qk_nope 128 + qk_rope 64） | 369-373 |
| `kv_a_proj_with_mqa` | 7168 → 512+64=576（kv_lora_rank + qk_rope） | 378-382 |
| `kv_b_proj` | 512 → 96×(128+128)=24576 | 384-389 |
| `o_proj` | 96×128=12288 → 7168 | 390-394 |
| `g_proj`（`mla_use_output_gate=true`，config:174） | 7168 → 12288，sigmoid 门控在 o_proj 前（470-472） | 398-401 |

confidence：**高**。

---

## 2. 层型混合：93 层中 KDA vs MLA 的判定逻辑与 dense 分界

### 2.1 判定逻辑：不是正则，是**显式 1-based 层号清单**

- `KimiLinearConfig.is_kda_layer`（`configuration_kimi_k3.py:152-156`）：

```python
def is_kda_layer(self, layer_idx: int):
    return (self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"])
```

  即 `kda_layers` 是 **1-based** 清单（`layer_idx+1` 与清单比对）。
- `KimiDecoderLayer.__init__`（`modeling_kimi_linear.py:883-892`）：`is_kda_layer → KimiDeltaAttention`；否则 `config.is_mla → KimiMLAAttention`；否则 `NotImplementedError`。缓存同样按此分 `linear_attention` / `full_attention`（120-147）。

### 2.2 数量统计（config `linear_attn_config`，/tmp/m8v2/kimi_k3_config.json:66-169）

- `kda_layers`：**69 层**（1,2,3,5,6,7,9,10,11,…,89,90,91）。
- `full_attn_layers`：**24 层**（4,8,12,…,92 共 23 个 4 的倍数 + 最后一层 93）。69 + 24 = 93 ✓。
- 模式：**每 4 层中前 3 层为 KDA、第 4 层为 MLA**（1-based 4k），且**最后一层（93）额外换成 MLA**，使最终层为全注意力。confidence：**高**（清单显式给出，代码判定逻辑已核对）。

### 2.3 dense MLP 分界

`modeling_kimi_linear.py:893-900`：

```python
if (config.num_experts is not None
    and layer_idx >= config.first_k_dense_replace     # K3: 1
    and layer_idx % getattr(config, "moe_layer_freq", 1) == 0):   # K3: 1
    self.block_sparse_moe = KimiSparseMoeBlock(config)
else:
    self.mlp = KimiMLP(config)
```

- `first_k_dense_replace = 1`（config:46）→ **仅 layer_idx=0（即 1-based 第 1 层）是 dense MLP**（intermediate 33792），其余 92 层全为 MoE。README 参数表 "Number of Dense Layers = 1"（README:70-71）与源码一致。
- 注意第 1 层同时是 KDA 层（kda_layers 含 1）：**layer0 = KDA + dense MLP**；最后一层（1-based 93，MLA）是 MoE。confidence：**高**。

### 2.4 附加结构：Attention Residuals（AttnRes）

`attn_res_block_size = 12`（config:26）→ `use_attn_residuals=true`（907-917）。每层增加 `self_attention_res_norm/mlp_res_norm`（RMSNorm）与 `self_attention_res_proj/mlp_res_proj`（Linear(7168 → 1)）。`layer_idx % 12 == 0` 时把 prefix_sum 存入 `block_residual`（995-998），后续层通过 `_apply_attn_res`（1075-1088）用 softmax 权重（每残差块一个可学标量打分）对历史 block residual 做加权和回注；模型尾部还有 `output_attn_res_norm/proj`（1103-1108, 1226-1233）。confidence：**高**。

---

## 3. 官方参数量

来源：**Hugging Face 模型卡 `https://huggingface.co/moonshotai/Kimi-K3#model-architecture-overview`（raw: https://huggingface.co/moonshotai/Kimi-K3/raw/main/README.md，本地 /tmp/m8v2/kimi_k3_README.md:40,58-63）**：

| 指标 | 官方数值 |
|---|---|
| **Total Parameters** | **2.8T**（"It is a 2.8T-parameter model built on Kimi Delta Attention (KDA)…"，README:40 与参数表 58-59 双处一致） |
| **Activated Parameters** | **104B**（每 token 激活，README:62-63） |
| 视觉编码器 | MoonViT-V2，401M（README:126-127） |

confidence：**高**（官方模型卡原文）。

---

## 4. 字段语义确认

### a. `num_expert_group = 1`

- 消费点：`KimiMoEGate.forward`（724-746）。只有 `num_expert_group > 1 且 > topk_group` 时才走分组路由（view 成 (n, group, -1) 取组内 top2 求和再选 topk_group 组）；**=1 时该分支完全不触发，等价于无分组的 noaux_tc sigmoid+bias top-k**。`topk_group=1`（config:258）、`topk_method="noaux_tc"`（config:259）。
- 结论：K3 的 896 选 16 **不做 expert 分组限制**，`num_expert_group=1` 是"关闭分组路由"的占位值。confidence：**高**。

### b. `attn_res_block_size = 12`

- 语义：**不是** "每 12 层插一个全注意力层"，而是 **Attention Residual 机制的块大小**：每 12 层（`layer_idx % 12 == 0`）把当前 prefix_sum 追加进 `block_residual` 存档（995-998），非存档层通过可学标量打分的 softmax 注意力回注历史存档（1075-1088）。层型切换由 §2.1 的 kda_layers/full_attn_layers 清单独立决定。confidence：**高**。

### c. `num_key_value_heads = 96`（KDA 层下它是什么）

- `KimiDeltaAttention` **完全不读取** `config.num_key_value_heads`——其头数来自 `linear_attn_config["num_heads"]=96`（486）。grep 全文件，`num_key_value_heads` 只被 `KimiMLAAttention.__init__` 消费（346-347：`num_key_value_groups = num_heads // num_key_value_heads = 96/96 = 1`），且在 MLA forward 里实际未用到 GQA 展开（MLA 是 latent-KV，天然单 KV 组）。
- 结论：在 K3 语境下它是 **MLA 层的占位/兼容字段**（值=头数→ groups=1），对 69 个 KDA 层**无任何作用**；KDA 层的"96 头"来自 `linear_attn_config.num_heads`。confidence：**高**。

### d. `routed_expert_hidden_size = 3584`

- 消费点：`KimiSparseMoeBlock.__init__`（776-813）：`moe_hidden_size = routed_expert_hidden_size`，用作 **latent expert 空间的输入/输出宽度**：`routed_expert_down_proj(7168→3584)` → expert 内部 `w1/w3(3584→3072), w2(3072→3584)` → （可选 RMSNorm）→ `routed_expert_up_proj(3584→7168)`。
- 结论：**它不是 expert intermediate**（intermediate 是 `moe_intermediate_size=3072`），而是 token 进入 expert 前被压缩到的 **latent expert 隐宽度**（把 expert FFN 的输入输出从 7168 降到 3584 以省参，类似把 routed expert 挂在低秩 latent 空间）。confidence：**高**。

---

## 5. 主要结论速览

1. K3 文本塔 = 69×KDA + 24×MLA 共 93 层，每 4 层 3 KDA + 1 MLA，末层（93）也是 MLA；仅第 1 层 dense（33792），其余 92 层 MoE（896 expert / top-16 / +2 shared，latent MoE 3584↔3072）。
2. KDA 层输出门在 `use_full_rank_gate=true` 下为单投影 `g_proj: 7168→12288`；decay 走 `f_a(7168→128)→f_b(128→12288)` 低秩；beta 走 `b_proj: 7168→96`；输出经 sigmoid 门控 RMSNorm（head_dim=128 粒度）后 `o_proj: 12288→7168`。
3. 官方参数量：total **2.8T**，active **104B**。
4. `num_expert_group=1`=关闭分组路由；`attn_res_block_size=12`=AttnRes 存档周期；`num_key_value_heads=96` 对 KDA 层无效（MLA 占位）；`routed_expert_hidden_size=3584`=expert latent 宽度而非 intermediate。
