# assembleKimiK3（linear KDA + MLA hybrid）cache 口径运行时对账

用户"减层"路线：Kimi-K3 本体（VL, 93 层, KDA 线性 + MLA, mxfp4）过大无法在 A100 全跑。
对**文本塔** `KimiLinearForCausalLM`（SGLang 原生 `kimi_linear.py`，arch=kimi_linear）建减层随机
checkpoint，直接对 KDA/MLA 两个 cache 池的**运行时 sizing 代码路径**对账 —— 这正是 SGLang scheduler
分配 Mamba 池 + KV 池所用的同一段代码。

## 减层 checkpoint

`$MODELS/_reduced/build_kimi_linear_tiny.py`（`from_config` → 随机 init → `save_pretrained`）：
- text 塔 8 层：`full_attn_layers=[4,8]`(1-idx) → MLA，其余 6 层 KDA；`num_heads=16, head_dim=128,
  short_conv_kernel_size=4`；MLA `kv_lora_rank=512, qk_rope_head_dim=64`；experts 减到 8、vocab 4096。
- remote code（`modeling_kimi_linear.py`）建模需 transformers 4.5x（`OutputRecorder`/`check_model_inputs`
  在 5.12.1/5.17.0 已删）——用隔离 `--target` 安装 `transformers==4.57.6`（+ tokenizers 0.22.2 /
  hf-hub 0.35.3 / einops）仅供建 checkpoint；serve 用 `$SGLANG`（transformers 5.12.1）原生 kimi_linear。
- 减层适配（不影响 cache 口径）：去 `attn_res_block_size`（SGLang 不建 `mlp_res_norm`）、`hidden_act`→silu、
  去 `quantization_config`（mxfp4）。

## 运行时 cache 口径（SGLang 自身代码路径，`$SGLANG` python）

`KimiLinearConfig` → `KimiLinearStateShape.create(tp=1, ...)` + MLA latent = `kv_lora_rank+qk_rope_head_dim`：

```
num_hidden_layers 8
linear (KDA) layer ids     : [0, 1, 2, 4, 5, 6]
full-attn (MLA) layer ids  : [3, 7]
KDA conv shape     : (3, 6144) -> 18432 elems      # (K-1) × (proj_q + proj_k + proj_v) = 3 × (2048+2048+2048)
KDA temporal shape : (16, 128, 128) -> 262144 elems # num_heads × head_dim × head_dim
KDA state per layer/req total elems : 280576
MLA latent kv per token/layer elems : 576 (kv_lora_rank 512 + qk_rope 64)
```

## 前端（MSV）预测 · `assembleKimiK3`

`node msv_predict_kimi.mjs`（arch=KimiK3ForConditionalGeneration，同维 text_config）：

```
linear: state_elements=280576   [layers.0.self_attn.state_update]
mla   : cache_kv_elements=576    [layers.3.self_attn.sdpa]
```

- `linearStateResidentDecl`：conv `keyHeads·keyDim·2 + valueHeads·valueDim = 6144`，×(kernel−1=3)=18432；
  recurrent `valueHeads·valueDim·keyDim = 262144`；合计 **280576**。
- MLA `cacheResidentDecl.kvElements = kvLoraRank + qkRopeHeadDim = 512+64 = 576`。

## 对账结论

| 量 | SGLang 运行时 sizing | MSV 前端 | 差 |
|---|---|---|---|
| KDA state /层/请求（conv+temporal） | 280,576 elems (18,432+262,144) | 280,576 | **0.0%** |
| MLA latent kv /token/层 | 576 elems | 576 | **0.0%** |
| KDA/MLA 分层 | KDA=[0,1,2,4,5,6], MLA=[3,7] | 同 | ✅ |

**`assembleKimiK3` 的 linear(KDA)+MLA 两 cache 口径逐点与 SGLang 运行时分配代码一致（0.0%）。**

## 边界

- 全服务端 boot 在 `load_weights` 命中随机 checkpoint 的权重 layout 不匹配
  （`IndexError: Dimension out of range ... got 2`，某 3-D 权重 vs SGLang 期望 2-D）——与 cache sizing
  正交。cache 口径改由 SGLang 自身 cache-param 代码路径（scheduler 分配两池所用同一段）直取，已 0.0% 对齐。
- 真机全前向（KDA gated-delta kernel、MLA）仍待 H20/完整权重；本项收口的是 **cache 口径**。

复现：`build_kimi_linear_tiny.py`（建减层）→ 上面的 `KimiLinearStateShape.create` 片段（SGLang 口径）
+ `msv_predict_kimi.mjs`（前端口径）。

## 追加（2026-09-19）· KDA 线性**前向**真机尝试 —— 卡在 HF↔SGLang 权重参数化分歧

目标：把 kimi_k3 从 cache 口径升到端到端前向真机（KDA 线性前向不需 SM90，理论上 A100 可跑）。逐个排障：
1. **A_log IndexError 已修**：HF 存 `A_log=[num_heads]=[16]`（1-D），SGLang `KimiDeltaAttention.A_log` 是
   `[1,1,num_heads,1]` 且 `sharded_weight_loader(2)` 在 dim 2 narrow → 1-D 报 `IndexError dim 2`。
   converter reshape `[16]→[1,1,16,1]` 后越过。
2. **KDA 投影参数化分歧（未解，非硬件问题）**：`KeyError g_proj.weight`。根因——SGLang `kimi_linear.py:565`
   构造 `KimiDeltaAttention` 用默认 `no_kda_lora=False`，bf16 下走 **branch-2 融合**（`fused_qkvbfg_a_proj`
   = q,k,v,b + **f_a,g_a**；`fused_fg_b_proj` = f_b,g_b），即 f、g **都按 LoRA(rank=head_dim=64)** 融合存储；
   而 HF remote-code checkpoint 是 **g 全秩(`g_proj[2048,2048]`) + f 按 rank-128 LoRA(`f_a[128,2048]/f_b`)**、
   且 q/k/v/g/b 全分离。二者 KDA 投影的**秩与融合结构根本不同**（full-g/rank128-f-separate vs 融合 rank64-f+g-lora），
   要对齐须对随机权重做 SVD 降秩 + 重融合的有损重参数化，脆弱且数值无意义。
- **结论**：kimi_k3 的 KDA **cache 口径已 0.0% 验证（本文件上半）**；**前向真机**受 HF-remote-code 与 SGLang
  kimi_linear 的 KDA 投影参数化分歧阻塞（**非 A100 硬件限制**，KDA 线性前向本身 SM80 可跑）。要前向真机须
  从头构造符合 SGLang 融合布局的 checkpoint（超出减层随机 checkpoint 的合理范围）。A_log 修复与根因见
  `fix_kimi_conv.py`。
