# SGLang 第二架构 profile：DeepSeek-V2-Lite（真实 bf16 MLA+MoE）

继 Qwen3-0.6B（GQA）之后补第二架构族 **MLA + MoE**。V2-Lite（bcecmd 下载，30GB，bf16、A100 可前向；
V4.1/V3 因 fp8 在 A100 跑不了，用同族 bf16 的 V2-Lite 做 MLA 运行时代理）。SGLang `--tp 8 --ep-size 8`
（SGLang 运行输出）。config：`deepseek_v2`、kv_lora_rank=512、qk_nope=128/qk_rope=64、v_head_dim=128、
n_routed_experts=64、topk=6、n_shared=2、first_k_dense_replace=1、27 层。

## 运行时观测

- **每卡权重** `DeepseekV2ForCausalLM mem usage=3.76 GB`（总 ~30GB / 8 卡，TP8+EP8 分片）。
- **MLA KV 签名（关键）**：日志 `KV Cache is allocated ... KV size: 62.33 GB, #tokens: 2,151,682`——
  **单个合并 "KV size"（非 GQA 的分列 K/V）**，即 MLA 只存**压缩 latent**。反推每 token：
  62.33 GiB / 2,151,682 = **31,104 B/token = 27 层 × (kv_lora_rank 512 + qk_rope 64=576) × 2B**——
  **与 MLA 压缩 latent 结构精确一致**。对比 Qwen3-0.6B(GQA)：分列 K+V = kv_heads(8)×head_dim(128)×2(K,V)×2B×层，
  每 token 112 KiB——两架构 KV 口径本质不同（MLA latent 576/层 vs GQA 2048/层），**第二架构族运行时签名坐实**。
- **MoE**：ep8 → 每 rank `E=8`（=64/8）完整专家、`N=1408`（=moe_intermediate，moe_tp=1 未切）——见 `../../parallelism/ep48.md`。

## 与前端口径对齐

- MSV 前端**未内置 `DeepseekV2ForCausalLM`**（`buildStructureFromConfig` resolution=**unsupported**，kvBytesPerToken=0）——
  V2-Lite 非 catalog 内置模型。故对账为 **运行时 MLA vs MLA latent 结构公式**（= 前端已支持的 `DeepseekV3`/dsv4
  家族 MLA 建模所用的同一 latent 口径：cache = kv_lora_rank + qk_rope）。
- 结论：SGLang MLA 运行时**单 latent 缓存 = kv_lora_rank + rope**（31,104 B/token 精确），验证了前端 MLA 建模的
  latent KV 口径在真实 MLA 模型上成立；补齐了"仅 Qwen3-0.6B(GQA)"的第二架构族（MLA）真机证据。
- 诚实边界：V2-Lite 未接入 MSV 前端（unsupported），本项验证的是 **MLA 运行时口径**（latent 组成/单缓存），
  非 V2-Lite 的前端结构图逐点对；MLA 前端结构图的对账由已支持的 V3/dsv4 家族承担。
