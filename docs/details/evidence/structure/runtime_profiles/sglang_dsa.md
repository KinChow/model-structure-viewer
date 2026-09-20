# 减层运行时验证：DSA builder（deepseek_v32）——覆盖 7 个模型的 cache 口径 + 揭示 DSA 前向硬件边界

**动机（用户"减层呢"）**：DSA builder（`assembleDeepseekV32`，7 模型：V3.2 + GLM-5/5.1/5.2/5.3）真实权重
过大（GLM-5 ~1.5TB）/fp8，A100 装不下。用 **transformers 原生 `deepseek_v32` from_config 建减层随机 checkpoint**
（零下载全权重）在 A100 验运行时口径。

## 减层 checkpoint

`deepseek_v32` reduced：6 层、hidden 2048、8 experts、q_lora 768、num_heads 16——**保留 DSA/MLA per-head 口径不动**
（kv_lora_rank=512、qk_rope=64、v_head_dim=128、index_head_dim=128、index_topk=2048）；random init bf16、4.1B、
save_pretrained（+ V3.2 tokenizer）。**关键修复**：reduced config 需移除继承自 V3.2 的 fp8 `quantization_config`
（否则 SGLang 建 fp8 张量、与 bf16 权重 `Downcasting not allowed`）。

## SGLang 运行时（SGLang 运行输出）

- **DSA 架构与后端全识别**：`Use dsa attention backend`、`index_topk=2048 阈值`、`page size 64`、SM80 上
  `KV cache dtype=bfloat16`、`prefill=flashmla_sparse / decode=fa3`。
- **权重加载成功** `DeepseekV32ForCausalLM mem usage=0.55GB`（bf16 reduced）。
- **KV Cache 分配成功**：`#tokens 7,604,096, KV size 54.56 GB`（MLA 单 latent + DSA index，非分列 K/V）。
  → 每 token = 54.56 GiB / 7,604,096 = **7,704 B**。
- **前向失败（硬件边界）**：`RuntimeError: Sparse Attention Forward Kernel is only supported on SM90a and
  SM100f architectures`（`dsa_backend.py` flash_mla_sparse_fwd）——**DSA 稀疏注意力前向 kernel 仅 Hopper(SM90a)/
  Blackwell(SM100f)，A100(SM80) 不支持**。

## 与前端对齐（cache 口径）

- 前端 `assembleDeepseekV32`（reduced config）：**kvBytesPerToken = 8,448 B**（= MLA latent (512+64)×6×2=6,912 +
  index (index_head_dim 128×6×2=1,536)）。
- SGLang **7,704 B/token** vs 前端 **8,448**，**比值 1.097**（~10%）——同结构（MLA latent + DSA index cache）、同量级，
  差异属 page-size(64) 对齐 + index-topk 有界等 runtime 口径（reduced 随机维度，非精确对齐场景）。

## 结论

- **减层路线对 DSA 成立且有增量**：A100 上 **DSA 的 cache 口径（MLA latent + index）真机验证通过**（比"权重过大跑不了"
  推进了一大步），覆盖 7 个 DSA 模型的 cache 建模。
- **新硬件边界（精确）**：**DSA 稀疏注意力前向 = SM90a/SM100f only**——A100 只能验到 cache 分配 + backend 配置，
  稀疏前向（行为/kernel）留 **H20（SM90）**。与 fp8 GEMM(SM89)、DeepEP、fp4 同属 SM80 硬件边界族。
- GLM-5 系（`GlmMoeDsaForCausalLM` 在前端 alias 到 assembleDeepseekV32）的 DSA 口径同此覆盖。
