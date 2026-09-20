# assembleQwen3_5 的 MoE 分支（qwen3_5_moe，与已验 dense 结构不同）

用户纠正：我此前只真机跑了 **dense** 的 Qwen3.5-4B（`intermediate_size=9216`，无专家），而
`assembleQwen3_5` 的 **MoE 分支**（`Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text`：256~512 专家 + shared expert
+ router + GatedDeltaNet 线性 hybrid）结构完全不同，**从未被直接真机跑过**。qwen4_exp 虽复用同一
`moeOperatorSpecs`，但外面裹了 PLE/hyper-connection，不能代替 qwen3_5_moe 纯 MoE 层。本项补真机。

## 减层 checkpoint（transformers 原生，无需 remote code）

`build_qwen3_5_moe_tiny.py`（基于 `Qwen/Qwen3.8-2.4T-A95B` 配置缩小，`$SGLANG` transformers 5.12.1
`from_config`→随机→`save_pretrained`，81M）：
- 8 层 hybrid：`layer_types = [linear,linear,linear,full] × 2`（保留 `full_attention_interval=4`）
- 线性(GatedDeltaNet)：`linear_num_key_heads=4, linear_key_head_dim=64, linear_num_value_heads=8,
  linear_value_head_dim=64, linear_conv_kernel_dim=4`
- 全注意力(GQA)：`num_attention_heads=8, num_key_value_heads=2, head_dim=128, attn_output_gate=true`
- **MoE：`num_experts=8, num_experts_per_tok=2, moe_intermediate_size=256, shared_expert_intermediate_size=256`**

## 真机结果（SGLang，A100，端到端跑通）

```
Load weight end ... type=Qwen3_5MoeForCausalLM
Mamba Cache is allocated. max_mamba_cache_size: 26652, conv_state size: 0.92GB, ssm_state size: 19.52GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 8192, K size: 0.01 GB, V size: 0.01 GB
Using hybrid linear attention backend for hybrid GDN models.
Using default MoE kernel config ... E=8,N=256 ...      # 8 专家 + moe_intermediate 256 的 MoE runner 生效
The server is fired up and ready to roll!
# /generate input_ids=[1,10,20,30,40,50] -> output_ids=[1443,1319,1080,275,1530,914,535,764]（8 token，含 decode）
```

MoE 路由/dispatch/combine + shared expert + GatedDeltaNet 线性 + 全 GQA + decode **全部真机执行**。

## cache 口径逐点对账（真机分配 vs MSV）

| 量 | SGLang 真机 | MSV 前端 | 差 |
|---|---|---|---|
| 线性(GDN) state /层/请求 | 35,840 elems（conv 3,072 bf16 + ssm 32,768 fp32） | 35,840 | **0.0%** |
| 全注意力 GQA kv /token/层 | 512 elems（2·kv_heads·head_dim=2·2·128） | 512 | **0.0%** |

- 线性 state 反推：conv_state 0.92GB = 3,072·2B·**6 线性层**·26,652 req = 0.92GB ✓；
  ssm_state 19.52GB = 32,768·4B(fp32)·6·26,652 = 19.53GB ✓。
  SGLang `qwen3_5.py:359 conv_dim = key_dim·2 + value_dim = 256·2+512 = 1024`（×(K-1)=3 → 3,072）；
  ssm = num_value_heads·value_head_dim·key_head_dim = 8·64·64 = 32,768 → 合计 **35,840** =
  MSV `linearStateResidentDecl`（conv `4·64·2+8·64=1024`×3 + `8·64·64` = 35,840）。
- MoE 结构：MSV `Qwen3_5MoeSparseMoeBlock` = router+topk+dispatch+expert_mlp(fused_moe)+combine +
  shared_experts(Qwen3_5MoeMLP)+shared_expert_gate+shared_expert_add，与 SGLang MoE runner（E=8,N=256）一致。

## 结论

- **`assembleQwen3_5` 的 MoE 分支（qwen3_5_moe）已真机跑通**（不再只覆盖 dense 路径）：MoE router/专家/shared
  expert + GDN 线性 hybrid + 全 GQA cache 口径与 SGLang 运行时 **0.0%**。
- dtype 细节：conv=bf16、ssm=fp32（mamba_ssm_dtype），元素口径一致，字节按 dtype 拆分。
- 对审计表修正：qwen3_5 从"真机(仅 dense)"细化为"**真机(dense + MoE 分支均已验)**"。

复现：`build_qwen3_5_moe_tiny.py` → SGLang serve（skip-tokenizer-init）→ `/generate` + `msv_predict_qwen_moe.mjs`。
