# glm4_moe / minimax_m2 / deepseek_v3 —— 三个此前未直接真机的独立 builder

用户指出 MiniMax-M2 系列、DeepSeek-V3.1、GLM-4 系列结构各不相同、是否都真机跑过。核查属实：此前
`assembleMiniMaxM2`/`assembleGlm4Moe` 仅"假设同 GQA 口径"、`assembleDeepseekV3` 仅经 V2-Lite 代理，
**均未直接真机**。本项对三者建减层 bf16 checkpoint 并 SGLang 真机跑通。

## 减层 checkpoint（transformers 原生，去 auto_map/去 fp8）

`build_families_tiny.py`（`$SGLANG` transformers，`from_config`→随机→`save_pretrained`）：
- **glm4_moe_tiny**（GLM-4.7，本就 bf16）：6 层（first_k_dense_replace=3 → 3 dense+3 MoE），hidden 1024,
  GQA(8/2 heads, head_dim 128, partial_rotary 0.5, use_qk_norm), 8 routed + 1 shared expert, top-2 → 50.6M
- **minimax_m2_tiny**（MiniMax-M2.7，去 fp8）：6 层全 full-attn，GQA(8/2), num_local_experts 8, top-2,
  sigmoid 路由 + routing_bias, qk_norm per_layer → 171M
- **deepseek_v3_tiny**（DeepSeek-V3.1，去 fp8）：6 层，MLA(q_lora 512, kv_lora 512, qk_nope 128, qk_rope 64,
  v_head 128), 8 routed + 1 shared, top-2, n_group=1 → 58.9M

## 真机结果（SGLang，A100，均端到端出 token）

| builder / 模型 | SGLang type | cache 分配 | MoE runner | /generate |
|---|---|---|---|---|
| Glm4MoeForCausalLM / GLM-4.7 | ✅ | GQA KV（K+V 0.04GB） | **E=9**（8 routed+1 shared 融合）,N=256 | ✅ 6 token |
| MiniMaxM2ForCausalLM / MiniMax-M2.7 | ✅ | GQA KV（K+V 0.04GB） | **E=8**,N=1024 | ✅ 6 token |
| DeepseekV3ForCausalLM / DeepSeek-V3.1 | ✅ | **MLA latent** KV 0.05GB | **E=8**,N=256 | ✅ 6 token |

## cache 口径对账（真机 vs MSV）

| builder | 量 | SGLang 真机 | MSV | 差 |
|---|---|---|---|---|
| glm4_moe | GQA kv/token/层 | 512（2·kv_heads(2)·head_dim(128)） | 512 | **0.0%** |
| minimax_m2 | GQA kv/token/层 | 512（2·2·128） | 512 | **0.0%** |
| deepseek_v3 | MLA latent kv/token/层 | 576（kv_lora 512 + qk_rope 64） | 576 | **0.0%** |

- glm4_moe：K 0.02GiB / 8192 tok / 6 层 = kv_heads·head_dim·2B = 2·128·2 = 512B/侧… K+V=512 elems ✓；
  MoE E=9 = 8 routed + 1 shared expert 融合（SGLang shared-expert-fusion），dispatch/combine 真机执行。
- deepseek_v3：MLA 单一压缩 latent 池 KV 0.05GB = 576·2B·6 层·8192 tok ✓；MoE + MLA absorb 前向真机跑通
  （首轮用过小 MLA 维触发 triton absorb `make_shape_compatible` 报错，恢复标准 MLA 维 qk_nope=128/v_head=128/
  kv_lora=512 后通过——减层需保持 MLA 维对齐，非模型问题）。

## 结论 + 对审计表修正

- **MiniMax-M2（minimax_m2）、GLM-4（glm4_moe）、DeepSeek-V3.1（deepseek_v3）三个独立 builder 均已直接真机跑通**，
  cache 口径与 MSV 0.0%，MoE 路由/dispatch/combine + 共享专家 + MLA/GQA 全部真机执行。
- 从"△ 假设同 GQA 口径 / V2-Lite 代理"升级为"**✅ 减层真机**"。
- DeepSeek-V3.2（DSA）此前已减层验（`sglang_dsa.md`）；至此 DeepSeek 家族 V3.1(MLA)/V3.2(DSA)
  两个 builder 均真机。真实本体仍 fp8+过大（全前向留 H20），本项收口的是结构 + cache 口径真机。

复现：`build_families_tiny.py {glm4_moe|minimax_m2|deepseek_v3}` → SGLang serve（skip-tokenizer-init）→
`/generate` + `msv_predict_families.mjs`。
