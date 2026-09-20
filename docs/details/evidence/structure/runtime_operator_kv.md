# 运行时对账：算子 / 精度 / 压缩 KV vs MSV（2026-09-19，A100）

配合 `deepseek-gate-bias-attn-sink` 改动后，对 MSV 的算子结构、精度(dtype)、压缩 KV 做**运行时**（非静态）复核。

## 环境
- 8×A100-SXM4-80GB（SM80, cc 8.0）；`$SGLANG` torch 2.13.0+cu130、sglang dev、transformers 5.12.1（含原生 deepseek_v3/v32/v4、glm4_moe、minimax_m2、qwen3_moe/qwen3_next 类）。
- 工具：`$MODELS/_reduced/runtime_probe.py`（transformers-eager 前向 + module 钩子，抓算子类/激活 dtype/参数 dtype/KV 结构）；SGLang `launch_server`（压缩 KV 分配报告）。
- 减层件：`$MODELS/_reduced/*_tiny`（bf16，由真实 config 缩层缩维，`build_families_tiny.py` / `build_v32.py`）。

## 一、算子 + 精度（transformers-eager，7 家族）
| 家族 | 运行时关键算子类 | 激活 dtype | 与 MSV |
|---|---|---|---|
| deepseek_v3 | RMSNorm, Linear, SiLU, DeepseekV3TopkRouter | bf16 + fp32×3 | MLA+MoE 算子一致 |
| deepseek_v32(DSA) | + LayerNorm(indexer k_norm), indexer.{wq_b,wk,weights_proj} | bf16 + fp32×3 | DSA indexer 结构一致（k_norm=LayerNorm→MSV affine_bias:true 印证）|
| deepseek_v4 | DeepseekV4GroupedLinear(o_lora), SqrtSoftplusActivation, UnweightedRMSNorm | bf16 + **fp32×26** | 印证 wo_a/wo_b 分组输出投影、sqrtsoftplus 路由、hc/ape/norm 的 fp32 dtype |
| glm4_moe | Glm4MoeTopkRouter | bf16 + fp32×3 | correction-bias 路由一致 |
| minimax_m2 | MiniMaxM2TopKRouter | bf16 + fp32×6 | 一致 |
| qwen3_moe | Qwen3MoeTopKRouter | bf16 | 一致 |
| qwen3_5_moe | **Conv1d + RMSNormGated** + TopKRouter | bf16 | 线性(GDN)层：conv+gated-norm 且**该层无 KV**→印证 MSV recurrent-state 建模 |

## 二、压缩 KV（SGLang serve，6 家族，全部对上 MSV）
口径说明：transformers-eager 存**解压后** KV，SGLang 存**部署压缩** KV（=MSV 口径），故 KV 数值以 SGLang 为准。

| 家族 | SGLang 分配 KV | MSV kvBytesPerToken | 对账 |
|---|---|---|---|
| deepseek_v3 (MLA, triton) | 0.05 GiB/8192 | 6912 (kv_lora512+rope64)×6×2 | ✅ 0.0527 GiB |
| deepseek_v32 (DSA, dsa 后端) | 0.04 GiB/8192 | 5376 (MLA latent320 + index128)/层 | ✅；前向 indexer 撞 fp8→H20 |
| glm4_moe (GQA) | K0.02+V0.02 GiB | 6144 (2×128×6×2×2) | ✅ |
| minimax_m2 (GQA) | K0.02+V0.02 GiB | 6144 | ✅ |
| qwen3_moe (GQA) | K0.06+V0.06 GiB | 16384 (8×128×4×2×2) | ✅ 0.125 GiB |
| qwen3_5_moe (线性+全注意力) | KV K0.01+V0.01 GiB；**Mamba conv0.83+ssm17.74GB** 单列 | 2048 (KV) + recurrent state/seq | ✅ 口径拆分一致 |

## 三、A100 硬边界（已实测为证，归 H20）
- **fp8**：deepseek_v32 DSA 前向 `dsa_indexer.py: act_quant(query,...,fp8)` → `ValueError: type fp8e4nv not supported in this architecture (SM80 只支持 fp8e5/fp8e4b15)`。DSA topk / V4 fp4 的真实量化前向必须 H20。
- **VL remote-code**：minimax_m3 / glm5_next 是多模态 `...ForConditionalGeneration`，SGLang 启动需自定义 processor/modeling（`processing_minimax.py` 等），减层 dummy 目录缺文件；glm5_next 另会撞同款 fp8-DSA。二者 cache 口径由既往 N-task 33/34 覆盖。

## 四、订正与残留
- minimax_m3 MSV KV：修正首版减层 config 的 bug（per-layer 数组未截断到 6 层，残留长度 60）后，MSV 给 **~13056 B/tok**（3 dense×1024 + 3 sparse×(1024+128)），量级正常，非 MSV 缺陷；SGLang 实测因 VL 墙未取到，待 H20。
- V4 compressor 内部（norm/wkv_gate/coff，本 spec 已改）与 fp4 KV 的运行时数字待用户在 H20 复核。

## 证据artifact
- 减层件/日志：`$MODELS/_reduced/{deepseek_v3_tiny,deepseek_v32_tiny,glm4_moe_tiny,minimax_m2_tiny,qwen3_moe_tiny,qwen3_5_moe_tiny}`、`v3_serve_new.log`、`v32_serve.log`、`kv_{glm4_moe,minimax_m2,qwen3_moe,qwen3_5_moe}.log`、`runtime_probe.py`。
- 静态回归：`node --test` 422/422、`verify:models` 60/60、`docs:check` 一致、`check_principles §8.1` 6/基线6。
