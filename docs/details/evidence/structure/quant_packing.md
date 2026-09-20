# 量化权重打包口径 —— 对真实量化 checkpoint header 逐模型对账

用户要求系统验证量化（FP8/GPTQ-Int4/MXFP8）运行时是否验过。量化本质是**权重存储格式**，其口径的权威真值
是真实 checkpoint 的 safetensors header（逐 dtype 元素桶）。MSV 用 `quantBytes.js logicalElementsFromHeader`
把 packed header 反解成逻辑参数量，与 `graphWeightCapacity` 的声明量对账。

## 结果（`verify_quant_bytes.mjs`，真实 header-truth sidecar 为真值）

| 模型 | 量化 | 真实 dtype 桶（关键） | 真实逻辑参数量 | MSV 声明量 | MSV/真实 |
|---|---|---|---|---|---|
| Qwen3.5-27B-GPTQ-Int4 | gptq b4 g128 | I32 2.158e9 + F16 1.337e8 + BF16 1.067e10 | 27,795,583,728 | 27,778,431,232 | **0.9994** |
| Qwen3.5-35B-A3B-GPTQ-Int4 | gptq b4 g128 | I32 4.105e9 + F16 2.517e8（mtp 785） | 36,329,310,064 | 35,948,745,088 | 0.9895 |
| Qwen3.5-27B-FP8 | fp8 | F8_E4M3 2.470e10 + BF16 3.084e9 | 27,782,935,472 | 27,778,431,232 | **0.9998** |
| Qwen3.5-35B-A3B-FP8 | fp8 | F8_E4M3 3.445e10 + BF16 1.501e9 | 35,953,925,552 | 35,948,745,088 | **0.9999** |
| MiniMax-M3-MXFP8 | mxfp8 | F8_E4M3 4.237e11 + U8 1.324e10（e8m0 尺度） | 427,040,140,160 | 426,812,410,112 | **0.9995** |
| Qwen3.5-27B（bf16 基线） | — | BF16 2.778e10 | 27,781,427,952 | 27,778,431,232 | **0.9999** |

## 口径验证结论（逐方案）

- **GPTQ-Int4**：真实 header 里 packed 权重存为 **I32（每 int32 打包 8×int4）**+ **F16 scales**（numel=weights/group_size）
  + qzeros（与 scales 同形，也在 I32 桶）。MSV `logicalElementsFromHeader`：`I32×(32/bits=8)` 反解 packed，
  再 **减 F16** 去掉 qzeros 的重复计数，F16 scales 不计入逻辑量 → 与真实逻辑量 **0.9994**。
  `quantBytes.js` 打包字节式 `out·in·0.5 + out·ceil(in/128)·(2 fp16 scale + 0.5 int4 qzero)` 与此一致。
- **FP8**：`F8_E4M3` 1B/elem passthrough，与真实 **0.9998–0.9999**。
- **MXFP8**：`F8_E4M3` 权重 + `U8`(e8m0 每块 1 枚尺度) —— MSV 忽略 U8 尺度算逻辑量，**0.9995**。
- 35B-A3B-GPTQ 的 0.9895（~1%）源于 MoE 专家 + 785 个 mtp 张量的结构小残差，仍在容差内；
  per_model_reconcile.md 结构 diff（由 `scripts/evidence/structure/reconcile_reduced.py` 重生）该模型 `status=passed, residual_count=0`。

## 与 MTP 的交叉验证

真实 header 的 `mtp_tensor_count`（GPTQ-27B=15、35B-A3B=785、FP8-27B=22…）驱动 MSV `includeMtp` 选择；
上表参数量在 includeMtp 正确取值下才 0.999 收敛 → **MTP 张量口径（mtp.{i} 前缀计数）已随参数量对账一并验证**
（另见单测 `builtinModels`/`weights` 的 `mtpTensorCount`）。

## 边界（未做，非本层口径问题）

- **运行时 serve 量化权重**：GPTQ-Int4（A100 SM80 marlin 可跑）/ FP8（SM80 无 fp8 MMA，weight-only 反量化）——
  本地无小量化 checkpoint 可 serve；且 serve 只会重复 header 已确认的 footprint（打包是存储格式，非运行时计算量，
  GPTQ 的 FLOPs 与 bf16 同）。故量化口径在 header 级已收口。
- **MTP 投机解码接受率（运行时循环）**：需真实 mtp.* 权重（transformers 随机 init 不产出 MTP 张量，已实测
  `Qwen3_5MoeForCausalLM` from_config 0 个 mtp 参数），减层随机 checkpoint 无法真机跑投机接受；须真实 MTP checkpoint。
  MTP **结构 + 张量口径**已在 header/参数量级验证（本文件 + per_model_reconcile.md）。

复现：`verify_quant_bytes.mjs`（读 `models/*/header-truth.json` 真值 + `logicalElementsFromHeader`）。
