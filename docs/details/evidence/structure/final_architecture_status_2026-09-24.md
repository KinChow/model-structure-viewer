# 全量模型结构修复状态表

日期：2026-09-24。范围：当前 catalog 的 60 个内置条目、16 个
`model_type`。Graph IR 保持 v2。本表是**证据和修复状态台账**，不是把
所有条目统一标记为“架构正确”。

## 状态定义

- **已修复/已回归**：已存在明确外部或发布实现证据，代码、机制测试、
  成本对账和相关浏览器检查已完成。
- **已修复/有边界**：结构问题已修复，但仍有 checkpoint 逐模块、融合搬运、
  fused kernel 成本或其他明确的 unknown。
- **基线/证据不足**：当前没有足够的一手资料支持新的拓扑修改；保留现有
  结构，不把 config-only 结果写成架构验证通过。
- **待审计**：需要补充独立 forward、header/skeleton 或变体资料。

## 16 个结构家族

| model_type | 条目数 | 当前状态 | 已完成的结构工作 | 剩余边界 |
|---|---:|---|---|---|
| `deepseek_v3` | 2 | 已修复/有边界 | 对照发布 `DeepseekV3Attention` 修正 MLA 的真实 `q_a_layernorm`、`kv_a_proj_with_mqa`、`kv_a_layernorm` 路径；保留前三层 dense、后续 MoE；MTP 尾层 61 的 `embed_tokens/eh_proj/shared_head` 路径已绑定到草稿分支 | 完整逐张量 truth、量化 scale 归属、MTP experts 逐张量绑定和 GPU fused kernel 仍 unknown |
| `deepseek_v32` | 1 | 已修复/已回归 | DSA indexer 的物理路径、主 sparse MLA 与真实 source-ref；DeepSeek-style MTP 尾层路径复用绑定规则 | 完整权重逐模块 truth、MTP experts 逐张量绑定未完成 |
| `deepseek_v4` | 5 | 已修复/有边界 | compressor、C4 indexer、norm/gate/APE 依赖；按 ratio 和变体区分；C4 首窗口 overlap 成本已按 pinned forward 修正 | GPU fused kernel、量化 scale 交通、跨请求 cache 追加 prefill 仍 unknown |
| `deepseek_v41` | 1 | 已修复/有边界 | CED、CSA2 Full/Reuse、DSpark、MTP 抑制基线 | 完整权重/GPU 行为和视觉融合物化仍未证明 |
| `kimi_k2` | 4 | 已修复/有边界 | 对照 Kimi 发布 `DeepseekV3` 实现修正 MLA 精确模块路径；Base/Thinking 的 index manifest 与 header 均确认无 MTP，不生成 MTP | Thinking QAT、完整逐模块 checkpoint truth、量化 scale 和 GPU 行为仍 unknown |
| `kimi_k25` | 3 | 已修复/有边界 | 视觉/文本两路融合入口；视觉塔、projector 和 3 个变体的 checkpoint header/source audit 已通过 | 融合搬运 unknown；GPU/融合物化未实测 |
| `kimi_k3` | 1 | 已修复/有边界 | AttnRes、NoPE、SiTU、Gated MLA、视觉 RoPE、LatentMoE；发布 `PatchMergerMLPV2` 无 bias 且后置 RMSNorm | 局部 header 不是全模型 truth；视觉生命周期仍保留边界 |
| `qwen3_5` | 15 | 已修复/有边界 | Gated Attention / GDN 必需 fan-in；多模态入口 | 量化/Base 变体 packed layout 和视觉 truth |
| `qwen3_5_moe` | 12 | 已修复/有边界 | Gated Attention / GDN fan-in、MoE 路径和融合入口 | 量化/Base 变体逐模块实装证据 |
| `qwen3_5_moe_text` | 2 | 已修复/有边界 | text-only 混合主干；未误加视觉塔；门控输入修复 | 量化变体 checkpoint/activation quantization 细节 |
| `qwen4_exp` | 2 | 已修复/有边界 | QSA 微块、尾部 token、PLE、第 2 层位置、四路 gated residual；视觉 packed attention、grid_thw 位置插值、pre-shuffle merger norm | 融合搬运 unknown；GPU packed-attention 性能未实测 |
| `minimax_m2` | 1 | 已修复/有边界 | 按随附 Transformers/source-ref 修正为独立 Q/K/V projection、Q/K RMSNorm、partial RoPE；保留 GQA/MoE；index manifest/header 均确认 config 声明的 MTP 未落地 | 融合搬运、量化 scale traffic、完整 GPU 行为仍 unknown |
| `minimax_m3_vl` | 2 | 已修复/有边界 | MSA 每 GQA group 独立选块、单共享 index key、local block；发布权重的 `multi_modal_projector → patch_merge_mlp` 双模块路径；两变体视觉 header/source audit 已通过 | 融合搬运 unknown；GPU/视觉物化未实测 |
| `glm4_moe` | 1 | 已修复/有边界 | 对照 GLM-4.7 source-ref 修正独立 Q/K/V、Q/K RMSNorm、partial RoPE 和 attention bias；MTP 尾层 92 的 `embed_tokens/eh_proj/shared_head` 路径已绑定；未把 GLM-5 DSA 外推到 GLM-4.7 | 完整逐模块 truth、量化 scale 和 GPU fused kernel 仍 unknown |
| `glm_moe_dsa` | 6 | 已修复/有边界 | IndexShare source/reuse schedule、真实 `wq_b/wk/k_norm/weights_proj` 路径 | 完整 checkpoint truth 与量化 scale 归属 |
| `glm5_next` | 2 | 已修复/有边界 | KDA/DSA 混合、四路 mHC/GR、GLM-5.3-Flash 独立视觉塔；两变体视觉 header/source audit 已通过；MTP 尾层 45 按发布 manifest 绑定为标准 decoder，不误继承主干 mHC，且不虚构本地 embedding/shared-head projection；MTP routed expert tensor 已按 288 个 folded expert 的完整 manifest 聚合绑定到 fused expert leaf | 融合搬运 unknown；MTP expert 的 GPU fused 物化、GPU/视觉物化未实测 |

## 60 个条目覆盖清单

以下列表来自 `models/catalog.json`，每个条目都有对应结构家族记录。括号内
只表示当前主要证据状态，不表示“所有参数均已核验”。

### MiniMax（3）

- `MiniMaxAI/MiniMax-M2.7`（已修复/有边界；独立 Q/K/V 路径）
- `MiniMaxAI/MiniMax-M3`（已修复/有边界）
- `MiniMaxAI/MiniMax-M3-MXFP8`（已修复/有边界；生产 `mtp_tensor_count=0`）

### Qwen（31）

- `Qwen/Qwen3.5-0.8B`、`Qwen/Qwen3.5-0.8B-Base`
- `Qwen/Qwen3.5-122B-A10B`、`Qwen/Qwen3.5-122B-A10B-FP8`、
  `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`
- `Qwen/Qwen3.5-27B`、`Qwen/Qwen3.5-27B-FP8`、
  `Qwen/Qwen3.5-27B-GPTQ-Int4`
- `Qwen/Qwen3.5-2B`、`Qwen/Qwen3.5-2B-Base`
- `Qwen/Qwen3.5-35B-A3B`、`Qwen/Qwen3.5-35B-A3B-Base`、
  `Qwen/Qwen3.5-35B-A3B-FP8`、`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
- `Qwen/Qwen3.5-397B-A17B`、`Qwen/Qwen3.5-397B-A17B-FP8`、
  `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`
- `Qwen/Qwen3.5-4B`、`Qwen/Qwen3.5-4B-Base`
- `Qwen/Qwen3.5-9B`、`Qwen/Qwen3.5-9B-Base`
- `Qwen/Qwen3.6-27B`、`Qwen/Qwen3.6-27B-FP8`
- `Qwen/Qwen3.6-35B-A3B`、`Qwen/Qwen3.6-35B-A3B-FP8`
- `Qwen/Qwen3.8-2.4T-A95B`、`Qwen/Qwen3.8-2.4T-A95B-FP8`
- `Qwen/Qwen3.8-27B`、`Qwen/Qwen3.8-27B-FP8`
- `Qwen/Qwen3.8-Flash-Next`、`Qwen/Qwen3.8-Flash-Next-FP8`

### DeepSeek（9）

- `deepseek-ai/DeepSeek-R1`、`deepseek-ai/DeepSeek-V3.1`
- `deepseek-ai/DeepSeek-V3.2`
- `deepseek-ai/DeepSeek-V4-Flash`、`deepseek-ai/DeepSeek-V4-Flash-0731`、
  `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- `deepseek-ai/DeepSeek-V4-Pro`、`deepseek-ai/DeepSeek-V4-Pro-0813`
- `deepseek-ai/DeepSeek-V4.1-Flash`

### Kimi（8）

- `moonshotai/Kimi-K2-Base`、`moonshotai/Kimi-K2-Instruct`、
  `moonshotai/Kimi-K2-Instruct-0905`、`moonshotai/Kimi-K2-Thinking`
- `moonshotai/Kimi-K2.5`、`moonshotai/Kimi-K2.6`、
  `moonshotai/Kimi-K2.7-Code`
- `moonshotai/Kimi-K3`

### GLM（9）

- `zai-org/GLM-4.7`
- `zai-org/GLM-5`、`zai-org/GLM-5.1`、`zai-org/GLM-5.2`、
  `zai-org/GLM-5.2-FP8`、`zai-org/GLM-5.3`、`zai-org/GLM-5.3-BF16`
- `zai-org/GLM-5.3-Flash`、`zai-org/GLM-5.3-Flash-BF16`

> Qwen 31 + DeepSeek 9 + Kimi 8 + MiniMax 3 + GLM 9 = 60。catalog 当前
> 总数以机器清单为准；本段按家族列举时不将同一条目重复计入。最终计数
> 校验命令：

```sh
node scripts/verify-builtin-models.mjs
```

## 证据来源和验证入口

- Raschka architecture gallery：用于发现 Decoder type、Attention、Layer mix
  和 Key detail 的交叉检查，不覆盖所有 fused kernel 细节。
- 各型号官方 Hugging Face model card：见
  `variant_card_review_2026-09-24.md/.json`。
- 官方/发布实现和 pinned source：见各家族的
  `*_repair.md` 与 `*_sources.json`。
- config-only 与 production artifacts：由各家族机制测试分别加载；
  不用总参数量替代逐模块绑定。
- 全量门禁：前端 608/608、后端 184/184、内置模型 60/60；
  Kimi-K3 MLA/AttnRes 与 39 条多模态入口的机制测试通过；Chrome
  全量展示巡检覆盖 60/60。视觉模型的 roofline `unknown` 是融合搬运
  未知的显式结果，不是空图或页面错误。
  这些证明当前代码链路稳定，不等于表中所有“证据不足”项目已经通过。

## 当前明确未完成项

1. DeepSeek-V4 C4/C128 fused compressor 的 GPU kernel 实测、量化 scale
   交通与 overlap state 的精确成本。
2. 所有量化/Base 变体的 packed logical shape、scale 和 activation quantization
   逐模块核对。
3. 16 个结构家族的最终移动端深展开证据整理。
4. GPU/推理框架实测；静态计算结果不替代性能 benchmark。

### 多模态 checkpoint 审计结果（2026-09-24）

39 个多模态条目已通过独立 header/source audit：

```text
total=39, verified=39, gap=0, unknown=0
```

该结果证明视觉 tower、projector、merger 的发布 tensor 能绑定到生产 Graph
IR，不能推导出 image placeholder scatter、融合搬运流量或 GPU kernel 时间；
这些仍按 unknown/未实测边界保留。
