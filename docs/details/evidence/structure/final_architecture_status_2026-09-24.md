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
| `deepseek_v3` | 2 | 基线/证据不足 | 保留 MLA、MoE、MTP 现有结构 | R1/V3.1 的逐模块发布差异未完成 |
| `deepseek_v32` | 1 | 已修复/已回归 | DSA indexer 的物理路径、主 sparse MLA 与真实 source-ref | 完整权重逐模块 truth 未完成 |
| `deepseek_v4` | 5 | 已修复/有边界 | compressor、C4 indexer、norm/gate/APE 依赖；按 ratio 和变体区分 | C4/C128 fused compression 精确动作成本仍 unknown |
| `deepseek_v41` | 1 | 已修复/有边界 | CED、CSA2 Full/Reuse、DSpark、MTP 抑制基线 | 完整权重/GPU 行为和视觉融合物化仍未证明 |
| `kimi_k2` | 4 | 基线/证据不足 | 保留 MLA/MoE；未因 README 缺少 MTP 就伪造结论 | Thinking QAT、MTP 和逐模块 checkpoint 审计 |
| `kimi_k25` | 3 | 已修复/有边界 | 视觉/文本两路融合入口 | 各变体视觉 projector 和 checkpoint 逐模块审计 |
| `kimi_k3` | 1 | 已修复/有边界 | AttnRes、NoPE、SiTU、Gated MLA、视觉 RoPE、LatentMoE | 局部 header 不是全模型 truth；视觉生命周期仍保留边界 |
| `qwen3_5` | 15 | 已修复/有边界 | Gated Attention / GDN 必需 fan-in；多模态入口 | 量化/Base 变体 packed layout 和视觉 truth |
| `qwen3_5_moe` | 12 | 已修复/有边界 | Gated Attention / GDN fan-in、MoE 路径和融合入口 | 量化/Base 变体逐模块实装证据 |
| `qwen3_5_moe_text` | 2 | 已修复/有边界 | text-only 混合主干；未误加视觉塔；门控输入修复 | 量化变体 checkpoint/activation quantization 细节 |
| `qwen4_exp` | 2 | 已修复/有边界 | QSA 微块、尾部 token、PLE、第 2 层位置、四路 gated residual | 融合搬运 unknown、变体 forward/header 审计 |
| `minimax_m2` | 1 | 基线/证据不足 | 保留当前 GQA/MoE/MTP 结构 | 官方公开资料不足，需继续保持 unknown |
| `minimax_m3_vl` | 2 | 已修复/有边界 | MSA 每 GQA group 独立选块、单共享 index key、local block | 两变体完整 checkpoint 逐模块审计、视觉物化 |
| `glm4_moe` | 1 | 基线/证据不足 | 保留 GQA/MoE，不从 GLM-5 DSA 规则外推 | 官方结构细节不足 |
| `glm_moe_dsa` | 6 | 已修复/有边界 | IndexShare source/reuse schedule、真实 `wq_b/wk/k_norm/weights_proj` 路径 | 完整 checkpoint truth 与量化 scale 归属 |
| `glm5_next` | 2 | 已修复/有边界 | KDA/DSA 混合、四路 mHC/GR、视觉入口 | Flash 的独立 k-pool、变体 truth 与融合物化 |

## 60 个条目覆盖清单

以下列表来自 `models/catalog.json`，每个条目都有对应结构家族记录。括号内
只表示当前主要证据状态，不表示“所有参数均已核验”。

### MiniMax（3）

- `MiniMaxAI/MiniMax-M2.7`（基线/证据不足）
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

### GLM（10）

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
- 全量门禁：前端 579/579、后端 184/184、内置模型 60/60；
  这些证明当前代码链路稳定，不等于表中所有“证据不足”项目已经通过。

## 当前明确未完成项

1. 多模态 39 条目的 projector/merger 与真实 checkpoint 逐模块审计。
2. DeepSeek-V4 C4/C128 fused compressor 的精确动作与缓存状态成本。
3. 所有量化/Base 变体的 packed logical shape、scale 和 activation quantization
   逐模块核对。
4. 16 个结构家族的最终移动端深展开证据整理。
5. GPU/推理框架实测；静态计算结果不替代性能 benchmark。
