# 全量内置模型结构审计（对照官方 / Raschka 画廊）

- 日期：2026-09-24
- 参考：Raschka LLM Architecture Gallery（https://sebastianraschka.com/llm-architecture-gallery/ ，已取每型号卡片 Decoder type / Attention / Layer mix / Key detail 字段）
- 真值策略：拓扑以本仓 `config.json` + `model.py` + 装配器为准；官方文档在揭示 config 隐含的真实结构特征时采纳（V4.1 CED 教训）；画廊为参考、非可覆盖真值。
- 取证方式：headless `buildStructureFromConfig` 探针导出各家族顶层树 / 注意力族 / 逐层 attn-mix / indexer 调度，与画廊卡片逐条对账。
- **本阶段 read-only，不改装配器/渲染代码。**
- 网络说明：sandbox 侧 huggingface.co 不可达（curl/Playwright 均 HTTP 000 / ERR_CONNECTION_RESET）。**改用用户本机 Chrome（内网可达 HF）重新验证**：成功加载 `deepseek-ai/DeepSeek-V3.2` 官方 README —— front matter `library_name: transformers`、`base_model: DeepSeek-V3.2-Exp-Base`（finetune），确认为标准 transformers 的 DeepSeek-V3 衍生（DSA 稀疏），**非 encoder-decoder**，与画廊/本仓一致。V4.1 官方 README+技术报告已于此前轮次取得。画廊卡片由作者依官方 config/论文整理，作为强代理。
- 浏览器取证限制：用户 Chrome 多窗口/多标签且 scroll AX 动作不稳定，逐 README 系统滚动阅读不可靠；因此其余家族的“官方 README/论文”核验以画廊详卡（= 官方 config/论文的蒸馏）+ 本仓 config/model.py 为准，V3.2 的直读作为交叉印证。网络放开 sandbox HF 后可一次性补全（见 P3）。

### Chrome 重新验证结论（2026-09-24 补充）
- HF 经用户 Chrome 可达（sandbox 不可达），DeepSeek-V3.2 README 直读确认审计判定（DSA、非 CED）。
- 同时在 Chrome 实机核对了本仓 V4.1-Flash 渲染：`Causal Encoder ×20 / Decoder ×20`、`CED：解码器全局 KV 由末端编码器隐状态投影` 关系边、`CSA2(2,Full)/(2,Reuse)` 标签、正交连线均正确呈现。
- 关键结论不变：**除 V4.1-Flash 外无第二个 encoder-decoder/CED 模型；逐层 attn-mix 与画廊 1:1 吻合；唯一真实结构缺陷为 Kimi-K3 Gated MLA 门控悬空（P1）。**

## 总结论
- **逐层 attn-mix 与画廊完全吻合**（见下表），本仓对每个家族的注意力族判定、MoE/MTP/视觉塔/混合线性/超连接调度都正确。
- **没有任何其他家族需要 V4.1 级别的结构重构**：画廊 “Decoder type” 字段里只有 DeepSeek-V4.1-Flash 是 causal encoder-decoder；其余均为 Sparse MoE / Dense-Sparse hybrid，无隐藏的编码-解码或 CED 类特征。
- 唯一“真实结构缺陷（A）”：**Kimi-K3 的 Gated MLA 门控算子悬空未连线**（与 dsv4 注意力此前的悬空同类）。
- 其余为可选“标注/可读性（B）”：把注意力族按官方/画廊术语命名（Gated DeltaNet / KDA / Gated Attention / DSA / MiniMax Sparse / Gated MLA / LatentMoE），不改拓扑。

## 优先级清单
- **P1（需修复 / 真实连线缺陷）**：Kimi-K3 `MLA Attention` 的 `MLA output gate`、`MLA full-rank output gate` 两个算子无任何兄弟连线（悬空瓷砖）。建议按 dsv4 注意力同款做法逐层补线（门控 → 输出投影 / o_proj），并在形状 oracle 登记语义边。
- **P2（可选 / 命名可读性，不改拓扑）**：统一注意力族显示名以贴合画廊/论文——`linear`→按家族显示 “Gated DeltaNet”(Qwen) / “Kimi Delta Attention (KDA)”(Kimi-K3、GLM-5.3-Flash)；`qwen35_full`→“Gated Attention”；`dsa_sparse_mla`→“MLA + DeepSeek Sparse (DSA)”；`sparse`(minimax)→“MiniMax Sparse”；Kimi-K3 `mla`→“Gated MLA”；MoE→按型号注记 “LatentMoE”等。
- **P3（低 / 待 HF 恢复后核验的变体增量，均疑似已正确建模）**：
  - DeepSeek-V4-Flash/Pro（compressorApe）注意力内 `compressor latent RMSNorm`、`indexer compressor wkv/gate`、`indexer compressor RMSNorm` 仍悬空（本仓既有、需三段引用才能连；仅 V4 系，V4.1 无此问题）。
  - Kimi-K2 无 MTP（config 未声明；画廊 K2 亦不提 MTP）——确认属实、非缺失。
  - GLM-5.2 “IndexShare”（DSA indexer 结果跨 3 层复用）——本仓 `indexerSchedule` 已按 compute/reuse 建模（GLM-5.2 实测 21 compute + 57 reuse），仅缺显式标注。
  - Qwen3.8-Flash-Next n-gram 嵌入所在层（画廊称 layer 2）——本仓 `ple` 已建模，核验层号标注。

## 逐层 attn-mix 对账（本仓探针 vs 画廊 Layer mix）

| 型号 | 本仓 attn-mix | 画廊 Layer mix | 一致 |
| --- | --- | --- | --- |
| Qwen3.8-27B | 48 linear + 16 qwen35_full | 16 gated attention + 48 DeltaNet | ✓ |
| Qwen3.8-Flash-Next | 36 linear + 12 qsa | 12 QSA + 36 DeltaNet | ✓ |
| Kimi-K3 | 69 linear + 24 mla | 69 KDA + 24 Gated MLA | ✓ |
| GLM-5.3-Flash | 34 linear + 11 qsa | 34 KDA + 11 MLA/DSA (+1 MTP) | ✓ |
| MiniMax-M3 | 3 gqa + 57 sparse | 3 full GQA + 57 MiniMax Sparse | ✓ |
| GLM-5.2 | 78 qsa（idx 21 compute + 57 reuse） | 78 MLA/DSA（IndexShare 每 3 层复用） | ✓ |
| DeepSeek-V4-Pro | 61 dsv4 | 61 CSA/HCA | ✓ |

## 按家族台账（覆盖全部 60 条；同拓扑变体只标增量）

| model_type（装配器） | 条数 | 代表 / 变体 | 本仓结构 | 画廊要点 | 判定 |
| --- | --- | --- | --- | --- | --- |
| deepseek_v3 (assembleDeepseekV3) | 2 | DeepSeek-R1, V3.1 | MLA + MTP + MoE | 61 MLA, Sparse MoE, MTP | N |
| deepseek_v32 (assembleDeepseekV32) | 1 | DeepSeek-V3.2 | DSA sparse MLA + MTP | MLA + DeepSeek Sparse, 保留 MTP | N |
| deepseek_v4 (assembleDeepseekV4) | 5 | V4-Flash×3, V4-Pro×2 | dsv4(CSA/HCA) + mHC + MTP | 61 CSA/HCA + mHC, 384 exp 6+1, hash 路由, MTP | N（P3: compressorApe 子算子悬空） |
| deepseek_v41 (assembleDeepseekV41) | 1 | V4.1-Flash | CED 两段 + CSA2 + Engram + DSpark | Causal encoder-decoder | **已优化** |
| kimi_k2 (→assembleDeepseekV3) | 4 | K2 Base/Instruct/0905/Thinking | MLA MoE, 无 MTP | 61 MLA, 比 V3 更多专家/更少 MLA 头 | N（P3: 确认无 MTP） |
| kimi_k25 (assembleDeepseekV3 复用) | 3 | K2.5/2.6/2.7-Code | 视觉 + MLA MoE | 原生多模态, K2/DeepSeek MoE, 256k | N |
| kimi_k3 (assembleKimiK3) | 1 | Kimi-K3 | linear(KDA)+mla+Output Attn Residual+视觉 | 69 KDA + 24 Gated MLA, Block Attn Residuals, LatentMoE | **Y（P1: Gated MLA 门控悬空）** |
| qwen3_5 (assembleQwen3_5) | 15 | 0.8B–9B, 3.6-27B, 3.8-27B(VL) | linear + qwen35_full 混合 (+视觉) + MTP | 16 gated attn + 48 DeltaNet（3:1 混合） | N（P2: 命名） |
| qwen3_5_moe (assembleQwen3_5) | 12 | 35B-A3B, 122B, 397B, 3.6-35B | 同上 + MoE + 视觉 + MTP | Next-style 混合注意力 | N（P2: 命名） |
| qwen3_5_moe_text (assembleQwen3_5) | 2 | 3.8-2.4T-A95B | 混合 + MoE（无视觉） | 同族 text | N |
| qwen4_exp (assembleQwen4Exp) | 2 | Qwen3.8-Flash-Next | linear + qsa + 四路超连接 + ple(n-gram) + MTP + 视觉 | 12 QSA + 36 DeltaNet, 四路门控残差, 20M n-gram | N（P2/P3: 命名+n-gram层号） |
| minimax_m2 (assembleMiniMaxM2) | 1 | MiniMax-M2.7 | GQA + MTP + MoE | 62 GQA QK-Norm + 3 MTP | N |
| minimax_m3_vl (assembleMiniMaxM3) | 2 | M3, M3-MXFP8 | 视觉 + GQA + block-sparse + MTP | 3 full GQA + 57 MiniMax Sparse, 128 exp 4+1 | N |
| glm4_moe (assembleGlm4Moe) | 1 | GLM-4.7 | GQA + MTP + MoE | GLM-4.5 风格（MLA 迁移前） | N |
| glm_moe_dsa (→assembleDeepseekV32) | 6 | GLM-5/5.1/5.2(+FP8)/5.3(+BF16) | DSA sparse MLA + MTP | 78 MLA + DeepSeek Sparse + MTP；5.2 IndexShare | N（P3: IndexShare 标注） |
| glm5_next (assembleGlm5Next) | 2 | GLM-5.3-Flash(+BF16) | linear(KDA) + dsa + mHC + 视觉 + MTP | 34 KDA + 11 MLA/DSA + 四路 mHC, 原生多模态 | N（P2: 命名） |

合计 60 条：deepseek 系 9 + kimi 系 8 + qwen 系 31 + minimax 系 3 + glm 系 9。

## 变体（quant/尺寸/VL）结构增量说明
- FP8 / GPTQ-Int4 / MXFP8 / BF16 变体：仅改权重/KV **dtype**，拓扑与基型号一致（本仓 `kvCacheDtype`/量化字段驱动），不单列。
- Base vs Instruct/Thinking：同拓扑（训练差异，无结构变化）。
- VL vs text（如 Qwen3.8-27B VL、MiniMax-M3、Kimi-K2.5/K3、GLM-5.3-Flash）：多 `Vision Tower`(+`Projector`/`merger`) 前置，其余骨干一致；画廊图多省略视觉塔与 MTP，与我们“可展开视觉塔”呈现不冲突。

## 复现命令
- 顶层结构探针：`node frontend/e2e/audit_probe.tmp.mjs <model_id...>`（临时脚本，已随审计删除；逻辑=`buildStructureFromConfig` 后打印顶层树/注意力族/特殊类型）。
- 逐层 attn-mix：`normalizeConfig` + `attentionScheduleOf`/`indexerScheduleOf` 统计。
- 悬空节点审计：对每个父模块检查是否有“无任何兄弟连线”的子节点（Kimi-K3 命中 Gated MLA 门控）。

## 下一步（待逐项确认后执行，非本阶段）
1. P1 修复 Kimi-K3 Gated MLA 门控连线（+ 形状 oracle 登记）——沿用 V4.1 门控：单测 + 重生成受影响 golden + `vite build` + Chrome 核验 + 完整 e2e。
2. P2 注意力族/MoE 术语化命名（纯显示，跨家族一次性）。
3. P3 网络恢复后补 HF README/论文核验，落实 IndexShare/n-gram 层号/K2-MTP 等标注。
