# 内置模型架构复核：图库提出检查项，一手资料决定语义

日期：2026-09-23（会话日期）。
审计基线：`84d5beba2240f9a413af4ee22ff926523d60245d`。
范围：60 个内置条目、16 个 model_type；V4.1 为已优化基线，其他 59 个条目为重点。

## 结论

**需要继续优化，不是只有 Kimi-K3 两个门控补线，也不能得出“其他模型结构全部正确”。**

这次先读取外部模型说明/论文，再检查生成的 Graph IR 是否表达那些机制。发现：

1. Kimi-K3：AttnRes 的位置和数据依赖不对；NoPE 被画成执行 RoPE；SiTU 被写成 SwiGLU；MLA 门控既悬空又存在 128/192 宽度冲突。需要家族级修正，而非纯命名。
2. GLM-5.2/5.3：compute/reuse 数量正确，但跨层 IndexShare 没有真实来源边，reuse 层仍画本层索引计算链。
3. Qwen3.5/3.6/3.8：主干混合比例正确，但 Gated Attention/Gated DeltaNet 的门控支路没有完整 fan-in。
4. Qwen3.8-Flash-Next：n-gram 注入层正确；QSA 微块选择和四路 Gated Residual 应独立表达，不能视作普通 HyperConnection 的改名。
5. MiniMax-M3：MSA 的每 GQA 组独立选块这一结构维度丢失；不只是给 sparse attention 换名字。
6. DeepSeek-V4 五个条目：压缩器内部仍有无数据流边的算子；不是 V4.1 已修就全家族已修。
7. 39 个多模态条目都存在视觉输出串入 token embedding 的顶层边。应改为视觉/文本两路在融合节点汇合，而不是把 embedding lookup 当作视觉投影的下游计算。

本次只新增审计文档和只读探针，**未修改产品装配器/渲染代码，未提交、未推送**。

## 取证边界与证据层级

- 图库：已联网读取整页，页面自报 105 models、Sep 21 更新；实际放大查看 Kimi-K3 和 Qwen3.8-Flash-Next 图。其余型号检查卡片元数据；不宣称所有图均逐像素审过。
- 一手资料：浏览器直接读取 **36 个当前内置型号的官方 Hugging Face 模型卡**，另读 Kimi 官方 K3 博客及其中架构示意、MiniMax MSA 论文 HTML §3.1 和官方 MSA 仓库说明。
- 本地：生成全部 60 个模型的当前 Graph IR，检查实际数据流边、节点、输入输出维度和层调度；不以“配置里有这个字段”代替图语义正确。
- 未直接读取的 23 个非 V4.1 变体按同族继承标记，**不能称 59/59 完成独立一手核验**。Base、量化变体也不预设“只改 dtype”；需单独保留 packing、激活量化、草稿头及发布差异。
- Kimi-K3 PDF 下载发生超时/截断，GitHub PDF 页面仅取得元数据，**没有把全文标记为已读**。NoPE 的图库线索由本地随模型入库的官方实现交叉确认；仍需补齐报告页码锚点。
- 网络：本轮 web 搜索接口未返回可引用正文；shell 访问 HF connection reset，但应用内浏览器可正常读取 HF。没有沿用旧草稿“浏览器逐页不可靠，画廊可代理官方”的结论。
- 官方卡片只介绍能力、未披露结构的型号标记证据不足；不是“未发现变化，因此结构通过”。

### 证据层次不能互相替代

| 层次 | 回答的问题 | 不足以证明 |
| --- | --- | --- |
| 官方论文/技术报告/架构图/说明 | 架构意图、残差/共享/索引机制 | 某发布 checkpoint 所有权重均实装 |
| 发布模型卡/版本说明 | 型号与版本的结构差异 | 所有细粒度连线 |
| config/官方实现/checkpoint | 参数、张量形状、实际前向路径 | 图形表示已经忠实；实现中的融合不是独立架构 |
| 当前 Graph IR/浏览器 | 用户实际看到的结构 | 外部机制本身正确 |

## 一手资料索引

以下均为本轮实际读取页面，不是仅找到链接。每型号官方卡的直接读取状态见末尾完整清单。

| ID | 来源 | 本轮用于核验的内容 |
| --- | --- | --- |
| S1 | https://huggingface.co/moonshotai/Kimi-K3 | 69 KDA + 24 Gated MLA；SiTU-GLU；Stable LatentMoE；MoonViT-V2 |
| S2 | https://www.kimi.com/blog/kimi-k3 | 按深度选择历史状态的 AttnRes；官方架构示意；Stable LatentMoE 的并行共享专家分支 |
| S3 | https://huggingface.co/zai-org/GLM-5.2 | IndexShare 每四个稀疏注意力层一组；MTP 更新 |
| S4 | https://huggingface.co/zai-org/GLM-5.3-Flash | 新训练底座，稀疏与线性混合，mHC，原生多模态；不能当 GLM-5.3 的轻量参数版本 |
| S5 | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | GDN + 微块 QSA；四路残差读写门；第二层 bigram/trigram；512 blocks / 2048 tokens |
| S6 | https://huggingface.co/Qwen/Qwen3.5-397B-A17B | 45 GDN + 15 Gated Attention；视觉语言；MTP |
| S7 | https://huggingface.co/Qwen/Qwen3.8-27B | 48 GDN + 16 Gated Attention；原生 VL |
| S8 | https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B | 69 GDN + 23 Gated Attention；开源版本是 text-only，不等于 API Max 的视觉能力 |
| S9 | https://arxiv.org/html/2606.13392 | MSA v2，§3.1、式(5)–(8)：每组 query、单共享 index key、块内 max pooling、每组 Top-k、强制 local block |
| S10 | https://huggingface.co/MiniMaxAI/MiniMax-M3 | MSA 与 M3 发布关联；论文实验 109B 模型不能直接当 M3 参数配置 |
| S11 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash | CSA/HCA、mHC、混合精度的架构概览 |
| S12 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 | 正式版附带 DSpark；不是笼统“所有 V4 都是 MTP” |
| S13 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 | 基于 Pro preview 结构，附带 DSpark |
| S14 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp | 视觉编码器、aligner 和 DSpark 路径；不能用纯文本图库排除视觉结构缺陷 |
| S15 | https://huggingface.co/deepseek-ai/DeepSeek-V3.2 | DSA；不能从普通 `ForCausalLM` 命名推导全部高层结构 |
| S16 | https://huggingface.co/moonshotai/Kimi-K2-Thinking | INT4 QAT 的发布差异，不能笼统写成仅权重/KV dtype 变化 |

图库入口：https://sebastianraschka.com/llm-architecture-gallery/ 。图库是检查线索，不凌驾于一手证据。

## 具体问题与验收建议

### P1-A：Kimi-K3 需要结构级修正

**AttnRes 不是在普通 DecoderLayer 之后追加一个盒子。** S2 明确描述跨深度选择历史表示；当前 `decoderLayer.js:42–61` 仍先走普通两次 residual add，再把 `ffn_residual_add` 接到 `attn_residual`。`residual.js:26–31` 又把 aggregate 接到 norm/projection，方向与“评分 → softmax → 加权聚合”相反。`Output Attention Residual` 只有 norm → proj，缺少产生 hidden 输出的聚合。

本地官方实现补充定位：`models/moonshotai/Kimi-K3/modeling_kimi_linear.py` 的 `_forward_attn_residual` 在 attention 前和 MLP 前分别调用聚合；`_apply_attn_res` 先计算分数，沿 block 维 softmax，再乘历史值。应呈现 snapshot bank / prefix、两个聚合点和最终聚合，维护 block 边界与有条件的 prefix 残差。

**NoPE/RoPE：** 图库标为 Gated MLA with NoPE；当前 MLA helper 无条件发射 `rope`。本地官方实现 `KimiMLAAttention` 明确 `assert self.use_nope`、`self.rotary_emb = None`，前向没有旋转，即使 `qk_rope_head_dim=64` 仍保留在形状配置。**这是不能只看字段名称的直接反例。** 应保留该部分通道形状，但不画执行 RoPE，不影响视觉编码器自己的 2D RoPE。

**SiTU：** S1/S2 指出 SiTU/SiTU-GLU；当前 `mlpOperatorSpecs` 仍生成 `SwiGLU activation`，K3 没有专属 activation 属性，LatentMoE 专家也没有 SiTU 标注。它不是等价改名，需独立公式与适用分支。Stable LatentMoE 的降维、共享专家并联、归一化、升维及相加已存在，不应整体推倒。

**MLA gate：** 生成图内有 46 个孤立 gate 节点（折叠图节点数，**不是 46 层**）。一个输出 `[B,S,96,128]`，另一个 `[B,S,96,192]`。实际应区别 gate projection 和 sigmoid×attention-output，而非直接把两个现有盒子串起来。修复前需回核 checkpoint 路径/尺寸与费用，防止重复计费。

验收：论文机制断言 + NoPE 无执行旋转 + SiTU 公式 + gate fan-in + block bank 来源 + 展开前后路径一致 + 参数/费用回归。

### P1-B：多模态的两路输入被串行化

Graph IR 实测 **39/39 多模态条目**具有 `visual/projector → embed_tokens` 边。公共 `models/common.js` 把顶层 children 顺序直接变成 declared dataflow。官方 S1/S4/S6/S7/S14 的视觉语言描述本身要求分别表达图像与文本来源；本地 K3 官方前向进一步证明先做 token embedding，再将 image features 与之合并。

建议结构：

```text
pixels → vision encoder → merger/projector ─┐
token IDs → token embedding ────────────────┴→ modality merge/scatter → language backbone
```

这是已有图的语义修正；各族 merger 在塔内部还是外部必须保留。图像 token 替换/插入策略逐族确认，不能用一个未经验证的融合算子覆盖所有模型。V4.1 CED 分段正确不代表其视觉入口也免于公共问题。

### P1-C：GLM-5.2/5.3 IndexShare 缺少跨层依赖

S3：**每四层一组，1 次生成 + 后续 3 层复用**，不是“每 3 层一个周期”。

当前 GLM-5.2 逐层 schedule 为 21 compute / 57 reuse，与首层/偏移边界有关，无需凭概述硬改成 78/4。问题在于 `dsaAttentionOperatorSpecs` 在 reuse 层仍无条件创建 query/key projection、norm、indexer，本层数据流相同，仅属性为 reuse。探针中 GLM-5.2、FP8、GLM-5.3、BF16 各有 19 个折叠 reuse 节点，**跨层 indexer→indexer 边为 0**。

应明确 index_source_layer / index group，并让缓存索引有真实来源；显示“索引复用”不等于共享 MLA KV，更不等于共享全部注意力。GLM-5.3 模型卡未详细复述 IndexShare，因此其独立官方版本证据仍须补齐；本地缺陷影响范围可确定为 4 条。

### P1-D：Qwen Gated Attention / Gated DeltaNet 的门控数据未接入

S6/S7/S8 的 gated 结构不是装饰名：

- full attention 当前有 `sdpa → output_gate → o_proj`，但缺 `qkv_gate_split` 的 gate 分量到 `output_gate`。
- GDN 当前有 `state_update → output_gate_norm`，但缺 `qkvz_split` 的 z 分量到 gated norm；不能把 z 当 qkv 一起送进卷积。

这些模块都有节点/公式，所以“无悬空节点”检测也抓不出缺失 fan-in。应以算子具名输入端口校验，不仅校验度数。影响全部 31 个 Qwen 条目的 GDN 分支，full gated 分支影响其中非 Flash-Next 的 29 条。

### P2-A：Flash-Next 架构细节被过度折叠

S5：

- 36 GDN + 12 QSA 已匹配。
- n-gram 模块当前 canonical id `layers.1.ple`，即 **1-based 第 2 层，正确**；不能因为零基层号误改。
- QSA 的预算单位应同时展示 **512 个微块 × 4 token = 2048 token**，不要仅画不带单位的 selected=2048。
- 残差是四路 widened stream，element-wise read gate + per-branch scalar write gate。当前 `HyperConnection` 复合节点有权重/阶段属性，但图面及输入输出 shape 仍是单 hidden，缺少可见四路流和读写分支。建议可展开语义图，保留实现融合信息。
- QSA 主干目前从 sparse attention 直接到 out_proj；是否有 output gate 应结合该发布配置/报告确认后再增节点，不把普通 Qwen Gated Attention 无条件套过来。

### P2-B：MiniMax-M3 的 MSA 需要组维度和两分支表达

S9 §3.1 明确：每 GQA group 有独立 index query，index key 单头共享，先 token score，再块内 max pooling，按组 Top-k，保留 local block，main branch 对这些块做精确注意力。

当前 `minimaxAttentionCommon` 已有 index/main 分支、单头 index key、block_size、local_blocks 属性，这是已正确部分；但 indexer 输出是 `[batch,sequence,selected blocks]`，**没有 group 维**，无法表达各组不同选择。建议补 `[B,S,H_kv,K_blocks]` 的语义形状与 group-shared index 关系，展开 maxpool/Top-k/local union。不把训练 KL loss 默认画成推理主路径。

M3 的 3 full GQA + 57 sparse 是本地图观察；论文 109B 实验模型的层数不能代替生产 M3。MTP 是否实装仍要 checkpoint 证据，不因 config 声明就断言随发布可用。

### P2-C：DeepSeek-V4 压缩路径仍需修复

S11/S12/S13/S14 要求区分 CSA/HCA、mHC、MTP/DSpark 和视觉版本。
当前 V4 Flash/0731/Vision 各有 83 个、Pro/0813 各有 120 个 compressor 相关无 dataflow 边节点（折叠图实测，不等同层数），涉及 compressor norm、indexer compressor projection/norm。

它们是结构图完整性问题，不应放在“低优先级等 HF 网络恢复”。修复需梳理 compressor projection → norm → positional transform / compressed cache → attention/indexer，逐比例和逐 source 条件验证。不同日期变体的 DSpark 不应被家族表一概写成 MTP。

## 按家族的覆盖与决策

| 家族 | 条数 | 外部核验结论与优化范围 |
| --- | ---: | --- |
| DeepSeek R1 / V3.1 | 2 | R1 官方指回 V3-Base；V3.1 卡偏重能力。暂未确认宏观主干重构，不是逐边通过 |
| DeepSeek V3.2 | 1 | DSA 已有；保留 token 索引与 MLA 主路径区别，未发现 CED 证据 |
| DeepSeek V4 | 5 | CSA/HCA+mHC 主干已有；修 compressor；正式日期版核验 DSpark；Vision 另修融合 |
| DeepSeek V4.1 | 1 | 已优化 CED 基线；本轮不重验其全部 CSA2；仍受公共视觉入口问题影响 |
| Kimi K2 | 4 | MLA MoE 主干说明一致；Thinking QAT 单列；不能用 README 未提 MTP 证明绝对无 MTP |
| Kimi K2.5/2.6/2.7 | 3 | 官方视觉编码器/MLA 信息已读；修视觉与文本汇合，未发现要改 CED |
| Kimi K3 | 1 | P1 家族级修正：AttnRes/NoPE/SiTU/gate，保留已有 LatentMoE 正确部分 |
| Qwen3.5 dense | 15 | 各尺寸/3.6/3.8 dense 官方层布局已读；修 gated fan-in、视觉融合 |
| Qwen3.5 MoE | 12 | 35B/122B/397B、3.6 主型号已读；修 gated fan-in、视觉融合；量化/Base 独立核验待补 |
| Qwen3.8 text MoE | 2 | 2.4T 主型号 92 层、text-only；修 gated fan-in，不添加 API Max 的视觉塔 |
| Qwen Flash-Next | 2 | GDN/QSA+n-gram 已有；修门控/融合，补四路残差和微块语义 |
| MiniMax M2.7 | 1 | 官方卡未披露足够结构；不下全图正确/MTP 已实装结论 |
| MiniMax M3 | 2 | MSA 论文直读；修按组索引形状与融合；发布版 MTP 待实装证据 |
| GLM4 MoE | 1 | GLM-4.7 官方卡已读但结构细节不足；暂保留 GQA MoE，不判全图通过 |
| GLM DSA | 6 | 5/5.1/5.2/5.3 主型号已读；5.2/5.3 4 条需显式复用关系；不是 Flash 的 KDA+mHC |
| GLM5 Next | 2 | Flash 官方明确 hybrid+mHC+VL；主干已有，修融合并补 linear/KDA 与 sparse/DSA 命名、k-pool/NoPE 执行语义复核 |
| **合计** | **60** | 所有条目有本地图观察；外部证据级别分别记录，不输出统一“结构正确”勾选 |

## 可复现检查

```bash
node scripts/evidence/structure/audit-gallery-semantics.mjs > /tmp/msv-gallery-observations.json
node scripts/verify-builtin-models.mjs
```

探针只读配置并生成 Graph IR，无网络、无模型实例化、无权重下载，输出覆盖范围、调度、顶层边、孤立 gate/compressor 等。它**不是架构正确性 oracle**。

本轮已有检查器输出 **60/60 passed**；与上述缺陷并存，说明现有验证主要覆盖“能生成图、算子注册、基本节点存在”，不能证明论文语义正确。

未进行：产品代码修复、完整单测/构建回归、当前 viewer 浏览器逐模型验收、GPU 执行。浏览器取证针对外部资料，不冒充本地 UI 验收。

建议实施顺序：K3 专属语义 → 公共多模态/门控 fan-in → GLM IndexShare → V4 compressor → QSA/MSA 语义展开。
每批都需论文/官方约束测试、Graph IR 断言、形状/参数/费用回归及实际浏览器展开/折叠验证；不能仅重生成 golden 让测试通过。

## 对旧草稿的处理

未覆盖用户原有未提交文件 `gallery-structure-audit.md`。其中这些结论应撤回或降级：

- “唯一定义问题是 K3 gate”：漏了 AttnRes、NoPE、SiTU、多模态融合等。
- “其他无 V4.1 级重构”：没有充分依据排除 K3 这种家族级拓扑问题；也不能凭 Decoder type 一列排除未知机制。
- “逐层 attn-mix 一致即所有结构正确”：不检查语义依赖。
- “GLM IndexShare 每 3 层”：正确是四层组内复用后三层。
- “量化仅权重/KV dtype”：QAT/激活精度/packing/发布模块不能合并成此断言。
- “复现临时脚本已删除”：本轮新增持久只读脚本，避免不可复现。

## 逐条外部核验清单（不是结构通过表）

图观察全部完成；“直读”仅说明官方卡片本轮读到，不能替代论文/逐算子验证。

| 型号 | 外部证据状态 | 本轮官方卡入口 |
| --- | --- | --- |
| MiniMaxAI/MiniMax-M2.7 | 官方卡直读，结构披露有限 | https://huggingface.co/MiniMaxAI/MiniMax-M2.7 |
| MiniMaxAI/MiniMax-M3 | 官方卡直读 | https://huggingface.co/MiniMaxAI/MiniMax-M3 |
| MiniMaxAI/MiniMax-M3-MXFP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-0.8B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| Qwen/Qwen3.5-0.8B-Base | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-122B-A10B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.5-122B-A10B |
| Qwen/Qwen3.5-122B-A10B-FP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-27B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.5-27B |
| Qwen/Qwen3.5-27B-FP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-27B-GPTQ-Int4 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-2B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.5-2B |
| Qwen/Qwen3.5-2B-Base | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-35B-A3B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.5-35B-A3B |
| Qwen/Qwen3.5-35B-A3B-Base | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-35B-A3B-FP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-397B-A17B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.5-397B-A17B |
| Qwen/Qwen3.5-397B-A17B-FP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-4B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.5-4B |
| Qwen/Qwen3.5-4B-Base | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.5-9B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.5-9B |
| Qwen/Qwen3.5-9B-Base | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.6-27B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.6-27B |
| Qwen/Qwen3.6-27B-FP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.6-35B-A3B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.6-35B-A3B |
| Qwen/Qwen3.6-35B-A3B-FP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.8-2.4T-A95B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B |
| Qwen/Qwen3.8-2.4T-A95B-FP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.8-27B | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.8-27B |
| Qwen/Qwen3.8-27B-FP8 | 同族线索，变体未独立直读 | — |
| Qwen/Qwen3.8-Flash-Next | 官方卡直读 | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| Qwen/Qwen3.8-Flash-Next-FP8 | 同族线索，变体未独立直读 | — |
| deepseek-ai/DeepSeek-R1 | 官方卡直读 | https://huggingface.co/deepseek-ai/DeepSeek-R1 |
| deepseek-ai/DeepSeek-V3.1 | 官方卡直读，结构披露有限 | https://huggingface.co/deepseek-ai/DeepSeek-V3.1 |
| deepseek-ai/DeepSeek-V3.2 | 官方卡直读 | https://huggingface.co/deepseek-ai/DeepSeek-V3.2 |
| deepseek-ai/DeepSeek-V4-Flash | 官方卡直读 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash |
| deepseek-ai/DeepSeek-V4-Flash-0731 | 官方卡直读 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 |
| deepseek-ai/DeepSeek-V4-Flash-Vision-Exp | 官方卡直读 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp |
| deepseek-ai/DeepSeek-V4-Pro | 官方卡直读 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro |
| deepseek-ai/DeepSeek-V4-Pro-0813 | 官方卡直读 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 |
| deepseek-ai/DeepSeek-V4.1-Flash | 已优化基线，本轮未独立重验 | — |
| moonshotai/Kimi-K2-Base | 官方卡直读 | https://huggingface.co/moonshotai/Kimi-K2-Base |
| moonshotai/Kimi-K2-Instruct | 同族线索，变体未独立直读 | — |
| moonshotai/Kimi-K2-Instruct-0905 | 官方卡直读 | https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905 |
| moonshotai/Kimi-K2-Thinking | 官方卡直读 | https://huggingface.co/moonshotai/Kimi-K2-Thinking |
| moonshotai/Kimi-K2.5 | 官方卡直读 | https://huggingface.co/moonshotai/Kimi-K2.5 |
| moonshotai/Kimi-K2.6 | 官方卡直读 | https://huggingface.co/moonshotai/Kimi-K2.6 |
| moonshotai/Kimi-K2.7-Code | 官方卡直读 | https://huggingface.co/moonshotai/Kimi-K2.7-Code |
| moonshotai/Kimi-K3 | 官方卡直读 | https://huggingface.co/moonshotai/Kimi-K3 |
| zai-org/GLM-4.7 | 官方卡直读，结构披露有限 | https://huggingface.co/zai-org/GLM-4.7 |
| zai-org/GLM-5 | 官方卡直读 | https://huggingface.co/zai-org/GLM-5 |
| zai-org/GLM-5.1 | 官方卡直读，结构披露有限 | https://huggingface.co/zai-org/GLM-5.1 |
| zai-org/GLM-5.2 | 官方卡直读 | https://huggingface.co/zai-org/GLM-5.2 |
| zai-org/GLM-5.2-FP8 | 同族线索，变体未独立直读 | — |
| zai-org/GLM-5.3 | 官方卡直读，结构披露有限 | https://huggingface.co/zai-org/GLM-5.3 |
| zai-org/GLM-5.3-BF16 | 同族线索，变体未独立直读 | — |
| zai-org/GLM-5.3-Flash | 官方卡直读 | https://huggingface.co/zai-org/GLM-5.3-Flash |
| zai-org/GLM-5.3-Flash-BF16 | 同族线索，变体未独立直读 | — |
