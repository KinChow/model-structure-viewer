# Raschka LLM Architecture Gallery cross-check

- 读取日期：2026-09-24
- 来源：<https://sebastianraschka.com/llm-architecture-gallery/>
- 用途：作为外部架构索引和家族关系交叉检查，不替代发布实现、配置或 checkpoint truth。

## 本次核对结果

页面当前将以下关系明确作为架构对照点：

- **DeepSeek R1**：建立在 V3 架构上，而不是一套新的基础 block；因此本仓保留 DeepSeek V3 的 MLA/MoE 主干，仅按 R1 发布实现和权重 truth 处理 MTP/变体边界。
- **Kimi K2**：描述为对 DeepSeek V3 配方的规模化延伸；因此本仓复用 MLA/MoE 语义，但仍以 Kimi 自己发布的模块路径和配置为准，不能用展示图替代 checkpoint 对账。
- **MiniMax M2**：描述为回到 full attention、较稀疏的 Qwen 风格 MoE；这支持 GQA/MoE 家族归类，但不支持把 runtime kernel fusion 画成 fused checkpoint QKV。随附 Transformers/source-ref 仍是独立 `q_proj/k_proj/v_proj` 的一手依据。
- **GLM-4.7**：页面将其描述为仍接近较早 GLM-4.5 风格、尚未进入后续 MLA 转向；这与本仓保留 GQA/MoE、独立 Q/K/V 和 Q/K norm、而不套用 GLM-5 DSA 的处理一致。
- **DeepSeek V3.2**：页面描述为保留 V3 模板并加入 sparse attention；这支持 DSA 是增量注意力机制，而不是把整个模型替换成另一种 MLA 主干。
- **GLM-5.2**：页面描述为 sparse MoE backbone 加 IndexShare；这与本仓独立 source/reuse 层、跨层 `index-reuse` 关系的组网一致。
- **Kimi K3**：页面描述为 KDA/Gated MLA 混合、Block Attention Residuals、Stable LatentMoE 和视觉模型；这与本仓 K3 的 KDA/MLA、AttnRes、LatentMoE 和视觉分支拆分相符。
- **Qwen3.8-Flash-Next**：页面描述为 GDN、micro-block QSA、四路 gated residual 和 n-gram embeddings；本仓继续按 QSA 与 MSA 不同的选择机制建模。
- **DeepSeek V4.1-Flash**：页面描述为 CSA2 cache sharing、Engram 和 causal encoder-decoder MoE；本仓保留 CED、CSA2 Full/Reindex/Reuse、Engram 和 DSpark 的独立边界。

## 使用边界

Gallery 是外部比较和术语校验来源，页面的简化图不证明具体张量名、权重是否存在、量化布局或融合 kernel 成本。具体修复仍必须同时通过发布 forward/source-ref、checkpoint header/skeleton 和独立机制测试；证据不足的部分继续标记 unknown。
