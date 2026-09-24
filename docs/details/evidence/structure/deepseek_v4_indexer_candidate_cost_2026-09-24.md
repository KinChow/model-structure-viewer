# DeepSeek V4 C4 indexer 的压缩候选域

日期：2026-09-24。范围：五个 V4 发布变体的嵌套 C4 indexer；
不套用到 V4.1 flat approximation 或普通 DSA。

## 前向依据

发布参考实现 `transformers v5.16.1` 的 `modeling_deepseek_v4.py` 中，
`DeepseekV4Indexer.forward` 先对自身的 `kv_proj` 和 `gate_proj` 做 C4
窗口压缩与归一化，随后调用 `scorer(q, compressed_kv, hidden_states)`。
`DeepseekV4IndexerScorer.forward` 的矩阵乘使用压缩 key 历史；
`index_scores` 生成后才屏蔽未来块，然后选 Top-k。这个顺序不等同于
“原始 sequence 长度的逐 token index key”，也不等同于在 matmul
之前套普通 token-level 三角密度。

## 本次修正

`dsv4_indexer` 提取器的嵌套 C4 分支以
`floor(sequence / compress_ratio)` 为候选 key 数，decode 仍以已可见
历史的总压缩 key 数计读；scorer 的 matmul 在 causal-invalid 屏蔽前
覆盖本轮 query 与压缩候选的矩形。新压缩 index key 仅在闭合窗口时
写入，decode 非边界不虚构写入。公共 DSA、MSA 与 V4.1 flat
索引器不继承这三项差异。

发布 Flash 配置 `batch=1, decode S=4096` 的单层 indexRead 从按原始
4096 个 key 估的 `4,194,304 B` 变为按 1024 个压缩 key 估的
`1,048,576 B`（index_head_dim=512，动作字节口径 `2 B/element`）。
这只是 **indexer 算子候选域**，不是全模型成本变化结论。

随后补齐了 V4 压缩链的计费归属：`compressor` 与嵌套 `indexer` 标记
`billing_mode=children`，真实投影、位置 bias、窗口 softmax/加权归约、
RMSNorm、压缩 RoPE 和 indexer scorer 子节点各自承担动作；父节点不再
与子节点重复计费。C4 indexer 的 RoPE 节点同时表达压缩 key 与当前
query 两次应用，外层 HCA/CSA compressor 只表达压缩 key。

## 仍未封闭的成本边界

窗口归约与 fused kernel 的真实片上复用、FP8/FP4 量化执行字节、
缓存 overlap 状态，以及 GPU 实测时延仍未知；本批只是理论动作/强制
流量账本，不声称 fused kernel 性能或全模型实测成本。

核验：config-only 与生产 artifacts 的 C4 几何断言；Flash 小工作点
`prefill S=8` 的矩形打分、`decode S=4/4096` 的压缩 indexRead；
父子不重复计费、压缩 RoPE 双路径断言；前端全量测试、后端测试、
docs check、构建及 Chrome 桌面/移动端 12/12。
