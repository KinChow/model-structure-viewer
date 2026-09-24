# MTP expert aggregate binding repair

- 日期：2026-09-24
- Graph IR：保持 v2

## 发现

GLM-5.3-Flash 和 GLM-5.3-Flash-BF16 的 MTP 尾层 manifest 发布了：

```text
model.language_model.layers.45.mlp.experts.<expert>.(gate_proj|up_proj|down_proj)
```

当前 Graph IR 的 MoE operator 使用一个 fused `expert_mlp` 叶节点，
而不是为每个专家生成独立的可执行节点。这是有意的执行抽象：成本和
驻留公式按 `experts × 3` 矩阵计算，不应把每个专家复制成新的 active
算子。但是此前 truth binder 只支持精确 canonical module path，导致
这些真实 checkpoint tensor 全部落入 gap，无法证明 MTP 专家被绑定。

## 修复

- 给 MTP fused expert leaf 增加 `truth_path_prefix`；
- truth binder 支持把指定 prefix 下的多个 checkpoint module 聚合到一个
  Graph IR 权重叶；
- 保留每个真实 tensor name、shape、dtype 和总参数元素；
- 不改变 Graph IR 的执行节点数量，不重复计费；
- router、shared expert、MTP norm 和 attention 投影仍按各自精确路径绑定；
- GLM-5.3-Flash 的普通 tail MTP 规则仍保持：
  - 无本地 `embed_tokens`；
  - 无独立 `shared_head.head`；
  - 不继承主干 mHC；
  - 输出头复用主 `lm_head`。

## 验证

新增机制测试用小型 fixture 验证：

- 三个 MTP expert tensor 聚合到唯一 `mtp.layer.mlp.expert_mlp`；
- 聚合节点保留全部 tensor names；
- 聚合参数元素总量闭合；
- 普通 DeepSeek/GLM MTP 的精确路径绑定不回归；
- truth graph 单元测试继续通过。

真实 GLM-5.3-Flash index manifest 已确认尾层包含 288 个 routed experts；
本修复只改变 truth 归属，不把 expert 逐个展开成新的展示层，也不声称
GPU fused kernel 的实际物化方式已经测得。
