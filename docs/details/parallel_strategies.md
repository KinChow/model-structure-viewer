# 成熟框架并行策略调研

## 目的和边界

本文记录本地 vLLM、SGLang 与 TensorRT-LLM 源码中的并行策略事实，作为 MSV
并行计划和权重归属协议的设计依据。本文不把运行时通信实现直接变成
MSV Graph 节点；MSV 只保留能够解释结构、权重归属和理论成本的稳定语义。

**协议结论已定稿**：本文是**证据层**，逻辑轴定义、约束等式、九项裁决和
口径边界见 [`parallel_protocol.md`](parallel_protocol.md)（协议唯一住址）。
本文与协议冲突时以协议为准；新增框架证据先落本文，再评估是否改协议。

调研对象：

- `/Users/zhouzijian01/Desktop/workspace/code/kinchow/vllm`
- `/Users/zhouzijian01/Desktop/workspace/code/kinchow/sglang`
- `/Users/zhouzijian01/Desktop/workspace/code/kinchow/TensorRT-LLM`

调研时间：2026-09-09。

## vLLM

### 并行轴和有效专家域

vLLM 的并行配置位于：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`

基础模型并行主要由 PP、TP 和 PCP 组成；DP 在特定启动模式下扩展整体并行规模。
EP、EPLB、DCP、sequence parallel 和 all-to-all backend 都有独立校验。

vLLM 的 MoE 实现并不把 MoE TP 和 MoE EP 始终作为两个独立的用户可见轴。
在启用 EP 后，MoE 配置会将 DP、PCP、TP 展平为有效专家域：

```text
flatten_tp_size = dp_size * pcp_size * tp_size
flatten_tp_rank =
  dp_rank * pcp_size * tp_size
  + pcp_rank * tp_size
  + tp_rank
```

随后，MoE 内部通常以 EP 为专家 ownership 轴，并将 MoE 内部 TP 设为 1。
因此不能把 vLLM 的 `ep_size` 直接解释成一个独立的 `moe_ep` 字段。

### Attention

相关实现：

- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/model_executor/layers/linear.py`

Q、K、V 的实际切分主要由模型 attention 实现和 parallel linear 层完成。
常见规则是：

- Q heads 通常按 TP 切分；
- KV heads 不少于 TP 时，要求可按 TP 整除并切分；
- KV heads 少于 TP 时，要求 TP 能被 KV head 数整除，然后复制 KV heads；
- `disable_tp` 可以让特定线性层不参与普通 TP。

这说明 attention 的“并行轴”不能只从一个全局 TP 数字推导，还需要记录
Q/K/V 的实际 head 数、复制策略和线性层方向。

### MoE 权重和专家 ownership

相关实现：

- `vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py`
- `vllm/vllm/model_executor/layers/fused_moe/routed_experts.py`
- `vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`

vLLM 区分以下概念：

- global expert id；
- local expert slot；
- expert map；
- backend 使用的 expert mask。

普通 backend 可以使用 `-1` 表示非本地专家，并把 global expert 映射到
local slot；AITER 等 backend 可能使用 mask，不能把 canonical expert map
直接作为 kernel 输入。

专家权重的常见切分方向：

- `w1/w3` 按 ColumnParallel 方向切分；
- `w2` 按 RowParallel 输入维度切分；
- 非本 rank 专家的权重通常不加载；
- checkpoint padding、量化布局和 local expert mapping 由加载路径共同处理。

### Shared expert 和通信

vLLM 的 shared expert 不应被默认视为 routed expert。实现中存在：

- 独立执行；
- kernel 内 overlap；
- 多 stream overlap；
- 受 backend、EPLB、DP、量化和输入条件限制的 overlap。

MoE token 流程通常是：

```text
dispatch -> local expert compute -> combine -> optional final reduction
```

最终 reduction 是否发生，以及属于哪个通信组，取决于 sequence parallel、
TP/EP、backend 是否已经完成 reduction 等条件。

## SGLang

### 显式并行轴

相关实现：

- `sglang/python/sglang/srt/arg_groups/fields/parallel.py`
- `sglang/python/sglang/srt/runtime_context.py`
- `sglang/python/sglang/srt/distributed/parallel_state.py`

SGLang 明确区分：

- global TP；
- PP；
- attention DP；
- attention CP；
- attention DCP；
- MoE EP；
- MoE DP；
- MoE TP。

其派生关系包括：

```text
attn_tp_size = tp_size / attn_dp_size / attn_cp_size
moe_tp_size  = tp_size / moe_ep_size / moe_dp_size
```

这使 attention 和 MoE 可以拥有不同的有效并行宽度。SGLang 的 group layout
也显式构造独立的 MoE TP、MoE EP 和 MoE DP group。

因此，若 MSV 需要表达混合 ETP，SGLang 的显式轴模型比 vLLM 的展平模型
更适合作为计划层参考；但仍需区分逻辑轴和物理通信 group。

### Attention

相关实现：

- `sglang/python/sglang/srt/layers/linear.py`
- `sglang/python/sglang/srt/layers/radix_attention.py`
- `sglang/python/sglang/srt/layers/dp_attention.py`

SGLang 允许 Q、K、V 使用不同的实际 TP 信息。KV heads 少于 TP 时可以复制
KV heads，checkpoint loading 也按实际 rank 和 size 处理。

Attention DP 还涉及 token layout：

- rank layout 通常可表示为 `(dp, cp, tp)`；
- 支持 `MAX_LEN` 和 `SUM_LEN` padding；
- 支持 all-reduce 和 all-gather 路径；
- attention CP 产生的 token layout 可能需要交接给 MoE DP。

因此 attention DP 不能只建模成“权重除以 DP”，还需要区分 token
replication、token gather 和参数 shard。

### MoE dispatch 和专家权重

相关实现：

- `sglang/python/sglang/srt/layers/moe/token_dispatcher/standard.py`
- `sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`
- `sglang/python/sglang/srt/models/deepseek_v2.py`

标准流程是：

```text
global expert id
  -> local expert mapping or backend-specific mapping
  -> dispatch
  -> local MoE core
  -> combine
```

不同 backend 可能选择：

- Python 侧 global-to-local mapping；
- kernel 直接解释 global expert id；
- expert mask；
- DeepEP、Mooncake、NIXL、PPLX 等不同通信实现。

所以 expert ownership 和 kernel 输入格式不是同一个概念。

### Shared expert

SGLang 的 DeepSeek 系列实现表明 shared expert 可能是：

- 独立执行；
- fused 到 expert slots；
- per-rank shared slot；
- TP1；
- 只在 home rank 执行；
- 与 routed expert 使用不同的 reduction 或 overlap 规则。

这直接支持 MSV 将 `sharedExpertsAreFused` 作为真实结构语义，而不是只作为
参数量计算开关。

## TensorRT-LLM

### 源码范围

本地存在完整 TensorRT-LLM 源码仓库。本文使用的关键实现和文档包括：

- `TensorRT-LLM/tensorrt_llm/mapping.py`
- `TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py`
- `TensorRT-LLM/tensorrt_llm/_torch/attention/`
- `TensorRT-LLM/tensorrt_llm/_torch/moe/fused_moe/`
- `TensorRT-LLM/docs/source/features/parallel-strategy.md`

此前“本地没有完整 TensorRT-LLM 源码”的判断是错误的，原因是检索时没有命中
实际仓库目录。以下 TensorRT-LLM 结论以该本地源码为准。

### 并行轴和约束

`mapping.py` 的 `MappingBase` 显式包含：

```text
tp_size
pp_size
cp_size
moe_tp_size
moe_ep_size
attn_tp_size
attn_cp_size
enable_attention_dp
```

常规 mapping 要求：

```text
world_size = tp_size × pp_size × cp_size
attn_tp_size × attn_cp_size = tp_size × cp_size
moe_tp_size × moe_ep_size × moe_cluster_size = moe_world_size
```

当 `moe_tp_size` 和 `moe_ep_size` 都没有显式指定时，TensorRT-LLM 会从
`moe_world_size` 派生默认值；显式指定时会校验 MoE TP/EP 与可用 MoE 域闭合。
`enable_attention_dp`、`enable_lm_head_tp_in_adp` 和 DWDP 会改变默认路径，
不能只依赖全局 `tp_size` 推导每个组件的实际分片。

### Attention

TensorRT-LLM 的并行策略文档明确说明：

- TP 下 attention 前后的 GEMM 权重和 attention heads 通常分片；
- DP 下 attention GEMM 权重复制，KV cache 按 DP rank 分区；
- GQA、MQA、MLA 在 KV heads 少于 TP 时可能复制 KV cache；
- 特定模型可以关闭某个 GEMM 的 TP；
- CP 分配长序列或 context，影响 activation 和 KV cache 布局。

因此 attention 需要分别记录：

```text
Q/K/V projection 的权重分片
Q/K/V head 数和复制
KV cache 分片
attention DP/CP 的 token 布局
```

### MoE 和 Hybrid ETP

TensorRT-LLM 文档将 MoE 分为：

```text
TP:
  每个 expert 的权重矩阵在所有 GPU 上切分，每个 GPU 处理全部 token

EP:
  每个 expert 的完整权重位于单个 GPU，每个 GPU 处理本地 expert token

Hybrid ETP:
  先按 EP 分配 expert，再按 TP 切分本地 expert 权重
```

文档给出的约束是：

```text
moe_tensor_parallel_size × moe_expert_parallel_size
  = tensor_parallel_size
```

这与 MSV 的 `moe_tp/moe_ep` 计划轴直接相关，但不能把该等式推广到所有
框架或所有通信 backend。TensorRT-LLM 还存在 Wide-EP、expert replication、
DWDP 和不同的 MoE communication implementation。

`mapping.py` 还明确展示了：

- EP/TP 可以独立指定；
- expert ownership 与 local expert layout 有单独配置；
- DWDP 会将 fused-MoE 视角的 `moe_tp_size`、`moe_ep_size` 置为 1，
  再通过额外的 expert layout 表达分区；
- CP 类型会影响 MoE TP/EP 是否可用。

### 其他会影响成本的组件

TensorRT-LLM 的并行策略不只影响 attention 和 MLP/MoE，还会影响或约束：

- token embedding；
- `lm_head` 和 vocab parallel；
- router/gate；
- shared expert；
- norm 和带参数的特殊连接模块；
- KV cache；
- residual/activation boundary；
- PP stage 的 embedding、final norm 和 `lm_head` 归属；
- cross-attention、vision/audio encoder 和 projector；
- sampling 前的 logits gather；
- communication buffer、quantization workspace 和 CUDA graph padding。

其中 embedding、norm、residual、RoPE 不一定需要切分，但必须判断它们是
replicated、vocab-parallel、无参数，还是会形成通信/布局边界。

## MSV 终态：共享事实，框架视图

### 不做三套独立成本系统

MSV 不定义三套互相独立的：

```text
vLLM cost calculator
SGLang cost calculator
TensorRT-LLM cost calculator
```

也不使用一套隐藏框架差异的“万能 runtime 公式”。

终态是：

```text
一套模型事实源
  + 一套逻辑并行和权重分片协议
  + 多个 framework execution profile
```

### 三层模型

#### 1. Model facts

由前端 Graph、operator、formula、checkpoint truth 和 `weightMatrices` 提供：

```text
节点和拓扑
operator identity
tensor shape
权重矩阵数量
dtype / quantization
expert count / top-k
KV cache 基础形状
```

这一层不因 vLLM、SGLang、TensorRT-LLM 而复制。

#### 2. Logical parallel and weight shard

由 MSV 自己定义：

```text
physical topology:
  world size / node count / devices per node

logical plan:
  pp / tp / dp / cp
  attention tp / dp / cp
  moe tp / ep / dp

weight shard:
  tp / ep / vocab / replicated
  split dimension
  divisor
  replication
  padding
```

`weightMatrices` 是权重归属的唯一入口。分片和量化消费者不得再从路径、
operator 名称或 shape 规则推断权重归属；声明缺失时返回 `unknown` 并诊断。

#### 3. Framework execution profile

框架 profile 只描述某个 runtime 如何落实逻辑计划：

```text
effective attention width
effective MoE width
expert ownership / placement
dispatch / combine
reduction group
KV cache partition
lm_head handling
backend-specific padding and workspace
```

profile 不重新定义模型 Graph，也不复制 formula registry。

### 三类成本输出

#### Framework-neutral

可以跨框架共享：

```text
参数量
矩阵 shape
理论 FLOPs / MACs
checkpoint weight bytes
基础 activation shape
由 weightMatrices 推导的逻辑 shard bytes
理论 token communication bytes
```

#### Framework-conditioned

必须选择框架 profile：

```text
effective TP/EP/DP
expert placement
dispatch/combine 方式
final reduction
KV cache partition
fused shared expert
quantized weight layout
communication buffer
```

#### Runtime-unknown

没有固定框架版本、硬件拓扑、backend 或真实 token 分布时，不输出精确值：

```text
实际通信时间
overlap 后的有效耗时
真实峰值显存
kernel workspace
expert imbalance 尾延迟
端到端吞吐和 latency
```

这些结果应标记为 `unknown` 或 `requires runtime benchmark`。

### 框架映射

| 框架 | MSV 主要参考内容 | 不直接复制的内容 |
|---|---|---|
| vLLM | effective expert domain、global/local expert、expert placement、权重加载、w1/w3 与 w2 方向 | vLLM 特定的 DP/PCP/TP 展平和 backend mask |
| SGLang | attention/MoE 独立逻辑轴、MoE TP/EP/DP、token layout 和 dispatch/combine | SGLang 特定的 group layout、padding mode 和 backend |
| TensorRT-LLM | Hybrid ETP、attention DP、CP、Wide-EP、fused/quantized layout、workspace 约束 | engine、kernel、硬件和通信 backend 的具体实现 |

最终原则：

```text
SGLang 提供逻辑轴参考
vLLM 提供 expert ownership 参考
TensorRT-LLM 提供 ETP 和 engine 条件参考
MSV 提供统一的模型事实和理论成本
```

## 对 MSV 的结论

### 保留的稳定语义

MSV 计划和 `weightMatrices` 应明确分层：

```text
配置声明
  -> 派生 attention / MoE 并行宽度
  -> 权重归属和矩阵切分
  -> 理论通信量和显存
```

`weightMatrices` 至少需要表达：

- attention Q/K/V 的实际 head 分片或复制；
- dense/shared GEMM 的 TP 亲和；
- routed expert 的 EP 亲和；
- `w1/w3` 的列切分；
- `w2` 的行切分；
- vocab、replicated 和特殊 shared expert 语义。

MoE 节点还应区分：

- global expert count；
- 每卡 local expert count；
- local slot 或 ownership；
- 是否使用 expert mask；
- shared expert 是 replicated、fused、per-rank 还是 home-rank。

### 不应直接进入 Graph 的内容

以下内容属于具体运行时或 kernel contract，不应作为 MSV Graph 的通用字段：

- NCCL 或其他通信库的具体 group 对象；
- DeepEP、AITER、FlashInfer 等 backend 类名；
- CUDA stream 和 graph capture 细节；
- 某个 backend 的 global id 与 mask 编码；
- overlap 的具体 kernel 调度实现。

这些内容应进入后端 evidence、调研文档或可选的 runtime metadata。

### M12 的设计约束

在修改 `validatePlan` 和通信估算前，必须先确定：

1. `moe_ep` 和 `moe_tp` 是逻辑分片轴，还是物理 rank 数；
2. `moe_tp * moe_ep` 与 global TP、DP 的关系；
3. attention DP、MoE DP 和专家权重分片是否共享 rank；
4. 无 EP 时 DP 是否切分专家权重；
5. shared expert 的 TP、EP、复制和 reduction 语义；
6. AllToAll 的输入布局、dispatch 后布局和 combine 后布局；
7. inter-node 带宽、通信时间和 overlap 的估算口径。

在这些问题确认前，不能仅凭 `moe_ep * moe_tp` 修补校验逻辑。

## 参考源码索引

| 框架 | 主题 | 源码 |
|---|---|---|
| vLLM | 并行配置与约束 | `vllm/vllm/config/parallel.py` |
| vLLM | 通信 group | `vllm/vllm/distributed/parallel_state.py` |
| vLLM | MoE 有效专家域 | `vllm/vllm/model_executor/layers/fused_moe/config.py` |
| vLLM | expert ownership | `vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py` |
| vLLM | expert 权重加载 | `vllm/vllm/model_executor/layers/fused_moe/routed_experts.py` |
| vLLM | shared expert | `vllm/vllm/model_executor/layers/fused_moe/runner/shared_experts.py` |
| SGLang | 并行参数与派生轴 | `sglang/python/sglang/srt/arg_groups/fields/parallel.py` |
| SGLang | 并行 group | `sglang/python/sglang/srt/distributed/parallel_state.py` |
| SGLang | 并行宽度派生 | `sglang/python/sglang/srt/runtime_context.py` |
| SGLang | attention DP | `sglang/python/sglang/srt/layers/dp_attention.py` |
| SGLang | attention linear | `sglang/python/sglang/srt/layers/linear.py` |
| SGLang | MoE dispatch | `sglang/python/sglang/srt/layers/moe/token_dispatcher/standard.py` |
| SGLang | Deep EP MoE | `sglang/python/sglang/srt/layers/moe/ep_moe/layer.py` |
| SGLang | shared expert | `sglang/python/sglang/srt/models/deepseek_v2.py` |
| TensorRT-LLM | 并行轴和约束 | `TensorRT-LLM/tensorrt_llm/mapping.py` |
| TensorRT-LLM | 并行策略说明 | `TensorRT-LLM/docs/source/features/parallel-strategy.md` |
| TensorRT-LLM | Graph sharding | `TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py` |
| TensorRT-LLM | MoE runtime | `TensorRT-LLM/tensorrt_llm/_torch/moe/fused_moe/` |
