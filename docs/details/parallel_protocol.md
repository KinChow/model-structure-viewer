# 并行与权重分片协议（定稿）

> 2026-09-22（Asia/Shanghai）增量：[framework accounting](framework_accounting.md) 实装 neutral /
> vLLM / SGLang 的公式适配边界。下文“本轮不实现 framework execution profile”
> 是 2026-09-10 的历史范围；现已实现 plan、cache ownership、state dtype、fusion
> 和统一驻留账本，但仍不实现 runtime/backend 仿真、自动方案搜索或实测校准系数。

2026-09-10 定稿。本文是 MSV **逻辑并行与权重归属协议的唯一住址**：
`validatePlan`、`sharding`、逐卡投影、容量分桶、通信估算和 UI 输入契约都以本文为准。

调研依据见 [`parallel_strategies.md`](parallel_strategies.md)（vLLM / SGLang /
TensorRT-LLM 本地源码事实）；权重声明体裁见
[`sharding_matrix.md`](sharding_matrix.md)（N2-4 三层设计）。
本文只定**逻辑语义**，不定运行时实现：NCCL group 对象、backend 类名、
kernel 调度、mask 编码都不进 MSV Graph。

## 一、协议分层与字段归属

```text
physical topology   —— 用户输入的硬件事实
  nodes / gpus_per_node / chip / world_size（派生）

logical parallel plan —— 用户输入的逻辑轴（本文 §二 定义）
  tp / pp / dp / ep / moe_tp / moe_ep / attnMode / vocabParallel

weight shard plan   —— 由 weightMatrices 声明 + plan 推导，非用户输入
  class(tp|ep|vocab|replicated) / divisor / setDegree / 不均衡区间

communication plan  —— 由结构 + plan 推导，非用户输入
  all-reduce / AllToAll(dispatch,combine) / PP p2p / PD transfer
```

分层纪律：
- 上层不得反向决定下层输入（例如通信估算不得改写 plan）；
- `weightMatrices` 是权重归属唯一入口，分片与量化消费者不得从路径、
  operator 名或 shape 反推归属（终态见 §三 Q 记录与 sharding_matrix §三）；
- framework execution profile（vLLM/SGLang/TRT-LLM 各自如何落实逻辑计划）
  是**独立第三层概念，本轮不实现**，只在本文登记边界。

### 默认部署策略（人因约定）

UI 的默认部署不是自动并行方案搜索，而是固定的单机卡数档位推荐：

```text
physical topology: 1 node × 8 GPUs
logical TP candidates: 1 → 2 → 4 → 8
selection: smallest tier whose theoretical per-card projection fits
fallback: TP8; if it still does not fit, keep no-fit and ask for manual expansion
```

因此小模型默认使用 TP1/TP2/TP4，但仍按单机 8 卡拓扑展示；较大模型优先落在
单机 TP8。推荐只消费现有 framework accounting、Graph IR 和 Fit 投影，不引入
实测常数、经验系数或 runtime simulator。用户一旦修改 TP/PP/EP/DP、节点数或
GPU/节点，状态转为手动配置；切换硬件或后台 checkpoint 真值更新不会覆盖手动值，
可以通过“恢复默认部署”重新采用该推荐。

- 基准负载固定为 batch=1、2048 tokens；使用模型权重精度、KV 显式 dtype 和所选
  framework profile。修改工作负载或 what-if 时不会偷偷扩卡，实际 Fit 继续按当前负载计算。
- 拓扑容量与策略用卡数分开展示，避免把“TP1 使用 1 卡”误读为使用了整机 8 卡。
- 集中式默认一台机器；用户主动选择 PD 后，Prefill/Decode **各一台独立机器**，
  不是将两份模型同时塞进同一台机器。两侧手动策略独立保存。
- 未知显存信息返回 unknown；超过单机上限保留 no-fit，不注入系数，不自动扩多机。
  本功能不判断 backend/量化 kernel 支持、TP shape 约束、runtime workspace 或吞吐最优性。
- 设计参考 vLLM 官方 `docs/serving/parallelism_scaling.md`（联网核查 revision
  `382970ee6ca490aeaaaf4e32c53695b581ff61ba`）：能装单卡时无需分布式，单机内采用 TP，
  超单机后再考虑 TP+PP。其无 NVLink/不均匀切分建议、MoE 专用 EP 优化仍由用户
  明确配置，本次不扩展成硬件性能优化器。

```text
https://github.com/vllm-project/vllm/blob/382970ee6ca490aeaaaf4e32c53695b581ff61ba/docs/serving/parallelism_scaling.md
```

## 二、逻辑轴定义与约束

| 轴 | 语义 | 缺省 |
|---|---|---|
| `tp` | 全局张量并行宽度（attention/dense GEMM 切分） | 1 |
| `pp` | 流水并行 stage 数 | 1 |
| `dp` | 数据并行副本数（attention 侧；MoE 侧见 Q4） | 1 |
| `ep` | 专家并行宽度（专家 ownership 轴） | 1 |
| `moe_tp` | 本地专家权重再切分宽度（混合 ETP） | undefined（不伪造） |
| `moe_ep` | 专家 ownership 宽度（与 `ep` 的关系见 Q2） | undefined（不伪造） |
| `attnMode` | `tp`（切 KV）\| `dp`（复制权重、按 rank 分区 KV） | `tp` |
| `vocabParallel` | embed/lm_head 是否按 vocab 切 | true |
| `world_size` | 校验用；显式给出时必须 = tp×pp×dp | 派生 |

约束等式（判据，`validatePlan` 执法）：

```text
world_size      == tp × pp × dp
ep ≤ experts    且  moe_ep ≤ experts
experts % moe_ep == 0                        （整除，否则不均衡区间无意义）
EP 启用 + attnMode=dp + 未显式 moe_ep:
                   ep == tp × dp             （vLLM EP_SIZE = TP_SIZE × DP_SIZE）
EP 启用:           moe_ep × moe_tp == ep × tp（专家域闭合，TRT-LLM Hybrid ETP）
无 EP:             moe_tp == tp
```

## 二点五、量化权重建模（2026-09-10 用户裁决补充）

量化方案只作用于 **Linear 权重矩阵**：模型权重配置遵循 HF
`quantization_config` 规范（quant_method / targets / ignore /
modules_to_not_convert），vLLM/SGLang 的 quant method 按 Linear 模块应用。
MSV 侧的三条对应：

1. **声明侧**：`weightMatrices` 组带 `quantizable` 标记（P4-1）与 `shape`
   数组、`split` 切分轴（schema v2）；
2. **字节侧（已实现）**：`quantLinearWeightBytes` 按方案精确计 **打包权重 +
   scale + zeros**（fp8 块量化 scale 数 = ceil(out/B)·ceil(in/B)——形状敏感，
   所以向量/卷积核类参数必须 quantizable=false 而非靠维度猜）；
3. **计算侧（登记，P10 实现）**：反量化走**融合算子**——vLLM 的两条实证：
   w8a8 block kernel 在 kernel 内 dequant（`quantization/fp8.py:443` "use
   BF16 dequant when direct FP8 is not supported"，per-tensor/channel 路径
   `fp8.py:456` "dequant to BF16 and run GEMM"）；mxfp4 为 W4A16，kernel 内
   反量化（`mxfp4.py:819` "the fallback dequantizes only the weights"）。
   建模口径：量化 GEMM 的额外成本 = scale 读（已含在 weights 字节）+ 逐元素
   convert 计算（融合进 GEMM 的 vector 动作），按 W4A16/W8A8 分方案定乘子。
   不新增独立反量化算子节点——与 kernel 现实一致（融合），也避免结构树膨胀。



每项格式：结论 → 依据 → 边界。

**Q1 `moe_ep`/`moe_tp` 是逻辑分片轴，不是物理 rank 数。**
依据：vLLM 把 dp/pcp/tp 展平成有效专家域（`config/parallel.py`），SGLang 用显式
独立轴（`arg_groups/fields/parallel.py`），TRT-LLM 的 `MappingBase` 逻辑字段与
物理 group 分离。三家的共同不变量是"每卡持多少专家、专家内切几份"，差异全在
rank 映射。边界：物理 rank 映射（哪个 rank 持哪些专家、local slot 编号、
expert map/mask）归 framework profile，本轮不实现。

**Q2 专家域闭合：`moe_ep × moe_tp == ep × tp`。**
依据：TRT-LLM 文档 `moe_tensor_parallel_size × moe_expert_parallel_size =
tensor_parallel_size`（无 attention DP 时）；vLLM 的 EP 域 = tp×dp（W-B 已实现，
`cost/parallel.js`）；SGLang `moe_tp = tp/(moe_ep × moe_dp)`。取三家交集写成
上式，并保留 `ep == tp × dp` 的 DP-attention 特化校验。边界：不把该等式推广到
所有 backend（Wide-EP、DWDP、expert replication 会改写它）——这些进 profile。

**Q3 `moe_dp` 不进 plan schema。**
依据：只有 SGLang 单家证据（`runtime_context.py` 的派生式），vLLM 无独立
moe_dp（展平进专家域），TRT-LLM 用 DWDP 另一套表达。与"缺省不伪造数值"纪律
（N2-4 W-B）冲突。边界：登记为 framework profile 概念；步骤 9 的 AllToAll
`dp > 1` 用现有 `dp` 轴近似，并在输出标注该近似。attention dp 与专家权重分片
**共享 rank**（vLLM 语义，现状不变）。

**Q4 无 EP 时 DP 也切专家权重。**
依据：AMD vLLM playbook + TRT-LLM 并行策略文档——"DP = 复制"只对 attention
成立；专家权重在 DP 下按专家 TP 切分语义分摊。现状 `cost/sharding.js`
`expertShardDivisor` 未启用 EP 分支 `divisor = (moe_tp ?? tp) × dp` 即此结论，
确认不改。边界：ZeRO/FSDP ÷dp 仍为范围外（推理工具）。

**Q5 shared expert 唯一语义 = ÷tp。**
依据：vLLM shared expert 独立于 routed（`fused_moe/runner/shared_experts.py`），
SGLang DeepSeek 系有独立/fused/per-rank/home-rank 多形态但权重亲和均为 TP。
边界："EP+TP+DP 全开 + 特定 all2all backend 时复制"仅 AMD playbook 单源，
**登记不做**，属 framework profile。

**Q6 AllToAll 三段布局（通信量公式的语义基础）。**

```text
dispatch 前：token 按 attention 域分布（attnMode=dp 时按 DP rank 分片）
dispatch 后：token 按 local expert ownership 聚合
combine 后 ：回到 token 布局（可能触发 final reduction）
```

依据：vLLM `prepare_finalize/naive_dp_ep.py` 与 SGLang
`token_dispatcher/standard.py` 的公共骨架。理论通信量：dispatch ≈
`T × topk × hidden × bytes`，combine 同量级；final reduction 是否发生取决于
sequence parallel / backend 是否已 reduce —— 属 runtime-unknown，不计精确值。

**Q7 时间类估算口径。**
① `interNode` 为显式开关，默认 false（保守取 intra-node 费率）；开启才切
`chips/rates.js` 的 `inter_node` 行，缺该规格时整段回退并标注 missing。
② PD 传输时间 = `bytes / min(两侧链路带宽)`。
③ overlap 只做**静态上限**：comm 与 compute 取 max 而非 sum，输出必须标注
"上限估计，非调度仿真"。这条同时是与"明确不做 overlap 仿真"的分界线：
取 max 是闭式不等式，不涉及 kernel 调度或 stream 建模。
   **实现现状（P10）**：roofline 五路聚合本就是 max 语义（bound = 最大单路），
   输出带 `overlapUpperBound: true` 标记；PD 传输时间 = bytes/min(两侧带宽)；
   AllToAll 的 dp>1 触发（无 EP 时 DP-shards-experts，Q4/Q6）；KV keep-ratio
   进 parallel plan schema（0<r≤1，streaming/滑窗 fit 估算口径）。

**Q8 并行 plan 独立 schema。**
现状：并行 plan 字段只存在于 `cost/parallel.js` 的 `validatePlan` 归一化代码，
与组网逐层调度（`structure/layers/schedule.js`）共用过 "plan" 一词。结论：新建
`cost/parallelPlan.js` 作为归一化与校验单一住址，`validatePlan` 委托；文档中
固定两个名字 —— **parallel plan**（并行）vs **build plan**（组网）。

**Q9 UI 是第四消费者。**
`moe_tp` / `moe_ep` / `vocabParallel` / `world_size` 协议层已支持且有校验，
但 UI 零入口 —— N2-4 W-B 的组合语义在用户路径上无法触发，等于
`sharding_matrix.md` §四"一处不落即死功能"的第四处。结论：`CostSummary`
PlanFields 补 moe_tp/moe_ep 输入与 vocabParallel 开关，world_size 只读派生，
并按 `MAINTENANCE.md` 纪律 7 补消费侧接缝测试。

## 四、三类成本输出（口径边界）

```text
framework-neutral   参数量 / shape / 理论 FLOPs·MACs / checkpoint bytes /
                    weightMatrices 推导的逻辑 shard bytes / 理论通信字节

framework-conditioned expert ownership / dispatch·combine 方式 / final reduction /
                    KV partition / fused shared expert / 量化布局 / comm buffer

runtime-unknown     实际通信时间 / overlap 后耗时 / 真实峰值显存 / workspace /
                    expert imbalance 尾延迟 / 吞吐与 latency
```

runtime-unknown 项一律输出 `unknown` 或 `requires runtime benchmark`，
不得用默认值伪装精确（诚实性纪律，M11 教训）。

## 五、本文不覆盖（显式登记）

- framework execution profile 的实现（vLLM/SGLang/TRT-LLM 视图）；
- `moe_dp`、EP 全开时 shared expert 复制、Wide-EP / DWDP / expert replication；
- ZeRO/FSDP ÷dp、activations 的 CP/SP 响应、EP 负载不均衡动态建模；
- plan 搜索与自动推荐（五支柱越界，指向 Vidur）。
