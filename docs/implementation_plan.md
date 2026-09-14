# 待实现计划

本文只记录当前仍待实现或需要继续维护的事项。已经完成的能力以代码、测试和 [`CHANGELOG.md`](../CHANGELOG.md) 为准；历史版本通过 Git 提交记录追溯。

## 当前基线

以下能力已经落地，不再作为待实现任务：

- 前端静态结构生成、内置模型 catalog、HF/ModelScope 配置读取和 safetensors header 真值接入。
- 模型 registry、通用和专用 builder、layers、ops、公式、IR、materializer 和诊断链路。
- 结构图、Layers、Inspector、JSON/Mermaid/DOT 导出、芯片 Cost Lens、并行投影和 PD 分析。
- Python API/CLI、local cache、transformers meta-device 验证和 GitHub Pages 构建流程。
- Graph IR v2 节点事实、Graph-first 消费和已有 parity 基线测试；legacy
  `root` projection 仍在退役过程中，不作为终态能力。
- Graph-first layout、compute/aggregate、通信、PP/PD projection、导出和 canonical node identity。
- 未知架构统一 `unsupported`：alias 精确表未命中的架构不再组网（generic-decoder
  兜底已删），空网络走完管线并枚举支持项。

## P0：验证流程收口（已完成）

Playwright 已替换旧版 SVG/CDP 验收路径。`npm --prefix frontend run test:e2e`
使用隔离 Vite 服务和桌面/移动 Chrome，覆盖入口来源、React Flow 节点与显式边、成本交互、窄屏布局和多模态视觉节点；`verify:page` 保留为兼容别名。
全量内置模型重用例在双 project 并行时可能资源竞争超时（2026-09-10 实测：并行 fail、单跑通过）——失败先单跑复核再定性。

## 当前执行路线（2026-09-09）

本节是当前有效计划，优先级高于历史波次中的未更新描述。MSV 的终态是：

```text
前端 Graph = 唯一产品结构事实源
后端 Python = Transformers evidence 和结构对账
weightMatrices = 唯一权重归属/分片协议
framework profile = vLLM/SGLang/TensorRT-LLM 的执行映射
```

执行顺序：

1. ~~**协议定稿**：确定 physical topology、logical parallel plan、
   weight shard plan、communication plan 的边界；调研依据见
   [`details/parallel_strategies.md`](details/parallel_strategies.md)。~~
   ✅ 已完成（2026-09-10）：协议唯一住址
   [`details/parallel_protocol.md`](details/parallel_protocol.md)。
2. ~~**结构正确性收口**：未知架构统一 `unsupported`，删除
   `generic-decoder` 的未知架构兜底。~~ ✅ 已完成（2026-09-10）。
3. ~~**权重协议收口**：补齐所有带权重叶的 `weightMatrices`，明确
   `tp`、`ep`、`vocab`、`replicated` 和 shared expert 语义，删除
   分片、量化和容量计算 fallback。~~ ✅ 已完成（2026-09-10，P2-P5：
   18399/18399 带权叶全声明，WEIGHT_PROJECTION_RULES 与 QUANTIZABLE_OPS
   回退删除，无声明带权叶 = unknown）。
4. ~~**并行计划实现**：按成熟框架证据统一 attention/MoE 的逻辑轴和物理
   rank 约束，完善 `moe_tp`、`moe_ep`、`moe_dp`、DP/EP/ETP 校验。~~
   ✅ 已完成（2026-09-10，P6：parallelPlan.js 单源 + 协议 Q2/Q4 校验执法 +
   UI 第四消费者；moe_dp 按 Q3 登记不做）。
5. ~~**fused shared expert 闭合**：贯通 recipe、builder、operator、
   `weightMatrices`、sharding、derived weights、communication 和测试。~~
   ✅ 已完成（2026-09-10 P3；2026-09-11 对照 transformers/vLLM/SGLang 复核）：
   checkpoint 融合 = 单个更宽 MLP（`intermediate = moeI × n_shared`），K3/DeepSeek/GLM
   同构。SGLang `num_fused_shared_experts` 是运行时把 shared 打进 routed GEMM，
   K3 不用这条路径。msv 只建模 checkpoint 布局；`sharedExpertsAreFused` 仅 K3
   需要（目录里唯一 `n_shared>1`）。不改 K3 配方。
6. ~~**后端对账**：后端输出带路径、class 和参数信息的 Transformers
   evidence，返回 `only_transformers`、`only_msv`、class/path/shape
   mismatch；区分构造通过和结构一致。~~ ✅ 已完成（2026-09-10，P7）。
9. ~~**通信成本扩展**：在上述协议稳定后，再实现 AllToAll `dp > 1`、
   inter-node/PD 时间、KV keep-ratio、overlap 和 per-stage roofline。~~
   ✅ 已完成（2026-09-10，P10；overlap 为静态上限口径 Q7③）。
7. ~~**Graph/root 收口**：迁移所有 root 消费者，删除 `root`、
   graph-to-tree projection 及仅为兼容层保留的代码。~~ ✅ 已完成
   （2026-09-10，P8：grep 清零 + fold 312 case 差分勘验）。
8. ~~**版本和文档治理**：统一版本字段，从 registry 自动生成公式、
   architecture、operator 和模型台账，扩大 `docs:check`。~~ ✅ 已完成
   （2026-09-10，P9：models/architectures 台账生成器并入 docs:check；
   版本边界文档化为 IR version:3 / graph schema_version:2 两轨）。

**收尾登记（下一步计划池；大件已迁「后续立项池」——refactor_plan.md，带触发判据）**：
- `.root` 清零 grep 纳入 check_principles 棘轮（P8 建议，防"root 复活"）；
- 折叠语义前后端双源（前端模板产 repeat vs 后端 fold.py）单源化；
- details/models.md 的 Qwen3.5 分组漂移按 models_reference 机器段回改（P9 登记）；
- verify 对账 diff 的 triage（linear_attn 命名/融合投影/类名后缀三类系统性
  噪声），fixture 优先再谈规则（P7 登记）；
- per-stage roofline 的计算路（stage 级 actions）与 evidence 的 I/O shape
  （P10/P7 登记的诚实缺项）。
8. **版本和文档治理**：统一版本字段，从 registry 自动生成公式、
   architecture、operator 和模型台账，扩大 `docs:check`。
9. **通信成本扩展**：在上述协议稳定后，再实现 AllToAll `dp > 1`、
   inter-node/PD 时间、KV keep-ratio、overlap 和 per-stage roofline。

成本输出必须区分：

```text
framework-neutral
  参数量、shape、理论 FLOPs、checkpoint bytes、逻辑分片 bytes

framework-conditioned
  expert ownership、dispatch/combine、KV partition、workspace

runtime-unknown
  实际通信时间、overlap 后耗时、真实峰值显存、吞吐和 latency
```

## P1：原则收口重构

[`principles.md`](principles.md) 已定为强制约束，当前代码存在多处存量偏离（登记在该文 §10）。
收口路线见 [`refactor_plan.md`](refactor_plan.md)：自底向上分
W0 / W0.5 / W1 / W2 / W3a / W3b / W4 / W4.5 / W5 / W6 + 四条可并行旁路，
每波含范围、入口、依赖、验收命令、不包含项与回退方式，以差分测试验收。
核心分界：config 归一层只做字段归一（§4.7），方案决定权归组网（W3b）；
组件配方为一级概念，家族名只在薄解析层（§4.3）。

未排期项（设计已定、实现未跟上；开工前需确认，不属于任一历史波次）：

- **§5 source_ref**：采集 / 绑定 / Inspector 已接通。catalog 59 个模型除
  Kimi-K3 外均已 `msv dump-source-ref --source builtin` 入库（58/59）。
  Kimi-K3 不做 dump / verify：Hub modeling 在 import 时 `from fla.modules` /
  `fla.ops.kda`，本机 fla-core 0.5.2 仍拉 Triton；transformers 无 `kimi_k3`
  入库实现。结构模板继续用 catalog modeling 作二等证据。
  绑定脚本 `scripts/bind-source-ref.mjs`：58 模型 bind 通过。
  `verify --graph` 构造通过才算脚本失败；`structurally_consistent` 只报告。
  Decoder 路径段跟 HF `_modules` key：默认注意力 `self_attn`、FFN `mlp`；
  M2 全程 `block_sparse_moe`；K3 MoE 层 `block_sparse_moe`。类名偏离写 HF 全名。
  FlopCounterMode 独立算子夹具已接（`verification/flop_counter.py`：Linear /
  BMM / 深度可分 Conv1d）；整模型 forward 抽查未接（catalog 无权重）。
- **§6.4 来源解析契约**：前后端保持两套 resolver，以支持静态前端和 Python
  服务；需要共享来源类型、revision、fallback、错误分类和 fixture 契约，
  不强行合并运行时代码。
- **§7 国产芯片条目**：每字段必须有公开来源，缺项保持 unknown。
- **§3.8 删 llm-analysis 容量旁路**：cost 只 walk 图。叶上声明 KV/KDA
  数值容量与 hash buffer；`memoryBreakdown` / Params / PP 切层改 walk；
  生产不再调用 config 闭式算容量；身份测试期望侧 walk 图。闭式已删。

### 统一结构协议的生成或契约测试

当前前后端 schema 与前端 materializer 仍由两边维护。前端 Graph 是产品
事实源，后端只输出 Transformers evidence；需要增加跨端契约样例和真实
Graph/evidence 对账测试。schema-first 生成仅用于协议字段，不能让后端
重新成为第二个产品结构事实源。

`schemas.py` 已抽 `StructureNodeBase`（Pydantic 继承）；树还要 `children`，图还要 `parent_id` / `order` / `canonical_id`。JS `materializeStructureGraph.js` 仍是图投影，不与 Python schema 生成绑定。

### 模型 catalog 维护自动化

继续保持模型清单、配置、发布时间来源和验证报告可追溯。新增模型时应同时更新 catalog、来源记录、模型专项说明和验证结果，避免只增加配置却没有来源和验收记录。

## 后续方案（2026-09-12）

config 闭式（`derivedWeights.js`）已删。无 header 时身份测试走锚 1 + T4 walk 图声明；
台账参数量级走 `graphWeightCapacity`，与 UI 同口径。

**S2 extractor 收成查表（对标 flop_registry）** ✅
- `countsForNode` 无 `switch` / `ctxBuilders`。
- `FORMULAS[operator_id].fromNode` 抽 ctx，`.counts(ctx)` 计价。
- 未注册 op → `unknownComputePaths`。
- 护栏 §3.1b：switch case = 0，每条有 `fromNode`。

**组网调度住址** ✅ `config/plan.js` 已删，helper 在 `layers/schedule.js`。

**共享 layer 家族分派** ✅ 改读配方旗标 / config 字段。
**1.1 fromNode 只抽 ctx** ✅ `countsForNode` = `fromNode(env)` → `FORMULAS[id].counts(ctx)`；动作向量只在 `counts.js`。

剩余：

**S3 header `parameterTotal` 入库** ✅ catalog 58/59 有 `header-truth.json`
（Kimi-K3 跳过）。图声明逻辑元素对 header 逻辑元素（容差 2%）。
量化行按 `parameterCount` dtype 解包（GPTQ I32×8 扣 qzeros、NVFP4 I8×2、跳过 scale 桶）。
身份图侧：sidecar `mtp_tensor_count`（扫 header 张量名 `mtp.{i}` / 越界 `layers.{n}`，
对标 vLLM load_weights）>0 才计入投机头；config 空声明不是实际。
登记残差：无。V4 `wo_a` 按 vLLM `ColumnParallelLinear(n_heads*head_dim/o_groups, o_groups*o_lora)` 声明，不再把 grouped 输出维乘进权重。Flash-Next ngram 表已按 Embedding 声明。
T4 DSV4 打分项按 `compress_ratio` 分层（与 `dsv4VisibleKeys` 共用），不再按稠密三角登记。
整模型 FlopCounterMode 仍需要真实 `forward` + 权重，catalog 做不到。

**normalize 瘦视图 / schema / resolver** ✅
- 删 `indexer*` 兼容别名；DSA/QSA 分族字段是唯一入口。
- `model_type` 子串退出 normalize：`sharedExperts` / `sharedExpertGate` 看
  `shared_expert_intermediate_size`（vLLM qwen3_moe / Qwen4Exp）；MHC 看
  `mhc` / `hc_mult`；`mhcPostMultValue` 缺省时 MHC 开则用 vLLM 默认 2.0；
  vision gated MLP 看 `hidden_act`。
- `StructureNodeBase` 抽出共用字段；`cli`/`api`/`service`/`tests` 从
  `model_structure_viewer.resolve` 导入，删顶层 `resolver.py` shim。

**明确后置 / 不做**
- 拆 59 个 checkpoint 文件。组网按 `architectures[0]`（16 类）拆，对标 vLLM `models/<arch>`。
- 对账 triage DSL 重写（继续当 fixture）。
- `{matrix, vector, sfu, bytes}` 保留。
- 前端下权重、plan 搜索、服务指标。

## P2：条件性需求

以下事项只有在明确需求出现时才启动：

- 公开国产芯片规格扩展：每个字段必须有公开来源，缺项保持 unknown。
- 后端生产化：补充路径约束、remote code 沙箱、鉴权、请求限制、日志脱敏和会话级 settings。
- 更细的模型专项模块或公式：先确认现有 IR 能否表达，优先扩展 `details/modules.md` 对应的 registry、builder、layers、ops 和 formulas。

## 明确不做

- plan 搜索、自动推荐最优并行配置、TTFT/TPOT/吞吐等**服务指标**（含排队与调度）。
  Roofline 理论时间下界允许，必须标明"估计 / 下界"。
- 训练内存规划、运行时 trace、调度仿真和 KV transfer overlap 仿真。
- 前端引入 torch 或下载权重。

## 任务规则

每项待实现任务必须同时写清楚范围、代码入口、依赖、验收命令和不包含的内容。完成后从本文移入 changelog 或实现细节文档，不在本文件长期保留已完成事项。
