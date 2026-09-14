# 待实现计划

本文记录拍板的**后续终态**、已闭合项与触发池。已经完成的能力以代码、测试和 [`CHANGELOG.md`](../CHANGELOG.md) 为准；历史版本通过 Git 提交记录追溯。终态正文见「后续终态（2026-09-14）」。正确性账本已闭合；触发池不是产品 backlog，触发未到不动工。

## 当前基线

以下能力已经落地，不再作为待实现任务：

- 前端静态结构生成、内置模型 catalog、HF/ModelScope 配置读取和 safetensors header 真值接入。
- 模型 registry、通用和专用 builder、layers、ops、公式、IR、materializer 和诊断链路。
- 结构图、Layers、Inspector、JSON/Mermaid/DOT 导出、芯片 Cost Lens、并行投影和 PD 分析。
- Python API/CLI、local cache、transformers meta-device 验证和 GitHub Pages 构建流程。
- Graph IR v2 节点事实、Graph-first 消费和已有 parity 基线测试。Graph 是唯一结构载荷，无 `structure.root`。
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

**后续终态（2026-09-14）**：不是最短路径，是各账本收口后的目标态。已完成能力见上文基线与 ✅ 条目。

### 结构

- 产品 Graph IR = 折叠后的图。同构 decoder / 专家列表只留一份代表 + `repeat`；walker 用 `childRepeatMultiplier` / `residentRepeat` 还原容量与计算。UI 若展开某一层，是视图，不写进 IR。
- 前端 `compactRanges` 是产品折叠；后端 `fold.py` 是 Transformers evidence 适配器。两套实现不合并。同构谓词若对账漂移，再抽共享规则；未漂移不抽。
- 组网按 `architectures[0]`（16 类文件）。MTP/DSpark 写在该架构文件里。未知架构 `unsupported`。
- 节点 id = HF `_modules`。checkpoint 绑定剥 `model.` / `language_model.` 后路径相等。

### 算子与成本（四本账并行，互不塞入）

```text
容量     graphWeightCapacity / weightMatrices     驻留多少（tied shared 不双计）
GEMM     counts.matrix  ↔ T4 / flop_registry      MAC；torch FLOP = 2×MAC
增量计算 counts.vector / counts.sfu               flop_registry 记 0 的部分
流量     counts.bytes.{weights,actIn,actOut}      torch 明确不算 memory movement
```

- `operator_id` = 算法身份（QSA/DSA/MSA 分条）。`FORMULAS[id].group` = SGLang `kernels/ops/` 功能域元数据，不改 id。抄有对位的组：`gemm` `attention` `moe` `layernorm` `activation` `embeddings` `elementwise` `memory` `mamba`。不抄执行域。不自造 `residual_mixing`。vision 不单列。ple 暂不归组（SGLang `qwen4_ple.py` 也未进 `_GROUPS`）。缺 group / 非法组名由护栏棘轮=0。
- A17 `topk` 写出 `(values, indices)`：`actOut` = `T·k·bytesPerElement`（values）+ `T·k·4`（int32 索引）。分解链 `reduce_sum` 的读有写来源。matrix 仍为 0。
- embedding：容量走表声明；forward gather 走 `embedGatherCounts`（matrix=0，bytes=行拷贝，不扫全表）。tied lm_head `shared:true`；T4 仅在 tied 时把 embedding 加回 `N_eff`（那张表当 GEMM 用了一次）。
- RMSNorm：`rmsnormCounts` 的 vector / sfu / `bytes.weights` 进入身份测试。T4 继续从 `N_eff` 剥 norm 元素。
- 身份测试三套并存：T4 matrix（GEMM）；S3 图声明对 header（容量）；新增 bytes + SFU 身份（流量与超越运算）。REGISTERED 保持空。
- `{matrix, vector, sfu, bytes}` 单位约定不变。matrix 夹具继续 Linear / BMM / 深度可分 Conv1d。不对 catalog 整模型跑 FlopCounter。

### 来源与对账

- 前后端两套 resolver 运行时不合并。共享契约见 [`details/models/source_contract.json`](details/models/source_contract.json)：键 `repo_id + revision + cache_dir`；来源 `auto|builtin|local|hf|config`；`auto` fallback = builtin → local → hf；ModelScope 空/`main` → `master`；错误分类 config=400 / not_found=404 / remote=502。前后端测试对这份 JSON。
- verify triage 继续 fixture（`canonical_path_contract.json` 四桶）。不写 DSL。
- `source_ref`：58/59 已入库。Kimi-K3 永不 dump / verify。缺席产物节点 `source_ref` 为 null。
- schema：`StructureNodeBase` 已抽。Graph 协议字段（`schema_version` / `parent_id` / `order` / `canonical_id`）锁在同一份 `source_contract.json`；后端不当第二份产品结构源。

### 文档与护栏

- `details/models.md`：清单按 `architectures[0]` 由生成器 + `docs:check` 守护。「读」段（结构类表、判据、S13）手写。不要整篇生成。
- `operators_reference.md` 按 `FORMULAS[id].group` 聚合，判断只写进 `FORMULAS`。`ple` 为 ungrouped。
- §8.1 家族名文件数只许下降。§7 芯片字段无公开来源保持 unknown。

### 明确不做（终态里也不做）

拆 59 个 checkpoint 文件；产品图展开全层；把 embedding/norm 乘进 T4 matrix；对账 triage DSL；合并前后端 resolver 运行时；Kimi-K3 dump/verify；整模型 FlopCounter；前端下权重；plan 搜索；服务指标；自造 operator group。

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

[`principles.md`](principles.md) 已定为强制约束。原则收口波次（W0–W6）已走完；
存量偏离清账见该文 §10。路线考古见 [`refactor_plan.md`](refactor_plan.md)。
核心分界不变：config 归一层只做字段归一（§4.7），方案决定权归组网；
组件配方为一级概念，家族名只在薄解析层（§4.3）。

### 终态正确性已闭合（2026-09-14）

下列条目曾写成「未排期缺口」，实现与契约已经对齐，不再当待办：

- **§5 source_ref**：采集 / 绑定 / Inspector 已接通。catalog 58/59 已
  `msv dump-source-ref --source builtin` 入库。Kimi-K3 永不 dump / verify
  （Hub modeling import 拉 `fla` / Triton；transformers 无 `kimi_k3`）。
  绑定脚本 58 模型通过。`verify --graph` 构造通过才算脚本失败；
  `structurally_consistent` 只报告。缺席产物节点 `source_ref` 为 null。
- **§6.4 来源解析契约**：两套运行时不合并。共享契约已锁
  [`details/models/source_contract.json`](details/models/source_contract.json)
  （键 / 来源类型 / auto fallback / revision 默认 / 错误分类 / Graph 协议字段）。
  前后端测试对这份 JSON。不是再写第三套路由。
- **§3.8 容量旁路**：cost 只 walk 图。闭式已删。身份测试期望侧 walk 图。
- **统一结构协议**：`StructureNodeBase` 已抽。Graph 协议字段锁在同一份
  `source_contract.json`。后端不当第二份产品结构源。`layoutGraph` 只接受
  `structure.graph`，无 `structure.root`。
- **FlopCounterMode**：独立算子夹具已接（Linear / BMM / 深度可分 Conv1d）。
  整模型 forward 抽查终态里也不做（catalog 无权重）。

### 触发池（触发未到不动工）

不是产品 backlog，也不是「下一步默认做前端体验」。每条带触发判据：

- **Cost Lens 按 `FORMULAS.group` 分栏**（UI）。触发：需要按功能域看成本，而不是只看整图合计。
- **verify fixture 桶扩展**。触发：对账出现未落入 `canonical_path_contract.json` 四桶的 diff。
- **折叠谓词共享**。触发：前端 `compactRanges` 与后端 `fold.py` 出现未分类漂移。终态默认两套实现不合并。
- **framework execution profile**。触发：第一次需要对比 vLLM 与 SGLang 在同一模型上的有效宽度。
- **per-stage roofline / evidence I/O shape**。触发：UI 或对账需要 stage 级动作向量。
- **新架构配方文件**。触发：新的 `architectures[0]` 或新 catalog 条目。门槛不变：字段判据、header-truth、生成清单、`docs:check`。公共模型工作没有降优先级，只是没有新架构时不预造空壳。
- **§7 国产芯片条目**。触发：有公开来源的字段要入库。缺项保持 unknown。
- **后端生产化**。触发：真正对外部署。路径约束、remote code 沙箱、鉴权、限流、日志脱敏。
- **家族知识 5 住址收口 / §8.1 继续下降**。触发：接新模型家族。棘轮基线现 6，只许下降。

新增模型时同时更新 catalog、来源记录、模型专项说明和验证结果，避免只加配置。

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

**明确后置 / 不做**：见上文「后续终态」末节，不在此重复。

## P2：条件性需求

已并入上文「触发池」。不要在这里另开一份待办。

## 明确不做

- plan 搜索、自动推荐最优并行配置、TTFT/TPOT/吞吐等**服务指标**（含排队与调度）。
  Roofline 理论时间下界允许，必须标明"估计 / 下界"。
- 训练内存规划、运行时 trace、调度仿真和 KV transfer overlap 仿真。
- 前端引入 torch 或下载权重。

## 任务规则

每项待实现任务必须同时写清楚范围、代码入口、依赖、验收命令和不包含的内容。完成后从本文移入 changelog 或实现细节文档，不在本文件长期保留已完成事项。
