# 跨节点通信时间「可调」+ 逐 stage 算力时间 —— 开工前对齐设计

> 状态：设计/对齐稿，未开工（M12 / N4 并行扩展 + 立项池 #4）。
> 目的：把两项候选功能（① 跨节点 comm 可调；② 逐 stage 算力时间）在开工前
> 落成书面方案——评估合理性、对标成熟方案、给出文件级实现路径与边界红线。
> 联网调研日期：2026-09-23。协议以 parallel_protocol.md 为准，证据以
> evidence/parallelism/internode_comm.md、evidence/cost/stage_vectors.md 为准。

## 0. 现状基线（代码实证）

- 通信字节：cost/comm.js 的 ringAllReduceBytes / expertAllToAllBytes /
  pipelineP2PBytes / pdKvTransferBytes 已在真·多机 A100（81↔41）逐 N 验证，
  比值全 1.000（evidence/parallelism/internode_comm.md）。字节 n 不缺。
- 通信时间：cost/roofline.js 的 classifyRoofline 用 commTime = commBytes / link，
  link = options.interNode ? rates.interNodeBytesPerSecond : rates.intraNodeBytesPerSecond。
  五路（matrix/vector/sfu/memory/comm）取 max = overlap 静态上界，无 latency（α）项。
- 费率：cost/chips/rates.js 的 interNodeBytesPerSecond =
  interconnect.inter_node.bandwidth · η.comm；三张公开卡（A100/H100/L40S）无
  inter_node 规格 → 该值 null → 跨节点 commTime = null（unknown）。实测参考
  ≈ 79.6 Gb/s（单 rail，N=2 纯 inter-node，64 MB all-reduce），刻意未写进芯片规格。
- 效率因子：cost/efficiency.js 默认 flops 0.7 / hbm 0.7 / comm 0.6 /
  intra_node_comm 0.8，芯片与 UI 可覆盖。
- UI：components/CostSummary.jsx 已有 interNode 复选框（默认 false，Q7①）；
  开启后 roofline 走 inter_node 费率行——但公开卡该行为 null 时时间即 unknown。
- 逐 stage：
  - cost/ui.js 的 costByFormulaGroup(cost) 已按 FORMULAS.group
    （gemm/attention/moe/layernorm/…）聚合逐算子 MACs 构成表（仅 MACs，未过 roofline）。
  - aggregateCost 返回的 cost.nodes[] 每行带完整动作向量 row.actions
    （matrix/vector/sfu/bytes/computeDtype）；summarizeActions 已做模型级聚合。
  - CostSummary.jsx 已有 per-stage 访存路（HBM）分解（stageRates），
    注释明写「计算路 per-stage 待 stage 级 actions 落地后再扩展」。
  - 证据 evidence/cost/stage_vectors.md（Qwen3-0.6B/A100）：stage matrix(FLOPs)
    可导且对账；stage TIME 在 0.6B 尺度被 kernel launch 开销主导（attn 占 77% 时间
    但非 FLOPs 主项 → launch-bound）→ 与 roofline stage 划分不匹配。

---

## 1. 第 4 项：跨节点 comm「可调」

### 1.1 评估：合理，且标准

精确映射支柱④（瓶颈在通信），不越界。缺口只是 β（inter-node 有效带宽）这一个
可调标量——字节 n 已算准。把 β 做成用户可调输入 = 业界标准。

### 1.2 成熟方案（同一套 alpha-beta / Hockney 模型）

- MPI 集合通信理论：T = α + n/β，α=链路延迟、β=有效带宽；集合按算法给系数
  （ring all-reduce = 2(N-1)α + 2(N-1)/N · n/β）。出处：Thakur/Rabenseifner/Gropp
  《Optimization of Collective Communication Operations in MPICH》(2005)；
  Chan et al.《Collective communication: theory, practice, and experience》(2007)。
  这就是「把 β（可选 α）做成可调输入」的理论出处。
- NCCL busbw vs algbw + 效率：实测有效带宽 < 峰值 → 需效率因子；MSV 已有
  η.comm = 0.6。参考 NVIDIA nccl-tests 的 busbw 定义。
- Calculon（SC'23，NVIDIA/Hoefler，github.com/paramath/calculon）：网络分多层
  tier，每层参数化 bandwidth/latency/size/efficiency，逐集合算时间 + overlap。
  「可调网络参数」最完整对标。
- LLMCompass（ISCA'24，arxiv.org/abs/2312.03134）：通信性能模型带 latency+bandwidth，
  区分拓扑与集合类型。
- ASTRA-sim / Chakra：链路 BW/latency + 拓扑全可调、逐集合建模——偏仿真，
  对 MSV 属越界重量级，仅作拓扑扩展参考。
- Vidur（arxiv.org/abs/2405.05465）：profile 集合再拟合——实测校准方向，
  principles.md §1 明确不做，需要真实端到端就指向它。

### 1.3 实现方案（文件级，贴合边界）

1. 可调 β 输入（核心）：CostSummary.jsx 的「假设」配置区，在 interNode 复选框旁增
   一个数值输入 interNodeBandwidthOverride（UI 统一单位）。默认空 = 仍 unknown；
   占位提示写实测参考 ≈ 79.6 Gb/s 单 rail。
2. 费率透传：chipRates(chip, options) 增可选 options.interNodeBandwidth 覆盖：
   interNodeBytesPerSecond = (override ?? chip.interconnect?.inter_node?.bandwidth) · η.comm。
   不改芯片数据文件（chips/public.js 保持无 inter_node，诚实）。
3. classifyRoofline 透传：options 增 interNodeBandwidth → 传入 chipRates。
4. 可选进阶（单独对齐后再上，不在首版）：加 α（latency）项，
   commTime = α + commBytes/β。对 decode 小消息/多次小集合更准；但当前「只用 n/β +
   取 max」是刻意的下界口径，加 α 会从「字节/带宽下界」变成「含固定开销的估计」，
   需改口径标注并单独对齐。首版只做可调 β。
5. 标注：产物固定标「估计 / 下界」；β 由用户输入时 value_source 徽标标 user_input
   （区别于芯片规格 spec）。

### 1.4 红线

产物只能是「单次集合 / 单卡通信时间下界」，不得升格为 TTFT/TPOT/吞吐。

---

## 2. 逐 stage 算力时间方案

### 2.1 评估：拆两半

- (a) stage 级 matrix/bytes 聚合——合理，直接做。数据已存在
  （cost.nodes[].actions + costByFormulaGroup 已按 group 聚合 MACs），只差把完整
  动作向量按 group 聚合并暴露 rollup。属支柱④。
- (b) 逐 stage「时间」——合理但有风险。roofline 只是下界；小模型实测被 launch/overhead
  主导（stage_vectors.md 已证实），逐 stage 时间不匹配 wall-clock，易被误读成真实耗时。
  仅大模型（GEMM 大、compute/带宽 bound）时逐 stage roofline 时间才逼近现实。
  必须以「理论下界」呈现并保留 launch-bound 诚实注记。

### 2.2 成熟方案

- LLM-Viewer（arxiv.org/abs/2402.16363）：layer-wise / 逐算子 roofline，逐层给
  compute+memory 时间并明确标「理论」。最直接对标——证明「逐 stage roofline 时间当
  理论量呈现」是业界认可做法。
- Timeloop/Accelergy ERT（MSV 已在用）：动作次数 × 单价 → 逐单元时间，天然支持 stage 聚合。
- Paleo（ICLR'17）：解析模型分 compute time + communication time，并对 kernel 加固定
  开销/带宽项——处理「小算子 launch 偏差」的经典办法（若将来要 launch 项可参考）。
- Calculon：逐层 compute time 含 efficiency + 固定开销。
- Vidur：逐算子 profile 拟合出含 launch/调度的真实 stage time——实测校准/仿真，MSV 不做。

### 2.3 实现方案（文件级）

1. stage 级动作向量聚合：cost/ui.js 新增 actionsByFormulaGroup(cost)：与
   costByFormulaGroup 同构，但累加完整 row.actions
   （matrix/vector/sfu/bytes.{weights,actIn,actOut,kvRead,indexRead}/computeDtypes）
   到各 FORMULAS.group 桶；compute 未完整时返回空表（不猜）。
2. 逐 stage 五路时间：对每个 group 桶调 classifyRoofline(groupActions, chip,
   { dtype, efficiency, interNode }) → 得该 stage 的五路时间 + bound 分类。
   commBytes 归属：**已实现**（comm.js `communicationBytesByFormulaGroup`）——通信字节按
   **发起算子自身的功能域**归组（TP all-reduce→gemm、EP all-to-all→moe），与算力/访存
   同一归组键、单源无 role→域 映射表；组件把各域 commBytes 合并进桶再过五路 roofline。
   PP 的 P2P 是 stage 边界、不属算子域，**刻意不计入**逐 stage（仍由 planCommunicationBytes
   的 ppBytes 在模型级体现）。
3. UI：复用现有 Cost Lens 分栏，把 MACs 构成表升级为「MACs 占比 + 逐 stage 时间下界
   + bound 标签」，沿用现有 per-stage HBM 展示位。
4. 标注（强制）：显式标「理论下界 / 估计」；保留诚实注记——小模型 stage time 被 launch
   主导、roofline 是下界，仅大模型 GEMM 主导时才逼近。
5. 不做：不为「对齐 wall-clock」引入 launch-overhead 拟合项（那是 Paleo/Vidur 的校准
   闭环，越 principles.md「不做实测校准」边界）。

### 2.4 红线

逐 stage 时间只能叫「理论下界 / 估计」，不得叫 stage 级 TTFT/耗时实测。

---

## 3. 共同边界与验收

- 两项都停在「理论 / 下界」，标注「估计/下界」，不得叫 TTFT/TPOT/吞吐；需要真实端到端
  指向 Vidur（principles.md §1 明确不做清单）。
- 验收沿用仓内纪律：焦点单测（cost/__tests__/{comm,roofline}.test.js 扩展）+ 全量
  npm test / npm run verify:models + 生产 build + 真实浏览器验证（逐 stage 面板与 comm
  可调输入在 desktop/mobile 呈现、无 console 报错）。
- PR 描述写明服务支柱④，且逐条对照本文红线。

## 4. 参考文献

- Williams, Waterman, Patterson. Roofline: An Insightful Visual Performance Model. CACM 2009.
- Thakur, Rabenseifner, Gropp. Optimization of Collective Communication Operations in MPICH. IJHPCA 2005.
- Chan, Heimlich, Purkayastha, van de Geijn. Collective communication: theory, practice, and experience. CCPE 2007.
- Qi, Wang et al. Paleo: A Performance Model for Deep Neural Networks. ICLR 2017.
- Isaev, McDonald, Dennison, Hoefler. Calculon: high-level co-design of systems and LLMs. SC 2023. github.com/paramath/calculon
- Yuan et al. LLM Inference Unveiled: Survey and Roofline Model Insights (LLM-Viewer). arxiv.org/abs/2402.16363
- A Hardware Evaluation Framework for LLM Inference (LLMCompass). arxiv.org/abs/2312.03134
- Agrawal et al. Vidur: A Large-Scale Simulation Framework For LLM Inference. arxiv.org/abs/2405.05465
- NVIDIA nccl-tests（busbw/algbw 定义）；ASTRA-sim / Chakra（拓扑级通信仿真，越界参考）。
