# NV 验证清单（NVIDIA GPU 运行时待验证项）

> 锚点：commit `b2ee018`（60 个内置模型，含 DeepSeek-V4.1-Flash）。
>
> **本文件收口触发池里“需 NVIDIA GPU 运行时才能验证”的项** —— 它们无法在当前纯前端 /
> CPU 的 MSV 仓库里闭合（MSV 只做“config → 结构图 → 理论估算”，是静态工具）。出处见
> [`../implementation_plan.md`](../implementation_plan.md) 触发池与
> [`../refactor_plan.md`](../refactor_plan.md) 后续立项池。
>
> **纪律（沿用仓库既有取证规范）**
> - 不许凭空写结论：每项必须有**在机证据**（命令回显文本 / 截图 / 校验和 / 产物文件），
>   没有 GPU 在机证据前状态一律 `待验证`，禁止把理论估算冒充实测。
> - 分析型 / 取证型验证保持**只读**：不重启、不发布、不改远端状态（除非单独授权）。
> - 主机接入走既有 skill：`dev-machine-access`（H20/A800/A100 跳板）、`nvidia-pd-host-check` /
>   `nvidia-pd-container-check`（环境准出）、`model-download`（取权重）、`perf-analysis`（日志→xlsx/图）。
> - 一次只对一个明确的 checkout / host / image / SHA 取证；不拿兄弟证据顶替。
> - 触发未到不动工：每项带触发判据，判据未命中只登记、不开工。

## 汇总

| ID | 项 | 触发判据 | 为何需 NV | 状态 |
|---|---|---|---|---|
| NV-1 | 结构对账真值化（`compare_structure.py` 真实模型） | 对账出现未落入 `canonical_path_contract.json` 四桶的 diff | 需真实框架实例化 nn.Module 树（含自定义 kernel/量化） | **已验证（2026-09-17，A100/transformers 5.17.0）：59/60 零残留；DeepSeek-V4.1-Flash 构造受阻边界见 NV-5。证据 `nv_evidence/nv1/`** |
| NV-2 | framework execution profile（vLLM vs SGLang 有效宽度） | 第一次要对比同模型在两框架的有效 attention/MoE 宽度 | 需在 GPU 上起两套 serving 栈实测 | 待验证 |
| NV-3 | per-stage roofline / evidence I/O shape | UI 或对账需要 stage 级动作向量 | 需真实 profiler（nsys/ncu 或框架计数器） | 待验证 |
| NV-4 | A2 kernel 口径对齐（flash-attention scores/probs） | 需 kernel 级口径而非理论上限 | 需 GPU 上跑 flash-attention kernel 取实测 | 待验证 |
| NV-5 | DeepSeek-V4.1-Flash 运行时/权重实证 | 拿到实际 checkpoint / safetensors index，或要跑推理 | 需真实权重 + GPU 推理（fp4/fp8、engram、DSpark 投机） | 待验证 |
| NV-6 | 后端生产化（部署硬化） | 真正对外部署 | 运行时/部署环境（非 GPU 计算，独立登记） | 待验证 |

---

## NV-1 结构对账真值化

- **触发判据**：对账出现未落入 `src/model_structure_viewer/verification/fixtures/canonical_path_contract.json`
  四桶（only_transformers / only_msv / mismatch / 已登记豁免）的 diff；或前端 `compactRanges`
  与后端 `fold.py` 折叠出现未分类漂移。
- **为何需 NV**：`compare_structure.py` 现状是 meta 构造 + 自测；**真实对账**要在真实框架里
  实例化每个 catalog 模型的 nn.Module 树（尤其 V4.1-Flash 的原生 `model.py`、fp4/fp8 量化、
  DSA/indexer/engram 自定义算子），meta 设备与自定义 kernel 在无 GPU 环境常无法构造。
- **依赖**：H20/A800/A100 host（`dev-machine-access`）；transformers / vLLM / SGLang；
  对应 checkpoint（`model-download`，大模型仅取 config + index，逐张量对账再取权重）。
- **复现**：
  1. 在 GPU host 逐模型 `AutoModelForCausalLM.from_pretrained(..., torch_dtype=..., trust_remote_code=True)`
     或 vLLM/SGLang 加载，dump `named_modules()` 的 `module_path / class_name / param_shapes`。
  2. 本地 `node ../scripts/verify-builtin-models.mjs` 产 MSV 结构图；将后端 dump 喂
     `compare_structure.py` 的 `diff_module_evidence`。
  3. 新增的 diff 若属真实语义分支，按 §6.4 契约扩 `canonical_path_contract.json` 桶。
- **期望证据**：每模型三分类 diff 的 JSON 产物 + 命令回显；新扩桶的 fixture 变更。
- **判定**：全 60 模型 diff 落入四桶零残留；折叠谓词漂移归零或登记。
- **状态**：**已验证（2026-09-17）**。在机：A100-SXM4-80GB ×8 / CUDA 13.0 / transformers 5.17.0 /
  torch 2.13.0+cu130；harness `scripts/nv1_reconcile.py` + `verify-builtin-models.mjs --dump-graphs`。
  结果：60 模型中 59 个 meta 构造成功且三桶零 unclassified 残留（`structurally_consistent=true`）；
  `deepseek-ai/DeepSeek-V4.1-Flash` 因 `deepseek_v41` 无框架支持（`auto_map:null`、未带 config 类）
  构造受阻，按 NV-5 边界登记，不伪造通过。修复的真实结构差异：DSA `indexer.k_norm` 前端 RMSNorm→LayerNorm
  （补 `affine_bias`）、MiniMax-M3 解码层 pathing（`language_model`→`language_model.layers`）+ 补 `embed_tokens`
  + 稠密 MLP 改用 `dense_intermediate_size`、Kimi-K3 `tie_weights` 兼容垫片泛化。合法命名/粒度差异按 §6.4
  逐条登记进 `canonical_path_contract.json`（known_divergences 达 54 条，均带 reason+source）。
  回归：后端 `pytest` 183 pass、前端 `node --test` 410 pass、`verify:models` 60/60、W5 恒等式全绿。
  证据：[`nv_evidence/nv1/`](nv_evidence/nv1/)（per-model diff JSON + `summary.json` + `env.txt`）。

## NV-2 framework execution profile（vLLM vs SGLang 有效宽度）

- **触发判据**：第一次需要对比 vLLM 与 SGLang 在同一模型上的有效 attention/MoE 宽度差异。
- **为何需 NV**：有效宽度、expert ownership/placement、KV partition、dispatch/combine 方式是
  **运行时**属性（受 TP/EP/DP 轴与框架实现支配），config 与结构图给不出，必须在 GPU 上起两套
  serving 栈观测。
- **依赖**：GPU host + vLLM + SGLang；同一 checkpoint、同一并行度。
- **复现**：分别用 vLLM 与 SGLang 起服务，打印/抓取每层有效 attention 头数、专家分片与归属、
  KV 分区、dispatch/combine 路径；对齐到 MSV 的 `parallel_protocol.md` 九项裁决口径。
- **期望证据**：两框架的层级宽度/专家归属表 + 版本/启动参数回显。
- **判定**：两框架有效宽度差异有据可查，或确认一致；差异项归入 framework execution profile 层设计。
- **状态**：待验证（载体 JSON 视图 / UI 分栏 / CLI 选项待需求定型后裁决）。

## NV-3 per-stage roofline / evidence I/O shape

- **触发判据**：UI 或对账需要 stage 级动作向量（当前 introspect 不产数值，是诚实缺项）。
- **为何需 NV**：stage 级 roofline 的 I/O shape 与实际访存要真实 profiler 采样。
- **依赖**：GPU host + `perf-analysis` skill（vLLM/SGLang benchmark_serving 日志 → xlsx/图）；
  nsys/ncu 可选。
- **复现**：跑 benchmark_serving（或 nsys/ncu），采 per-stage 的 matrix/vector/sfu/bytes 与
  实测 TTFT/TPOT/吞吐；与 MSV 的理论 roofline（`cost/roofline.js`）对齐、标注偏差来源。
- **期望证据**：benchmark 日志 + `perf-analysis` 产出的 xlsx/对比图 + 采样命令回显。
- **判定**：stage 级动作向量非 null 或显式 missing；实测与理论偏差有归因。
- **状态**：待验证。

## NV-4 A2 kernel 口径对齐（flash-attention scores/probs）

- **触发判据**：需要 kernel 级口径，而非现在的理论上限口径。
- **为何需 NV**：MSV 的 A2 假设 scores/probs 中间量按理论上限（4×）计，flash kernel 下融合单遍、
  不落 HBM；实测口径需在 GPU 上跑 flash-attention kernel 取证。
- **依赖**：GPU host + flash-attn / 框架 attention backend；ncu 采 kernel 级访存。
- **复现**：对代表模型（含 MLA / DSA / dsv4 稀疏）用 ncu 采 attention kernel 的实际 scores/probs
  访存与 FLOPs，与 MSV `sdpa_attention` / `dsv4_*` 的理论口径对比。
- **期望证据**：ncu 报告 + kernel 访存/FLOPs 数值 + 对比表。
- **判定**：确认保持理论口径，或给出 kernel 级修正口径（需另行对齐、更新 `cost_counts.md` A2 假设）。
- **状态**：待验证。

## NV-5 DeepSeek-V4.1-Flash 运行时 / 权重实证（本次接入的保真边界）

> 本次接入（commit `b2ee018`）已通过 header-truth 逻辑元素恒等式（ratio=0.9998）、
> `verify:models` 60/60、真实 Chrome 全量扫描。以下是**需真实 checkpoint / GPU 才能进一步收口**
> 的保真边界，非阻塞项，逐条登记：

- **权重逐张量恒等式**：现按 `header-truth.json`（manual，`parameterTotal=508,182,659,298`、
  `tensor_count=96085`）的 dtype 逻辑元素对账。缺 `model.safetensors.index.json`（.gitignore 不入库），
  未做逐张量路径核对。触发：拿到 index/checkpoint 后，逐张量 shape vs 结构声明对账。
- **compressor / indexer 层位**：现复用 V4 的 `compress_ratio>1` 启发式摆放 compressor，
  **未**按 config 的 `kv_source_layer_ids=[2,8,14,20]` / `index_source_layer_ids=[2,8,14,20,24,28,32,36]` /
  `candidate_source_layer_id` 精确摆放（与既有 V4 保真度一致）。触发：真实模块树对账（见 NV-1）暴露层位差异。
- **engram / DSpark 运行时**：engram（第 1/14 层 n-gram 门控写回）与 DSpark 投机头（128 专家、
  markov+confidence）的**运行时行为**（接受率、显存、吞吐）未实测。触发：跑 V4.1 推理。
- **依赖**：真实 checkpoint（`model-download`）+ 支持 `deepseek_v41` 的框架 + GPU host。
- **期望证据**：safetensors index / 逐张量对账 JSON；真实模块树 diff；推理日志（接受率/显存/吞吐）。
- **判定**：逐张量零差异（或登记容差）；层位与真值一致或修正结构；运行时指标有据。
- **状态**：**部分已验证（2026-09-17，config + HF safetensors index/header，未下权重区）**。
  已闭合：张量数 96085、参数量级、文本层 40 / MTP 3 / vision 32 / experts 384 —— 均与 config 一致；
  **compressor 层 = [2,8,14,20]、indexer 层 = [2,8,14,20,24,28,32,36] 与 `kv_source_layer_ids` /
  `index_source_layer_ids` 逐位一致**，并据此修正前端保真差（compressor/indexer 改按 source 层摆放，
  `normalize.js` + `ops/index.js`，V4-Flash/Pro 行为不变、对账仍 0 残留）；**逐张量恒等式：range-read 全
  48 分片 safetensors 头部聚合与 header-truth.json 逐 dtype 零差**（tensor_count 96085 / parameterTotal
  508,182,659,298 / mtp 2401 / BF16·F32·F8_E4M3·F8_E8M0·I8 全对）。证据
  [`nv_evidence/nv5/`](nv_evidence/nv5/)（`v41_config_index_reconcile.md` + `v41_header_tensor_identity.json`）。
  仍待真实权重/GPU 推理：engram/DSpark 运行时（接受率/显存/吞吐）、后端 transformers 构造（需
  `deepseek_v41` 框架支持或补齐 remote code）。

## NV-6 后端生产化（部署硬化）

- **触发判据**：真正对外部署 `backend/app.py`。
- **为何登记在此**：属**运行时/部署**环境项（非 GPU 计算），与上面 GPU 项同为“出了 MSV 静态仓库才能验”，
  一并收口。
- **依赖**：部署环境；不需 GPU 计算。
- **复现/清单**：路径约束（禁越权读）、remote code 沙箱（`trust_remote_code` 隔离）、鉴权、限流、
  日志脱敏；上线前逐项过。
- **期望证据**：部署配置审阅记录 + 安全项逐条勾验。
- **判定**：五项硬化全部落实。
- **状态**：待验证。

---

## 关联

- 触发池权威住址：[`../implementation_plan.md`](../implementation_plan.md)（§触发池）、
  [`../refactor_plan.md`](../refactor_plan.md)（后续立项池 / M11 遗留）。
- 对账代码：`src/model_structure_viewer/verification/compare_structure.py`、
  契约样例 `src/model_structure_viewer/verification/fixtures/canonical_path_contract.json`。
- 口径假设：[`cost_counts.md`](cost_counts.md)（A1–A7）、[`../refactor_plan.md`](../refactor_plan.md)（A2）。
- 纯前端 / 文档项（不需 GPU，另行处理，不在本清单）：Cost Lens 按 `FORMULAS.group` 分栏、
  算子 `explanation` / 芯片 `notes` / `collectDiagnostics` 双语、`cost_counts.md` 42 条 bytes 明细、
  前端公式数学式化。
