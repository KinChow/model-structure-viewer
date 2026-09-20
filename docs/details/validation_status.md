# 验证状态总览（按 structure / memory / cost / parallelism 维度）

> 锚点：commit `b2ee018`（60 个内置模型，含 DeepSeek-V4.1-Flash）。
>
> 本文记录各验证项的「计划 vs 已完成」状态；证据工件索引见 [`evidence/README.md`](evidence/README.md) 的覆盖矩阵。
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

| 维度 | 项 | 触发判据 | 为何需 GPU 运行时 | 状态 |
|---|---|---|---|---|
| structure | 结构对账真值化（`compare_structure.py` 真实模型） | 对账出现未落入 `canonical_path_contract.json` 四桶的 diff | 需真实框架实例化 nn.Module 树（含自定义 kernel/量化） | **已验证（2026-09-17，A100/transformers 5.17.0）：59/60 零残留；DeepSeek-V4.1-Flash 构造受阻边界见 V4.1 结构/KV 实证节。证据 `evidence/structure/`** |
| parallelism/memory | framework execution profile（vLLM vs SGLang 有效宽度） | 第一次要对比同模型在两框架的有效 attention/MoE 宽度 | 需在 GPU 上起两套 serving 栈实测 | **部分已验证（2026-09-17/18，A100/SGLang/Qwen3-0.6B）：TP=1 宽度/GQA/KV(112KiB) 与 MSV 一致；TP=2/4 权重÷tp（每卡 0.58/0.30GB，误差~3%）+ KV 池 ×2.002/×4.008 验证前端 TP 折叠。EP/vLLM 未做。证据 evidence/parallelism/** |
| cost | per-stage roofline / evidence I/O shape | UI 或对账需要 stage 级动作向量 | 需真实 profiler（nsys/ncu 或框架计数器） | **部分已验证（2026-09-17，A100/SGLang/Qwen3-0.6B）：① 聚合实测落 MSV roofline 地板之上（时长 1.99×、TPOT 1.65×）；② 逐算子对真值——线性 MSV MACs×2 == FlopCounterMode FLOPs 逐位相等、GEMM compulsory 读侧 == ncu DRAM 读(<0.4%)、bound 分类逐项一致。证据 evidence/cost/** |
| cost | A2 kernel 口径对齐（flash-attention scores/probs） | 需 kernel 级口径而非理论上限 | 需 GPU 上跑 flash-attention kernel 取实测 | **已验证（2026-09-17）：ncu flash_fwd_kernel 实测 scores/probs 不落 HBM，A2 物化口径确认为保守上界；证据 evidence/cost/flash_kernel_caliber.md** |
| structure/memory | DeepSeek-V4.1-Flash 运行时/权重实证 | 拿到实际 checkpoint / safetensors index，或要跑推理 | 需真实权重 + GPU 推理（fp4/fp8、engram、DSpark 投机） | 待验证 |
| backend | 后端生产化（部署硬化） | 真正对外部署 | 运行时/部署环境（非 GPU 计算，独立登记） | **已审计（2026-09-17）：路径/remote code/鉴权 3 项缺失(P0)+限流/脱敏 2 项部分，与本地工具定位一致；硬化待部署触发，见 backend_audit.md** |

---

## 结构对账真值化（structure）

- **触发判据**：对账出现未落入 `src/model_structure_viewer/verification/fixtures/canonical_path_contract.json`
  四桶（only_transformers / only_msv / mismatch / 已登记豁免）的 diff；或前端 `compactRanges`
  与后端 `fold.py` 折叠出现未分类漂移。
- **为何需 GPU 运行时**：`compare_structure.py` 现状是 meta 构造 + 自测；**真实对账**要在真实框架里
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
  torch 2.13.0+cu130；harness `scripts/evidence/structure/reconcile_reduced.py` + `verify-builtin-models.mjs --dump-graphs`。
  结果：60 模型中 59 个 meta 构造成功且三桶零 unclassified 残留（`structurally_consistent=true`）；
  `deepseek-ai/DeepSeek-V4.1-Flash` 因 `deepseek_v41` 无框架支持（`auto_map:null`、未带 config 类）
  构造受阻，按 V4.1 实证边界登记，不伪造通过。修复的真实结构差异：DSA `indexer.k_norm` 前端 RMSNorm→LayerNorm
  （补 `affine_bias`）、MiniMax-M3 解码层 pathing（`language_model`→`language_model.layers`）+ 补 `embed_tokens`
  + 稠密 MLP 改用 `dense_intermediate_size`、Kimi-K3 `tie_weights` 兼容垫片泛化。合法命名/粒度差异按 §6.4
  逐条登记进 `canonical_path_contract.json`（known_divergences 达 54 条，均带 reason+source）。
  回归：后端 `pytest` 183 pass、前端 `node --test` 410 pass、`verify:models` 60/60、W5 恒等式全绿。
  证据：结论见 [`evidence/structure/per_model_reconcile.md`](evidence/structure/per_model_reconcile.md)（原始 per-model diff JSON / `summary.json` / `env.txt` 由 `scripts/evidence/structure/reconcile_reduced.py` 重生）。

## 框架执行 profile · 有效宽度/并行（parallelism / memory）

- **触发判据**：第一次需要对比 vLLM 与 SGLang 在同一模型上的有效 attention/MoE 宽度差异。
- **为何需 GPU 运行时**：有效宽度、expert ownership/placement、KV partition、dispatch/combine 方式是
  **运行时**属性（受 TP/EP/DP 轴与框架实现支配），config 与结构图给不出，必须在 GPU 上起两套
  serving 栈观测。
- **依赖**：GPU host + vLLM + SGLang；同一 checkpoint、同一并行度。
- **复现**：分别用 vLLM 与 SGLang 起服务，打印/抓取每层有效 attention 头数、专家分片与归属、
  KV 分区、dispatch/combine 路径；对齐到 MSV 的 `parallel_protocol.md` 九项裁决口径。
- **期望证据**：两框架的层级宽度/专家归属表 + 版本/启动参数回显。
- **判定**：两框架有效宽度差异有据可查，或确认一致；差异项归入 framework execution profile 层设计。
- **状态**：**部分已验证（2026-09-17，A100-SXM4-80GB / SGLang / Qwen3-0.6B / TP=1）**。用户指定仅跑
  SGLang + Qwen3-0.6B（不跑 vLLM、不跑其它模型），故"跨框架对照"退化为**单框架 TP=1 稠密**基线：
  TP=1 无张量切分，单卡有效宽度 == config 宽度（退化恒等）。在机确认 flashinfer backend、GQA 16/8=2:1、
  每 token KV = 28×8×128×2×2 = 112 KiB（KV 池 578608 token ≈ 61.8 GiB，与 80GB×0.8 余量吻合）——
  runtime 有效宽度 / GQA 分组 / 每 token KV 字节与 MSV 结构口径逐项一致。跨框架(vLLM)与多卡(TP>1)
  有效宽度对照不在本次范围。证据 [`evidence/memory/sglang_width_kv.md`](evidence/memory/sglang_width_kv.md)。
- **状态（线 B · TP 折叠，2026-09-18，A100×4/SGLang/Qwen3-0.6B）**：补齐 TP>1 空缺（此前只 TP=1 退化）。
  真机 TP=2/4 vs 前端投影（`scripts/evidence/parallelism/tp_projection.mjs`）：**每卡权重 ÷tp**——实测 0.58/0.30 GB，前端 596/298 MB，
  误差 ~3%（`declaredWeightBytesPerCard` 成立）；**KV 池容量 ×tp**——578,608→1,158,357(×2.002)→2,318,630(×4.008)，
  与 `kvBytesPerCard` 的 GQA `min(tp, kv_heads=8)` 分片逐点一致；**all-reduce** 公式结构（Megatron 每层两段）
  确认，字节级 nsys/NCCL 实测留后续。EP（专家并行）与 vLLM 跨框架未做。证据
  [`evidence/parallelism/tp_parallel.md`](evidence/parallelism/tp_parallel.md)（原始 `frontend_tp_projection.json` / `sglang_tp_mem.log` 由 `scripts/evidence/parallelism/tp_projection.mjs` 重生）。
- **状态（线 B · EP + all-reduce 字节级，2026-09-18）**：**EP**——减层随机 qwen3_moe（8 experts/topk2/4层，零下载
  save_pretrained）起 SGLang `--tp2 --ep2`，MoE runner 打印 `E=4,N=768`（每 rank 4 专家 = experts/ep），
  与前端 `expertShardDivisor`（每 rank E/ep 完整专家）**逐点一致**；all-to-all 触发门控与公式结构确认（字节级
  nsys 留后续）。**all-reduce**——2×A100 NCCL 微基准 all-reduce [1,4096,1024]bf16，前端 `ringAllReduceBytes`
  =2(N-1)/N·D 与 NCCL ring 总线传输量 N=2 逐位吻合（8.39MB，比值 1.000）。证据
  [`evidence/parallelism/ep.md`](evidence/parallelism/ep.md) + [`evidence/parallelism/allreduce.md`](evidence/parallelism/allreduce.md)。
  未验证：混合 ETP/DeepEP/多机、all-to-all 与 all-reduce 的 nsys 字节级直测。
- **状态（8×A100 字节级收口，2026-09-18）**：补齐上一条的三处"留后续"。**all-reduce N 依赖**——扩到 N=2/4/8，
  前端 `2(N-1)/N·D` vs NCCL busbw×dt 逐 N 比值均 **1.000**（8.39/12.58/14.68 MB，(N-1)/N=0.5/0.75/0.875），
  N 依赖标度坐实。**all-to-all 字节级**——`scripts/evidence/parallelism/alltoall_bench.py` ep=2/4/8 搬运 dispatch 载荷 == 前端
  `B·T·topk·H·b`（16.78MB，比值 1.000，与 ep 无关），从"仅结构"升级为**字节直测**。**nsys per-kernel**——确认
  `ncclDevKernel_AllReduce_*_RING_LL` / `SendRecv` kernel，流量走 NVLink（PCIe≈0），per-kernel 时序与 busbw 同量级
  （GPU 指标为峰值%采样，字节精确口径仍以 microbench 为准）。证据 [`evidence/parallelism/nsys.md`](evidence/parallelism/nsys.md)。
  仍未验证：混合 ETP（moe_tp>1）/DeepEP、真·多机（跨节点 NCCL / PD 分离）——需换环境。
- **状态（混合 ETP / DeepEP，2026-09-18）**：补上一条的"混合 ETP"缺口。**ETP（moe_tp>1）已验证**——SGLang
  `--tp-size 4 --ep-size 2`（4 rank 分 2 EP 组、组内 2 路 TP → moe_tp=2），MoE runner 打印 **`E=4,N=384`**
  （每 rank 4 完整专家、每专家 intermediate 768→384 减半），与前端 `expertShardDivisor({ep:2,moeTp:2})=ep×moe_tp=4`
  逐点一致。all-to-all 字节口径**与 moe_tp/传输后端解耦**（已由 Part 2b NCCL 直测）。**DeepEP**——后端在框架层生效并进入
  `forward_deepep→dispatch`，但预编译 kernel 与本机 CUDA 13/驱动 ABI 不匹配（`layout.cu:128 'named symbol not found'`）
  无法前向 → 字节直测留换环境（属环境/构建限制，非前端）。证据 [`evidence/parallelism/etp_deepep.md`](evidence/parallelism/etp_deepep.md)。
  仍未验证（**需换环境**）：真·多机（跨节点 NCCL / PD 分离）、可运行 DeepEP 构建下的字节直测。
- **状态（A100 全节点收尾，2026-09-19，spec nv2-nv3-a100-closure）**：① **TP=8**（Qwen3-0.6B）每卡权重 ×0.135
  （≈÷8）、KV 池 4,646,648 tokens=TP1 的 ×8.03，与前端 `min(tp,kv_heads=8)=8` 分片一致（tp==kv_heads 边界，min 截断
  逻辑成立）。② **EP=4/8**（真实 DeepSeek-V2-Lite 64 experts）：`--ep8`→`E=8,N=1408`（moe_tp=1）、`--ep4`→
  `E=16,N=704`（moe_tp=2），EP 切专家数(÷ep)、moe_tp 切 intermediate(÷moe_tp)、总 `expertShardDivisor=ep×moe_tp=8`
  与前端逐点一致。③ **通信 RS/AG**：NCCL 直测 reduce_scatter+all_gather 总线字节逐 N(2/4/8) 精确 == 前端
  `ringAllReduceBytes=2(N-1)/N·D`（比值 1.000）→ all-reduce 口径已涵盖 RS+AG 分解、无需新原语。④ **SGLang 第二架构
  profile**：V2-Lite（bcecmd 下载，bf16 MLA+MoE，A100 可前向；V4.1/V3 fp8 在 A100 跑不了用其代理）——MLA 单 latent
  KV=31,104 B/token=27层×(kv_lora_rank512+qk_rope64)×2B 精确（对比 Qwen GQA 112KiB/token），补齐 MLA 家族运行时证据。
  证据 `evidence/parallelism/tp8.md` / `evidence/parallelism/ep48.md` / `evidence/parallelism/allreduce_rs_ag.md` / `evidence/structure/runtime_profiles/sglang_v2lite.md`。
  诚实边界：V2-Lite 非 MSV 前端内置（unsupported），对的是 MLA latent 口径；真·多机仍需换环境。
- **状态（DeepEP 源码重编尝试，2026-09-19）**：试在 A100/CUDA13 从源码重编 DeepEP 修 `named symbol not found`。
  **未撞 sm_80 架构墙**（arch 可 patch 8.0、kernels 无 Hopper 专属指令），而是卡在**构建依赖链**：NVSHMEM 3.4.5 cu13
  头 `#include <cuda/std/tuple>` 缺 CCCL/libcu++ 头、`nvidia-cuda-cccl-cu13` wheel 构建失败（+ pip NVSHMEM 拆分链接布局、
  torch dlink dash `Bad substitution`）。根因：DeepEP+NVSHMEM 是 CUDA-12/Hopper 栈、本机 CUDA-13/Ampere 无官方适配。
  **DeepEP 验证留 Hopper（H20）匹配镜像**；all-to-all 字节口径已 NCCL 直测收口（增量非口径关键）。全程临时目录 + 符号链接，
  已清理、生产 deep_ep 2.1.0 不受影响。证据 [`evidence/parallelism/deepep_source_build_attempt.md`](evidence/parallelism/deepep_source_build_attempt.md)。
- **状态（运行时补维·线性注意力 hybrid，2026-09-19）**：SGLang 跑 Qwen3.8-Flash-Next（`Qwen4ExpForConditionalGeneration`、
  bf16、linear:full=3:1、512 experts）`--tp 8`——**qwen4_exp 首个真机 GPU 运行**（填算子成本"非原生 exotic arch 未跑整模型
  GPU 真值"）。观测 **hybrid 双 cache**：linear 层 `Mamba Cache`（conv 0.32 + ssm 16.14 GB/卡、1223 slots）+ full 层
  `KV Cache`（1,504,960 tokens）。前端支持 qwen4_exp（architecture-alias），双通道 `kvBytesPerToken=27648` +
  `stateBytesPerSequence=58.8MB` 均生效、定性一致；state 精确字节（前端 per-seq vs SGLang slot 化，同量级）留 caliber
  细化。**运行时架构维度覆盖扩到 3 类：GQA / MLA+MoE / 线性注意力 hybrid**。证据 `evidence/structure/runtime_profiles/sglang_qwen4exp_linear.md`。
- **状态（运行时补维·qwen3_5 builder，2026-09-19）**：修正——31 个内置 Qwen 用两个前端装配器，`assembleQwen3_5`
  覆盖 **29/31**（此前只验了 `assembleQwen4Exp` 的 2/31）。补跑 **Qwen3.5-4B**（`Qwen3_5ForConditionalGeneration`、
  bf16、linear:full=3:1）SGLang tp1，**server fired up**：Mamba cache（conv 0.59+ssm 25.08GB）+ KV（936,109 tokens）；
  **KV per-token 32,760 B ≈ 前端 32,768（0.02%）精确吻合**、state 同量级。**两个 Qwen 装配器（qwen3_5+qwen4_exp）
  运行时均已验证**。证据 `evidence/structure/runtime_profiles/sglang_qwen35_linear.md`。
- **状态（减层运行时·DSA builder + builder 覆盖审计，2026-09-19）**：代码级审计 60 模型×11 builder（一次性审计脚本，已移除；覆盖状态见
  `evidence/structure/builder_coverage.md`）——修正此前过度声明：运行时真机仅 qwen3_5/qwen4_exp，其余 builder 靠
  假设。**用减层补验 DSA**（`assembleDeepseekV32`，7 模型）：transformers 原生 `deepseek_v32` from_config 建 6 层/4.1B
  /bf16 随机 ckpt（去 fp8 quant），SGLang 起——**DSA backend + index_topk + KV cache 分配成功**，KV 7,704 B/token vs
  前端 8,448（比值 1.10，MLA latent+index 同结构）→ **DSA cache 口径 A100 补验通过**；但 **DSA 稀疏前向 kernel =
  SM90a/SM100f only**（A100 到 cache 分配为止，前向留 H20）。证据 `evidence/structure/runtime_profiles/sglang_dsa.md`。
- **状态（H20 补维·glm5_next DSA 稀疏前向端到端，2026-09-20，8×H20-3e/SM90/sglang dev）**：环境切到 H20，补上
  A100 到不了的 DSA 稀疏**前向**。对 `assembleGlm5Next`（GLM-5.3-Flash）减层 + `--load-format dummy`
  （`scripts/evidence/structure/glm5_next_reduce.py`，8 层 DSA[3,7]+KDA[0,1,2,4,5,6]、16 experts、去 fp8→bf16、
  保全部 per-head 维度）SGLang TP1：**`prefill=flashmla_sparse`/`decode=fa3`/KDA `TritonKDAKernel` 在 SM90 全跑通**，
  长 prompt 3001 tok（>index_topk 2048）触发稀疏 top-k、`Prefill batch #new-token 3001` 成功端到端出 token；
  **三 cache 元素口径对前端 `assembleGlm5Next` 逐点 0.0%**——KDA state 1,122,304（conv 73,728+temporal 1,048,576）、
  MLA latent 512、DSA index 128（`Glm5NextTextConfig`/`KimiLinearStateShape` 真值）。**逐字节挖出一处前端 bug**：
  `dsa_sparse_mla` 算子把 DSA index 键缓存按 bf16 计（256 B/层），实测 SGLang 存 fp8+E8M0 尺度（132 B/层）→
  glm5_next/deepseek_v32(9 模型) KV-per-token 高估 **+10.7%**（实测 2312 vs 前端 2560 B/token）；dsv4/v41 分支正确传了
  index dtype、此分支漏传（`ops/index.js:1146` vs `768-779`）。**glm5_next"稀疏前向 SM90+ 留 H20"收口**；
  同一 `flashmla_sparse` kernel 亦解除 `assembleDeepseekV32`(DSA) 前向硬件前置（其减层全前向可同法补，未跑仅登记）。
  边界：dummy 随机权重（验 kernel 路径 + cache 口径，非输出正确性）；全权重忠实前向未做。证据
  [`evidence/structure/runtime_profiles/sglang_glm5next.md`](evidence/structure/runtime_profiles/sglang_glm5next.md)。
- **状态（H20 逐字节 KV/state dtype 审计 · 挖出 2 个前端 bug，2026-09-20，本机 /ssd*/models）**：把 KV/state 拆到
  **dtype×元素**对前端 memory lens 逐 builder 扫。**Bug 1**：`dsa_sparse_mla` 的 DSA index 按 bf16 计（256 B/层），
  实为 fp8+尺度（132 B/层）→ glm5_next/deepseek_v32(9 模型) KV-per-token 高估 +10.7%（glm5_next 减层 H20 实测 2312 vs
  前端 2560 B/token）。**Bug 2（影响最大）**：线性 recurrent(ssm) state 前端按 bf16 计、实为 **fp32**（`mamba2_state_dtype`
  默认；qwen3_5 config 显式 float32）→ 线性 state **低估 ~1.934×(48.3%)**，波及 glm5_next/kimi_k3/qwen3_5/qwen4_exp
  （真实 GLM-5.3-Flash 34 KDA 层：140.78 vs 72.78 MiB/seq）。qwen3_5 减层 H20 真机（GDN + GQA + MoE）已确认 ssm 池；
  GQA KV(2047.9 vs 2048)/MLA latent/dsv4 逐 dtype 均一致（clean）。**观察 3**：W8A8C8 ckpt 强制 int8 KV，前端默认 bf16 未自动读
  `kv_cache_scheme`（可增强）。根因：memory lens 对架构强制 dtype（fp8 index/fp32 ssm）套用户统一 KV dtype，仅 dsv4 做了逐 dtype。
  **未在本机改**（无 node，改后须回归 + 重生成 golden）。证据 [`evidence/memory/cache_dtype_audit.md`](evidence/memory/cache_dtype_audit.md)。
- **状态（前端 2 bug 修复落地 + 双框架在机补验，2026-09-20）**：① **Bug1/Bug2 已在本地 `main` 修复**（commit `3e6aa3b` DSA index fp8`F8_E8M0S128`；`4cd9a88` 线性 recurrent fp32）——`node --test` 422/422、`verify:models` 60/60、golden 仅目标族变（DSA 9 模型 / 线性 34 模型）、`build` 通过。② **vLLM 首次真机**（A100 `vllm-0920`，Qwen3-0.6B GQA）：KV/token 114,712 B ≈ MSV 114,688（0.02%）、每卡权重÷tp、KV 池×tp——**vLLM == SGLang == MSV**（`evidence/parallelism/vllm_width_tp.md`）。③ **glm5_next DSA 稀疏前向复跑**（H20）：KV/token **2312** == 修复后 MSV（Bug1 收敛 2560→2312，0.0%）。④ **qwen3_5 GDN**（H20）：GQA KV 4096 == MSV；线性 state 512-slot 精确复测 conv 293,601/ssm 12,603,883/总量 12.30 MiB vs MSV 294,912/12,582,912/12.28 MiB **逐分量 <0.5% → Bug2（ssm fp32）逐字节精确验证，GDN state-shape 开项关闭**（初测 ~1.25× 为 4-slot 舍入伪差）。证据 `evidence/structure/runtime_profiles/sglang_qwen35_gdn_h20.md`。

## 算子成本 / per-stage roofline（cost）

- **触发判据**：UI 或对账需要 stage 级动作向量（当前 introspect 不产数值，是诚实缺项）。
- **为何需 GPU 运行时**：stage 级 roofline 的 I/O shape 与实际访存要真实 profiler 采样。
- **依赖**：GPU host + `perf-analysis` skill（vLLM/SGLang benchmark_serving 日志 → xlsx/图）；
  nsys/ncu 可选。
- **复现**：跑 benchmark_serving（或 nsys/ncu），采 per-stage 的 matrix/vector/sfu/bytes 与
  实测 TTFT/TPOT/吞吐；与 MSV 的理论 roofline（`cost/roofline.js`）对齐、标注偏差来源。
- **期望证据**：benchmark 日志 + `perf-analysis` 产出的 xlsx/对比图 + 采样命令回显。
- **判定**：stage 级动作向量非 null 或显式 missing；实测与理论偏差有归因。
- **状态**：**部分已验证（2026-09-17，A100-SXM4-80GB / SGLang / Qwen3-0.6B / TP=1）**。用户指定仅跑
  SGLang + Qwen3-0.6B。`sglang.benchmark.serving`（100 req × in512/out128）实测：时长 1.55s、输出吞吐
  8276 tok/s、Mean TTFT 584ms、Mean TPOT 7.03ms、并发 95.5。按 MSV roofline 口径（`cost/rates.js`
  matrixPerSecond=peak·η/2、bytesPerSecond=bw·η_hbm；`roofline.js` 五路取 max）对 A100 算得：prefill
  算力受限（单序列 2.34ms、聚合 234ms）、decode 访存受限（B=100 每步 4.25ms），聚合地板 778ms。
  对照：每项实测延迟均落在对应地板**之上**——端到端 1.99×、TPOT 1.65×、TTFT 2.5×，bound 分类正确，
  量级正确 → 坐实 MSV roofline 是物理成立的**下界**（overlap 上界），偏差归因于 roofline 明确不建模的
  调度/排队/kernel 启动/非满带宽/context 增长。stage 级动作向量仍是诚实缺项（introspect 不产数值），
  本次以聚合 roofline 验证口径正确性。证据 [`evidence/cost/bench_vs_roofline.md`](evidence/cost/bench_vs_roofline.md)
  （原始 `roofline_output.txt` / `bench_serving.log` 由 `scripts/evidence/cost/roofline.py` 重生）。
- **状态（深化 · 逐算子计算量/访存量真值，2026-09-17，A100/Qwen3-0.6B）**：把验证从"整模型聚合"下沉到
  **每个原子/融合算子**，三通道各选 oracle。**matrix（计算量）**：全部线性/投影算子
  `MSV MACs×2 == torch FlopCounterMode FLOPs` **逐位相等**（prefill 610,288,009,216 / decode 1,191,968,768）；
  注意力 MSV 因果口径 = torch 全方阵公式 0.501×，与 flash kernel 只算下三角的实际一致（torch 公式高估）。
  **bytes（访存量）**：4 个 GEMM 的 MSV compulsory 读(weights+actIn) == ncu DRAM 读，误差 0.2–0.4%；写侧
  常驻 L2 → MSV total 为 DRAM 保守上界；逐元素算子 MSV 建融合 compulsory（朴素多 kernel 微基准重复物化
  中间量而高估，非代表）。**bound**：GEMM 算术强度 340–438 FLOP/B（compute-bound）、norm/rope/swiglu
  0.2–0.33（memory-bound），与 `cost/roofline.js:classifyRoofline` 逐项一致。**未发现前端公式错误**，
  合法口径差（因果 vs 全方阵、写侧 cache、融合 compulsory）登记进 `cost_counts.md`。回归全绿
  （pytest / node --test 410 / verify:models 60/60 / docs:check）。证据
  [`evidence/cost/operator_cost.md`](evidence/cost/operator_cost.md)（前端 dump + FlopCounter
  + ncu 微基准，原始 `reconcile.json` 由 `scripts/evidence/cost/operator_reconcile.py` 重生）；SDD 存档 `.comate/specs/nv3-operator-cost-truth/`。
- **状态（MoE/MLA 算子真值，2026-09-18，减层随机权重零下载）**：减层 `from_config`（HF_HUB_OFFLINE）建
  3.02B DeepSeek-V3（MLA+MoE）随机权重，前端 vs FlopCounter(eager)：标准 GEMM（lm_head/MLA 压缩 q_a·kv_a/
  dense MLP/shared/router）一致 ~3%、MLA 压缩投影**精确相等**；**routed experts FlopCounter(HF eager MoE)
  漏计**（非有效 oracle，前端 fused_moe_mlp 教科书口径正确，真值需 ncu）；MLA 注意力 bmm=0.30×（latent/吸收式
  ≠标准 MHA，口径差异）。未改前端公式，MLA q_b/kv_b 残余(3%)+bmm 口径+routed oracle 登记为待细化。
  DeepSeek-V4(DSA) 已能同法减层构建(1.73B)。证据 [`evidence/cost/operator_cost_moe.md`](evidence/cost/operator_cost_moe.md)。
- **状态（MoE/MLA ncu 补齐，2026-09-18）**：闭合上条两个待细化项。**matrix**：孤立 GEMM FlopCounter 证
  MoE routed active(gate+up+down)=45,097,156,608 FLOP、MLA q_a/kv_a 压缩**与前端 MACs×2 逐位相等**——前端
  `fused_moe_mlp`（tokens×topk×3×h×moe_inter）与 MLA 压缩公式**精确**，Part1 的 routed 漏计是 HF eager loop
  工具局限非前端错。**bytes**：ncu DRAM 读 vs MSV compulsory——moe_gate 1.0006(精确)、moe_down 1.031、
  MLA 压缩 1.12–1.13（小 GEMM 固定开销，同 roofline 尺寸扫描），**MSV 为有效下界**。仍未做：MLA 注意力
  bmm 0.30× latent 口径细化、MoE 聚合 all-expert 常驻权重字节。证据 `evidence/cost/operator_cost_moe_ncu.md`。
- **状态（MLA bmm 0.30× 归因，2026-09-18）**：**结案——减层配置 artifact，非前端错误**。根因：transformers
  `to_json_file` 给减层配置注入 `head_dim=64`，前端 `attentionHeadDim` 让 head_dim 优先于 qk_nope+qk_rope
  → MLA 打分宽度误用 64 而非 192。**去掉注入的 head_dim（=原始 catalog 无 head_dim 的口径）后，MLA sdpa matrix
  与 eager torch 在纯因果 0.504× 一致，与标准 MHA 同口径**。60 catalog 模型均无 head_dim 冲突（不触发），
  不改前端优先级（避免为假设修复破坏其它模型）；减层 harness 须剥离注入的 head_dim。证据
  `evidence/cost/mla_bmm_caliber.md`。至此 A-2 MoE/MLA 算子级 matrix+bytes 通道全部对齐真值。
- **状态（算子族全覆盖，2026-09-18）**：前端 cost 注册表**全部算子族**经**原子分解链**验到真值——
  ① 19 原子精确对 aten（`formulas/__tests__/` 66 pass）；② 每 fused 算子 = Σ 原子（`fused==decompose`
  恒等式 + `identities.test.js` 绿）；③ 关键原子 aten↔硬件 GPU 实测（matmul FlopCounter 逐位相等 + ncu
  DRAM 下界、conv1d fixture、attention bmm 因果、elementwise ncu 量级）。exotic 族（DSA indexer/sparse、
  mHC、hyper/ple、linear-attn/GDN/KDA、engram、gates、vision）均分解到这 19 原子，无新原语——传递式覆盖。
  deepseek_v4(native) 已实测 emit mHC/qsa 族并有动作向量。未发现前端公式错误。诚实边界：非原生 exotic
  arch(qwen4_exp/glm5_next/kimi_k3/deepseek_v41) 未跑整模型 GPU 真值（需框架支持），其算子族由 identity+atoms
  覆盖。证据 [`evidence/structure/operator_coverage.md`](evidence/structure/operator_coverage.md)。
- **状态（stage 级动作向量尝试，2026-09-19，spec nv2-nv3-a100-closure）**：hooks+CUDA events 按 stage
  （attn/mlp/norm）计 Qwen3-0.6B prefill 时间 + FlopCounter 整前向。**总 FLOPs 670,417,616,896 与前端聚合一致**。
  **结论（据实，不强凑）**：① stage 级 **matrix（FLOPs）可导且对账**——per-算子 MACs×2 按 stage 聚合即得，数据不缺；
  ② stage 级 **TIME 在 0.6B 尺度被 kernel 启动开销主导**（attn 77% 时间但非 FLOPs 主项，子核多而小 → launch-bound，
  非 compute-bound），**不匹配 roofline 算力划分**——与聚合算子成本（实测 1.99× 地板、overhead 不建模）同源，roofline 是
  stage 级下界；③ "introspect 不产 stage 级动作向量"实为**前端 introspect 未暴露 stage rollup（UI/接口面）**、非测量缺失。
  证据 [`evidence/cost/stage_vectors.md`](evidence/cost/stage_vectors.md)（原始 `stage_vectors.json` 由
  `scripts/evidence/cost/stage_vectors.py` 重生）。

## A2 kernel 口径对齐（cost）

- **触发判据**：需要 kernel 级口径，而非现在的理论上限口径。
- **为何需 GPU 运行时**：MSV 的 A2 假设 scores/probs 中间量按理论上限（4×）计，flash kernel 下融合单遍、
  不落 HBM；实测口径需在 GPU 上跑 flash-attention kernel 取证。
- **依赖**：GPU host + flash-attn / 框架 attention backend；ncu 采 kernel 级访存。
- **复现**：对代表模型（含 MLA / DSA / dsv4 稀疏）用 ncu 采 attention kernel 的实际 scores/probs
  访存与 FLOPs，与 MSV `sdpa_attention` / `dsv4_*` 的理论口径对比。
- **期望证据**：ncu 报告 + kernel 访存/FLOPs 数值 + 对比表。
- **判定**：确认保持理论口径，或给出 kernel 级修正口径（需另行对齐、更新 `cost_counts.md` A2 假设）。
- **状态**：**已验证（2026-09-17，A100 + ncu）**。ncu 采 `pytorch_flash::flash_fwd_kernel`（bf16 causal）
  DRAM 字节：S=4096 读+写≈192 MiB、S=8192≈483 MiB，与 Q/K/V/O（O(H·S·D)）同阶，比 A2 causal 4× 物化上限
  （2048 / 8192 MiB）小约 10.7×→17×，且随 S 增大（实测∝S，A2-4×∝S²）——**N×N scores/probs 不落 HBM，
  实证坐实 A2 物化口径为保守上界**。处置：保持理论口径（MSV 本就标"估计/上界"），已在 `cost_counts.md` A2
  假设旁标注 kernel 级实证。证据 [`evidence/cost/flash_kernel_caliber.md`](evidence/cost/flash_kernel_caliber.md)（探针 `scripts/evidence/cost/flash_kernel_attn.py`）。

## DeepSeek-V4.1-Flash 结构 / KV 实证（structure / memory）

> 本次接入（commit `b2ee018`）已通过 header-truth 逻辑元素恒等式（ratio=0.9998）、
> `verify:models` 60/60、真实 Chrome 全量扫描。以下是**需真实 checkpoint / GPU 才能进一步收口**
> 的保真边界，非阻塞项，逐条登记：

- **权重逐张量恒等式**：现按 `header-truth.json`（manual，`parameterTotal=508,182,659,298`、
  `tensor_count=96085`）的 dtype 逻辑元素对账。缺 `model.safetensors.index.json`（.gitignore 不入库），
  未做逐张量路径核对。触发：拿到 index/checkpoint 后，逐张量 shape vs 结构声明对账。
- **compressor / indexer 层位**：现复用 V4 的 `compress_ratio>1` 启发式摆放 compressor，
  **未**按 config 的 `kv_source_layer_ids=[2,8,14,20]` / `index_source_layer_ids=[2,8,14,20,24,28,32,36]` /
  `candidate_source_layer_id` 精确摆放（与既有 V4 保真度一致）。触发：真实模块树对账（见结构对账节）暴露层位差异。
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
  [`evidence/structure/deepseek_v41_config_index.md`](evidence/structure/deepseek_v41_config_index.md)。
  仍待真实权重/GPU 推理：engram/DSpark 运行时（接受率/显存/吞吐）、后端 transformers 构造（需
  `deepseek_v41` 框架支持或补齐 remote code）。
- **状态（CSA2 注意力修复，2026-09-18）**：联网确认 V4.1 用 **CSA2**（Full/Reindex/Reuse 跨层共享 KV/索引）
  ≠ V4 的 CSA。发现并修复真实 bug——前端 `deepseek_v41` 逐字复用 V4 组网、`ops/index.js` 注意力类型硬编码
  `ratio===4→sparse/===128→HCA/else→SWA`，导致 V4.1 的 **ratio=2 层（41/43）被误判为滑窗 MQA** 且 indexer
  悬挂。引入 `isSparse=ratio>1&&!==128` 后 ratio=2 正确归为 `dsv4_sparse_mla`；V4(0/4/128) 算子分布逐项不变、
  ops-spec-tree golden 仅 V4.1 一个 hash 变、回归全绿。**仍未收口（需 CSA2 规范/框架）**：CSA2 跨层 KV 共享
  （Reuse 层复用 source KV，前端仍每层计 resident KV → KV 字节偏高，未体现 890 B/token）、ratio=2 压缩 KV
  精确宽度。证据 [`evidence/structure/deepseek_v41_csa2_attention_fix.md`](evidence/structure/deepseek_v41_csa2_attention_fix.md)。
  **后续验证前提**（见证据文件"后续验证前提"表）：CSA2 KV 共享字节/宽度收口**不缺权重、缺框架**——一旦
  `deepseek_v41` 有可运行框架（或补齐 remote code），减层随机 checkpoint 即可对 shape/字节真值；仅
  engram/DSpark **运行时行为**（接受率/显存/吞吐）必须完整训练权重。前端折叠签名修复为纯前端、无外部依赖。
- **状态（框架已合入，2026-09-18 联网确认代码）**：**"缺框架"前置已解除**——vLLM `registry.py`(main) 已含
  `"DeepseekV41ForCausalLM": ("vllm.models.deepseek_v41", ...)`，keystone PR vllm#56214 已 merge（+config/tokenizer，
  稀疏 indexer #56254 / 注意力 megakernel #56344），追踪 issue #56400 明确 **V4.1 与 V4 是不同架构**（独立 tree/tokenizer/
  parser/config）；SGLang 亦 day-0（"compressed KV shared"）。**影响**：① 印证前端 `deepseek_v41.js` 逐字复用 V4 组网是
  反模式（后续应拆开，非仅 isSparse）；② CSA2 跨层 KV 共享（890 B/token）验证**已解锁、转可执行**——减层随机 ckpt 实例化
  即可对字节真值，无需完整权重；③ engram/DSpark 运行时仍需完整权重 + GPU。详见证据文件"框架状态更新"节。
- **状态（V4 代理对账 + v41 骨架，2026-09-18）**：V4.1 无框架的过渡期，用 transformers 5.17 原生 `deepseek_v4`
  减层随机实例化（零下载）对账前端 dsv4 算子——ratio→类型映射取自 transformers 自带 `_COMPRESS_RATIO_TO_LAYER_TYPE`
  `{0:sliding,4:CSA,128:HCA}`，V4-Flash 43 主层 sliding=2/CSA=21/HCA=20 与前端 `dsv4_swa/sparse_mla/compressed`
  **逐点一致**（swa 差 1 = MTP 层 ratios[43]=0，非 bug），indexer=21 仅落 CSA 层亦一致；减层 `from_config` 逐层
  `self_attn.layer_type` == 映射（ok）。据此把 `deepseek_v41.js` 从**逐字 re-export V4** 改为独立 `assembleDeepseekV41`
  入口（去反模式、标注 CSA2 分叉 TODO，输出不变、golden 全绿）。证据 [`evidence/structure/deepseek_v4_proxy.md`](evidence/structure/deepseek_v4_proxy.md)。
  **V4 映射不含 ratio=2** → 再次印证 CSA2 需独立 `deepseek_v41`；890 B/token 精确口径仍留框架（换环境/升级后验）。
- **状态（真实 checkpoint + 参考栈 A100 收口，2026-09-18）**：`$MODELS/DeepSeek/DeepSeek-V4.1-Flash`
  到位（48 分片 fp8+fp4 权重 + 官方 `inference/` 参考栈 + tech report + `assets/dsv41_kv_cache.png`）——此前
  "缺权重/框架/规范"三前置**全部解除**。① **CSA2 KV-share shape 真值化**：参考栈减层小模型只读探针
  （`.venv`/torch2.14+cu130/tilelang0.1.8，`apache-tvm-ffi` 降 0.1.8.post2 修 py3.12 反射冲突）实测
  compress_kv_cache 仅 kv_source 层、index k_cache 仅 index_source 层、**Reuse 层 0 常驻**
  （`csa2_kv_share_ok/indexer_source_ok=true`）。② **前端修复 CSA2 跨层过计数**：`decoderStack.js` 折叠签名加
  `:kvsrc/:kvreuse/:idxsrc`（source 与 Reuse 不再折叠合并）+ `ops/index.js` `cacheResidentDecl` 按
  emitCompressor/emitIndexer 逐层门控 → V4.1 每 token KV 41216→**23936 elem**（Reuse 层压缩 KV/index 归 0），
  **V3.2/V4-Flash 及其余 59 模型逐字节不变、ops-spec-tree golden 仅 V4.1 一个 hash 变**、`node --test` 410/410、
  `verify:models` 60/60、`docs:check` 全绿。③ **结构对账 第 60 模型减层对账**：前端 compressor/indexer 层位 5/5 与
  参考栈真值一致。④ **Tier3 行为验证受 A100 硬件边界阻塞**：参考栈 fp8 GEMM 用 SM89 fp8 MMA
  （`SM89_16x8x32_F32E4M3E4M3F32_TN`），A100=SM80 无 fp8/fp4 张量核 → device-assert，**减层不可绕过**，
  engram/DSpark 接受率/真实 KV footprint 留 Hopper/Ada 重跑（fp8 量化 kernel 本身在 A100 可跑）。⑤ **残留口径专项**：
  绝对 890 B/token（边际口径 + 逐 dtype 字节，属模型级改动超"仅 V4.1 变"边界）、ratio=1 source 层常驻压缩 KV
  （滑窗分支未计）。证据：[`evidence/structure/deepseek_v41_module_tree.md`](evidence/structure/deepseek_v41_module_tree.md) /
  [`evidence/memory/deepseek_v41_csa2_kv_bytes.md`](evidence/memory/deepseek_v41_csa2_kv_bytes.md) /
  [`evidence/structure/deepseek_v41_engram_dspark.md`](evidence/structure/deepseek_v41_engram_dspark.md)（原始 `v41_kv_shapes_reduced.json` 由 `scripts/evidence/memory/deepseek_v41_kv_shapes.py` 重生）。
- **状态（KV 边际字节口径 + ratio=1 归类，2026-09-18）**：spec `dsv4-kv-marginal-bytes` 落地——dsv4 家族
  KV-per-token 改为**边际（排除有界滑窗）+ 逐 dtype（V4.1=fp4/V4=fp8）**，V4.1 ratio=1 层归入 sparse_mla
  （Full 模式全长压缩 KV 计入、dangling compressor/indexer 消除）。对官方图：**V4-Flash 3,440 vs 3,514（−2.1%）**、
  **V4.1 1,056 vs 890（+18.7%，残差如实登记、未强凑）**。blast radius 仅 dsv4 家族 6 模型（golden 重生），非 dsv4
  逐字节不变；`node --test` 410/410、`verify:models` 60/60、W5 capacity↔kvRead 恒等式保持、`docs:check` 全绿。
  **后续（用户认领）**：在 **H20（Hopper SM90，fp8 解除 A100 边界；fp4 expert 仍 Blackwell-only 需回退）** 复跑
  完整推理，用真实逐层 KV footprint 二次校准 V4.1 +18.7% 残差 + 取 engram/DSpark 运行时。详见
  `evidence/structure/deepseek_v41_engram_dspark.md`「后续：H20 复跑计划」。
- **状态（A100 静态收尾，2026-09-18）**：spec `dsv41-a100-static-closure` 落地——① **逐张量权重恒等式对账**
  （`index.json` 96,085 张量 name + 分片头部 shape）：compressor/indexer 查询/engram/experts/MTP/vision 层位与
  shape 逐点一致；**关键发现 `indexer.wk`（index 键 k_cache）只在 owns_k=[2,8,14,20]（非全 8 index_source）**。
  ② **V4.1 KV 残差静态收口**：据①把 index 常驻门控到 `emitCompressor && emitIndexer` + fp4 含 scale 摊销
  （压缩 KV=E4M3/16=0.5625、index=E8M0/32=0.53125）→ V4.1 **精确命中 890 B/token（0.0%）**、V4-Flash 3,440
  （−2.1% 未变、如实留残差），本轮 golden 仅 V4.1 变、非 dsv4 逐字节不变。③ **全模块树对账**：backbone
  Attention/MoE/Gate/Expert/Compressor/Indexer/Block/MHC 与前端节点类型逐类一致，engram/DSpark/vision 经①覆盖，
  注意力之外无结构缺口。`node --test` 410/410、`verify:models` 60/60、W5 恒等式保持、`docs:check` 全绿。证据
  `evidence/structure/deepseek_v41_tensor_identity.md`（原始 `v41_module_tree_reduced.json` 由 `scripts/evidence/structure/deepseek_v41_module_tree.py` 重生）。
- **状态（V4-Flash KV 残差归因 + A100 封板，2026-09-19）**：从 BOS 下 V4-Flash index+config（纯 I/O）逐张量核——
  `compressor.wkv` 全 41 个 ratio>0 层、`indexer.wq_b` 21 个 ratio==4 层与前端 `emitCompressor=ratio>1`/
  `emitIndexer=ratio===4` 逐层一致；**V4-Flash 无 `indexer.wk`（无独立 index 键 cache）、无 engram**（均与建模吻合）。
  −2.1%（3,440 vs 3,514）经归因为**小 fp8-cache scale/rope 尾口径残差、非结构错误**，方向为"加 ~74B"；精确拆分需
  V4-Flash 参考推理栈（本地无，仅 V4.1 有）→ **据实留残差、不为 2% 猜测改前端**（与 V4.1 有参考栈→精确 890 对照）。
  证据 `evidence/structure/deepseek_v4flash_tensor.md`。**A100 侧封板**：静态/框架/结构级验证（结构/成本/kernel 各项 + V4.1 结构/KV 实证 静态
  + 框架/并行 profile / 算子成本 并行·通信·算子）已收满；剩余项（V4.1/V3 fp8 前向、engram/DSpark 运行时、DeepEP、真·多机、V4-Flash
  精确残差）**均需 H20/Hopper 或换环境**。

## 后端生产化 · 部署硬化（backend，详见 backend_audit.md）

- **触发判据**：真正对外部署 `backend/app.py`。
- **为何登记在此**：属**运行时/部署**环境项（非 GPU 计算），与上面 GPU 项同为“出了 MSV 静态仓库才能验”，
  一并收口。
- **依赖**：部署环境；不需 GPU 计算。
- **复现/清单**：路径约束（禁越权读）、remote code 沙箱（`trust_remote_code` 隔离）、鉴权、限流、
  日志脱敏；上线前逐项过。
- **期望证据**：部署配置审阅记录 + 安全项逐条勾验。
- **判定**：五项硬化全部落实。
- **状态**：**已审计（2026-09-17，只读）；硬化待部署触发**。现状：路径约束 / remote code 沙箱 / 鉴权
  三项缺失（P0），限流·超时·大小·CORS 与日志脱敏两项部分——与 README「本地/可信内网」定位一致，属设计如此。
  强制硬化会破坏本地开发与 结构对账 / V4.1 实证 verify（依赖 `trust_remote_code=True` + 任意 config 路径），故应做成
  production/untrusted **opt-in 开关（默认关）**。逐项证据与最小加固点见
  [`backend_audit.md`](backend_audit.md)。
  触发（真正对外部署）到达前不落地硬化实现。

---

## PD 分离（Prefill-Decode disaggregation）（parallelism）

- **目标**：验证 MSV 的 PD 分离建模（`comm.js pdKvTransferBytes`、`parallel.js
  projectPdFit/validatePdPlan`）——KV 传输字节口径、布局重排判据。
- **依赖**：≥2 GPU；SGLang disaggregation + mooncake_tcp。
- **状态**：**机制 + KV 传输字节口径已 A100 单机双卡真机验证（2026-09-19）**。Qwen3-0.6B
  prefill(GPU0)+decode(GPU1)+mini_lb router，KV 经 mooncake TCP 逐层传输、输出正确；`transfer_layer_num=28`、
  每 token 114,688 B（`kv_heads·head_dim·层·dtype·2`）与 MSV `pdKvTransferBytes` **0.0%**；tp1→tp1 无
  `layoutRepack` 与 MSV 判据一致。**未覆盖**：跨 TP 布局重排（prefill_tp≠decode_tp）、inter-node RDMA 实测带宽
  （本次 TCP localhost）——属多机/跨拓扑。证据 [`evidence/parallelism/pd_disaggregation.md`](evidence/parallelism/pd_disaggregation.md)。

## 计算量 / 访存量 / 通信量 / Roofline —— 已验状态（汇总）

- **计算量(FLOPs)**：算子成本 `evidence/cost/operator_cost.md` —— GEMM `MSV MACs×2 == torch FlopCounterMode` 逐位相等；
  MoE `fused_moe_mlp`、MLA 压缩投影与 FlopCounter 逐位相等；注意力因果口径合法。**已验(实测)**。
- **访存量(HBM bytes)**：算子成本 / kernel 口径 —— GEMM 读侧 vs ncu DRAM ±0.4%；A2 flash scores 不落 HBM；MoE 权重读 ncu≈1.0；
  线性 state cache 0.02%。**已验(ncu 实测)**。
- **通信量**：框架/并行 profile —— all-reduce N=2/4/8 busbw×dt 比值 1.000；EP ownership 逐点；all-to-all 16.78 MB 字节直测+nsys；
  RS/AG=all-reduce 分解。**已验(NCCL+nsys 实测)**。缺 DeepEP(A100 不可)、真·多机。
- **Roofline**：算子成本 `evidence/cost/bench_vs_roofline.md` —— prefill=compute-bound / decode=memory-bound 分类一致；实测≥地板
  (1.65–2.5×含调度开销)；算子 AI bound 逐项对齐。**已验(SGLang bench)**。缺 inter-node 费率。
- **量化权重打包(FP8/GPTQ-Int4/MXFP8)**：结构对账 `evidence/structure/quant_packing.md` —— 对 6 个真实量化 checkpoint
  header 逐模型：GPTQ-Int4 0.9994、FP8 0.9998–0.9999、MXFP8 0.9995、bf16 基线 0.9999。GPTQ 的 I32×8 packed +
  F16 scales/qzeros、FP8 E4M3 1B、MXFP8 e8m0 尺度口径全部对齐。**已验(真实 header)**。运行时 serve 只重复 footprint。
- **MTP/投机头**：结构（fc+decoder+norm+shared_head，draft resident-only）+ `mtp.{i}` 张量计数随参数量对真实
  header 收敛（0.999）。**结构/张量口径已验**；运行时投机接受率循环须真实 MTP 权重（随机 reduced 不产出 mtp.*）。

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
