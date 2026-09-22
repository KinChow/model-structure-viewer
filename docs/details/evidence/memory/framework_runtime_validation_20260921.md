# Framework runtime validation —— vLLM / SGLang / MSV accounting（2026-09-21 UTC）

容器日志为 `2026-09-21 19:xx`；宿主 `date -Iseconds` 记录为
`2026-09-22T03:xx:xx+08:00`，是同一执行时段的时区差异。文档/工件目录按 UTC
日期归档，远端目录名称保留实际创建的 `msv-validation-20260922`，不改写取证路径。

**结论：10 个最终配置的功能 smoke、20 次请求通过；不是 60 模型 GPU 全覆盖，
也不是总驻留显存估算全面准出。** 分项中 state dtype/shape、MTP KV、TP/EP 语义通过。
历史运行暴露出的 vLLM DSA k-pool 线性增长已按 profile 规则修复；SGLang 投机
state 已支持显式 workload 的理论公式，但不同版本的物理预分配、页保留和 backend
workspace 仍不属于 Graph IR 可证明范围。

> 本轮是功能与 cache/accounting 取证，不是吞吐 benchmark。最终矩阵中的 10 个配置均监听 loopback，使用短请求或 dummy/reduced 权重，
> 只停止本轮启动的进程；A100 上用户确认的旧 `sglang-20260918-20518d85` 服务已获授权停止。
> 原始证据归档见 `artifacts/framework-runtime-validation/20260921/`，压缩包 SHA-256 见该目录文件名对应的本地校验记录。

## 环境边界

| 主机 | GPU / 驱动 | 容器与版本 | checkout |
|---|---|---|---|
| H20 `10.98.95.16` | 8× H20-3e / CUDA 13.0 | `vllm-0920`: vLLM `0.29.1rc1.dev397+ga8d1aa9c9`；`sglang-dev-20260918-20518d85`: SGLang `0.0.0.dev1+g20518d851`；`dsv41_zzj_deploy`: SGLang `0.0.0.dev0` | `/ssd2/zhouzijian01/model-structure-viewer` 为旧 `5043e98` 且有既有未提交改动；本轮未修改该 checkout |
| A100 `10.55.87.81` | 8× A100-SXM4-80GB / driver `575.57.08` | `vllm-0920`: vLLM `0.28.1rc1.dev278+g73029d424`；`sglang-20260918-20518d85`: SGLang `0.0.0.dev1+g20518d851` | 宿主无可用 git；既有 MSV checkout 的 SHA 未核验，本轮未修改 |

H20 的三张指定容器均存在；A100 实际 SGLang 容器名为
`sglang-20260918-20518d85`。A100 原有 GPU 0/1 服务先确认后按用户授权停止，
本轮使用 GPU 2–6；最终本轮 GPU 显存均回到 0 MiB。H20 的
`dsv41_zzj_deploy` 保持运行状态但其本轮服务进程已清理。

理论对账使用本地 `681f4c7`，同版本 `git archive` 快照传到两机独立目录
`/ssd2/zhouzijian01/msv-validation-20260922/snapshot`，SHA-256 为
`8f6b60b4096069024175a81a5d469138bb775c61f635f6c82d9177238b7e67da`。
宿主 Node 不能运行（H20 无 Node，A100 的 Node 遇到宿主 glibc 不兼容），
因此实际 config 回收后在本地同版本 oracle 执行，不声称远端运行过 MSV Node。

### 功能矩阵

| 主机 | 框架/模型/配置 | 权重 | 请求数 | 最终结果 |
|---|---|---|---:|---|
| A100 | vLLM V2-Lite TP4 | 真实 | 2 | 通过 |
| A100 | vLLM V2-Lite TP4+EP4 | 真实 | 2 | 通过 |
| A100 | SGLang Qwen3.5-4B 无投机 | 真实 | 2 | 通过 |
| A100 | vLLM Qwen3.5-4B MTP=2 | 真实 | 2 | 通过 |
| A100 | SGLang Qwen3.5-4B NEXTN/EAGLE=2 | 真实 | 2 | 通过 |
| H20 | vLLM Qwen3.5 reduced GDN | dummy | 2 | 功能通过，不判语义精度 |
| H20 | SGLang Qwen3.5 reduced GDN | dummy | 2 | 功能通过，不判语义精度 |
| H20 | vLLM GLM5-Next reduced DSA/KDA | dummy | 2 | 功能通过，不判语义精度 |
| H20 | SGLang GLM5-Next reduced 自动 backend | dummy | 2 | 功能通过，含 5401-token prefill |
| H20 | SGLang V4.1 FP8 代理 + DSpark TP8/EP8 | 真实 | 2 | 通过，有投机统计 |

不计入上述最终矩阵：初次 GLM 强制 fa3 失败、EP4 首次遇到端口 TIME_WAIT
在启动前被 harness 拒绝、以及早期 EP2 的一条探路请求。均未篡改为通过结果。

## 通过项

### 1. vLLM A100 TP-only / EP：DeepSeek-V2-Lite

使用真实权重 `/ssd2/models/DeepSeek/DeepSeek-V2-Lite`、vLLM `0.28.1rc1.dev278`，
各运行两次短 completion：

- TP4（GPU 2–5）通过，真实生成返回非空文本；每 rank 日志显示 `E=64, N=352`。
- TP4 + `--enable-expert-parallel`（GPU 2–5）通过，真实生成返回非空文本；日志显示
  `Local/global number of experts: 16/64`，即 `E=16, N=1408`。
- vLLM 日志的 KV spec、模型权重加载和 MoE placement 与 MSV 计划语义一致：
  EP-off 是所有专家、intermediate 按 TP 切；EP-on 是专家按 EP 切、expert TP=1。
- MSV 的 vLLM profile 没有给未开启 EP 的 TP-only MoE 注入 `moeTp=1`；显式 EP 才使用
  effective `EP=TP×DP, moeTp=1`。这一点与两种真实 placement 对齐。

V2-Lite 不是 MSV 支持的内置架构，**此项是共享并行/MLA 公式的代表性验证，
不是 V2-Lite 整图通过**。runtime MLA = `27×576×2 = 31,104 B/token`；
TP4 每 rank unique cache bytes 为 `21,602,101,248`，除以 `43,407×16`
tokens 恰为 `31,104`，无需从两位小数日志反推。TP4/EP4 每 rank 实际
named parameter bytes 均为 `7,934,630,912`，两种布局等量但形状不同。

A100 原有 vLLM 容器保留运行，是宿主已有容器/agent；本轮验证进程已经退出，GPU 已清零。

### 2. vLLM / SGLang A100：Qwen3.5 GDN state dtype 与 MTP

使用真实 Qwen3.5-4B：

- vLLM 单卡 MTP 真实生成通过，runtime capture 的 KV group 为
  `conv [5,8192] BF16` + `temporal [32,128,128] FP32`；其中 5 是
  基础 conv 窗口 3 加 2 个 speculative slots。安装版本的 `config.py`
  明确读取 config `mamba_ssm_dtype`。
- SGLang 单卡真实生成通过，日志与 capture 为相同的 BF16 conv + FP32 temporal；
  Mamba cache 的固定大小和请求池均成功初始化。
- vLLM MTP `num_speculative_tokens=2` 真实生成通过，日志解析为
  `Qwen3_5MTP`，runtime capture 的 KV group 包含 `mtp.layers.0.self_attn.attn`，
  且配置标识 `num_speculative_blocks=2`。
- SGLang generic EAGLE/MTP 配置真实生成通过；capture 单独记录
  `Qwen3_5ForCausalLMMTP` draft model、独立 MHA draft KV pool，以及 speculative
  Mamba scratch。SGLang 日志同时报告 target KV、draft KV、intermediate SSM / conv
  window allocation。

注意：reduced/dummy 的输出只证明 kernel、cache 初始化和请求链路可用，不能作为模型精度结论。

### 3. H20 SGLang：DSA / KDA / GDN cache dtype

使用 H20 `sglang-dev-20260918-20518d85`：

- Qwen3.5 reduced dummy：SGLang GDN 前向与真实请求通过；capture 为
  `conv [6,33,8192,3] BF16`、`temporal [6,33,32,128,128] FP32`。
- GLM5-Next reduced dummy：显式 `fa3` backend 在 zero-rope reduced 配置上触发
  SGLang reshape 边界错误（非 MSV 公式错误）；改用自动 backend 重跑，DSA + KDA 请求通过，
  长 prompt 约 5401 tokens 也通过。capture 的 DSA index buffer 为 `uint8`，带有
  `uint8` index storage；KDA temporal 为 FP32。
- vLLM H20 对 Qwen3.5 / GLM5-Next 的真实 reduced dummy 请求也通过；vLLM KV spec 分别
  报出 GDN/KDA state dtype 和 DSA indexer `uint8`。

因此 MSV 当前的两个关键 profile 规则得到运行时方向确认：

```text
KDA/GDN temporal state: FP32（当前两框架/配置路径）
DSA index: uint8 / FP8 index storage，不是通用 BF16 fallback
```

### 4. H20 SGLang DSV4.1-Flash + DSpark

使用用户指定的 `dsv41_zzj_deploy`、真实代理权重
`/ssd1/models/DeepSeek-V4.1-Flash-Attn-W8A8-MoE-W4A8-INT8-Dynamic`，参数包括
`--tp-size 8 --ep-size 8 --attention-backend dsv4 --kv-cache-dtype fp8_e4m3`
和 `DSPARK`：

- 服务 ready，真实请求两次成功；返回非空文本。
- 运行时 `spec_accept_rate`、`spec_num_proposed_drafts`、`spec_accepted_drafts`
  等字段出现，证明请求确实经过 DSpark speculative path，而不是普通 forward fallback。
- 日志报告 target `DeepSeekV4TokenToKVPool`：`c4_size=3579072`、`c128_size=111846`；
  draft pool 的 `c4_size=0`、`c128_size=0`，两者都有 SWA region。
- capture 的 target/draft SWA tensor storage pointer 不同，说明这两个运行时池是独立存储；
  不能因为 full-to-SWA 映射或 SWA window 相同就把 draft pool 计为 shared。
- 运行时报告 `bytes_per_full_token=1670.75`（FP8 代理 build），这是 backend/runtime
  layout 的值；MSV 对 V4.1 的模型设计口径仍是 FP4 `890 B/token`。本轮不把 FP8 实测值写回产品。

## 分项对账与边界

### 精确分项（整数张量字节，不是日志舍入值）

| 分项 | runtime 真值 | MSV / 判定 |
|---|---:|---|
| H20 reduced Qwen GDN state/slot | `424,968,192 / 33 = 12,877,824 B` | 完全一致；33 为 32 用户 slots + sentinel |
| H20 reduced GLM KDA state/slot | `859,668,480 / 33 = 26,050,560 B` | 完全一致 |
| A100 Qwen3.5-4B target GQA KV/token | 8 full layers × K/V × 4 heads × 256 × BF16 = `32,768 B` | 完全一致 |
| A100 Qwen3.5-4B MTP KV/token | 1 full layer × K/V × 4 heads × 256 × BF16 = `4,096 B` | 两框架一致；SGLang target/draft 张量 storage 无交集 |
| A100 Qwen3.5-4B 基础 state/slot | `875,692,032 / 17 = 51,511,296 B` | 完全一致，不含 speculative scratch |
| SGLang MTP speculative state 实际额外 storage | `408,944,640 B` | 验证时基础 state 公式未覆盖；2026-09-22 已以有效 3 请求槽、2 draft tokens 回放精确命中该分项，不准出完整 resident total |
| vLLM GLM k-pool=4 index 增长 | 每层 `132/4 = 33 B/token` | 验证时旧版 MSV 仍按 132 B/token；减层全模型 `2114` vs `2312 B/token`，后续已加 framework-conditioned compression rule |
| SGLang GLM index 预分配容量 | 每层 `129×8448 B`（8192+64 slots） | 此版本仍按完整 slot 预留，主 KV+index 容量为 `2312 B/token`；不能套用 vLLM 的压缩增长 |
| H20 V4.1 draft SWA 实际 storage/rank | 3 × 77 pages × 149760 = `34,594,560 B` | 当前逻辑窗口 `393,216 B` 不是实际 pool allocation 上界；缺保留槽/packing |

SGLang speculative conv view 有重叠：逻辑 view 为 `9,437,184 B`，实际 backing
storage 为 `6,291,456 B`；总 scratch 使用 unique storage，而非相加所有 view。
DSpark target 和 draft 的 SWA 指针在同一 worker 内逐个比较，无交集；
不同进程之间的相同指针值从不作为共享证据。

权重 probe 的 `load_model` 时点可能早于 draft embedding/head 与 target 绑定；
记录到的 draft `named_parameters` 总量不能直接当作服务 ready 后的独占权重。
本轮不把这个瞬时总量拿来“校准”前端 draft weight。

| 项 | 结论 |
|---|---|
| weight bytes | Qwen3.5 / V2-Lite / V4.1 runtime 可取到；MSV 对内置模型按图和 checkpoint 口径，未把 runtime workspace 或量化 packing 写成经验系数 |
| KV bytes/token | GDN、DSA、V2-Lite MLA、V4.1 FP8 runtime 均取到；V4.1 FP8 与 MSV FP4 设计值不是同一 dtype regime |
| state bytes/request | GDN/KDA 的 BF16 conv + FP32 temporal 与 profile 一致；SGLang speculative scratch 已支持显式 workload 公式，缺 workload 参数时仍列为 unknown |
| draft bytes/token | MTP/EAGLE 有独立 draft KV；DSpark 真实 build 的 draft compressed KV 为 0、SWA 独立，不能统一套 MTP 线性增长公式 |
| shared pool | 只有明确同一 storage/pool alias 才去重；本轮 DSpark capture 不支持 target/draft 存储去重 |
| total resident | 框架还包括 page rounding、pool reserve、CUDA graph/workspace、scratch；不与理论 ledger 总数直接相等 |
| communication | 本轮验证了 EP placement 和服务请求；没有新增 all-to-all/NCCL 性能 benchmark，沿用既有通信证据 |

### 明确未闭合

1. DSpark 具体 backend page packing、reserve slots、压缩 state 和 workspace 的逐字节公式。
2. V4.1 原生 FP4 indexer；H20 只验证了 FP8 代理 build，原生 FP4 需要相应硬件/镜像。
3. 每个真实 checkpoint 的逐张量 draft weight attribution；runtime 能证明 draft weight 独立存在，
   但不能把总 checkpoint 权重简单当作精确 draft 分量。
4. SGLang/vLLM 不同版本的 speculative scratch 与 cache-group 预留差异。
   当前 MSV 对 SGLang 已支持显式 `draftTokens` / 有效 `stateSlots` 的理论
   scratch 公式，但不猜物理页保留、CUDA graph 或 workspace；vLLM MTP 的
   cache-group scratch 尚未实现；本次没有把无法证明的预留量伪装为 0 或已对齐。
5. A100 vLLM MTP 出现 cache-group warning：未能识别独立 draft group，
   因而禁用跨请求 prefix-cache reuse。生成通过不意味着该 runtime 的 prefix-cache 功能通过。

这些项继续作为 `unknown/evidence gap`，没有把实测显存、吞吐、延迟或一次运行的比例硬编码进 MSV。
**因此本次验证不能把 framework accounting 标成“全部正确/完全对齐”。**

### 产品修复跟进（2026-09-22）

- vLLM DSA `index_kpool` 已进入 framework profile：index 的线性增长按
  `base_growth / index_kpool` 计算；SGLang 保留 token-granular capacity。
  该修复使用上游 cache layout 语义，不使用 H20/A100 实测倍率。
- SGLang speculative state scratch 已增加显式 workload 入口：给出
  `draftTokens`、每个 attention worker 的有效 `stateSlots`（以及可选
  attention-DP/top-k）时，按上游 `SpeculativeState` 的 SSM 与 conv-window allocation shape 计入
  `speculativeStateBytes`，并接入 stage Fit/Max Context；不提供这些运行时参数时
  仍保持 unknown，不把本次 408,944,640 B 预分配实测写成默认常数。
- PD prefill profile 按 SGLang 的 disaggregation 语义不分配 target-verify
  scratch；PD decode 才按有效 `stateSlots` 计入。该 worker-local scratch
  不进入 prefix-cache 的 PD 传输字节。
- 归档回放结果：SGLang SSM `402,653,184 B` + conv unique storage
  `6,291,456 B` = `408,944,640 B`，当前公式精确命中。命令行请求上限为 4，
  但 capture 的有效请求槽为 3（加 1 sentinel）；不以原始 CLI 值代替实际
  约束后的 workload。vLLM reduced GLM 主 KV+index 增长回放为
  `2114 B/token`，当前 profile 同值。该回放不是新增 GPU 请求结果。
- `CostSummary` 现在逐项显示 `runtimeWorkspace`、DSpark page packing/reserve、
  compressed-state/window allocation 等 unknown 字段，避免把理论账本误读为
  runtime resident total。
- MTP speculative scratch、DSpark physical page reserve、backend packing/workspace
  仍属于缺少完整 runtime 配置的 unknown；本次未把一次容器的预分配字节写入
  Graph IR 或默认公式。可由源码和显式 workload 推导的 SGLang state scratch
  已不再是“未实现”，而是“无 workload 时 unknown”。

## 复现材料

- runtime probe：`scripts/evidence/runtime/sitecustomize.py`、`run_smoke.py`、`summarize.py`、`accounting.mjs`。
- 公式回放：`scripts/evidence/runtime/reconcile-accounting.mjs`（只读归档
  shape/dtype/storage metadata，不重新起 GPU 服务）。
- H20 归档：`artifacts/framework-runtime-validation/20260921/h20-evidence-final.tar.gz`，SHA-256
  `f40417616cd2255b79ada64969fa9bd1cca9b657de226b375ff560083c8c83dd`。
- A100 归档：`artifacts/framework-runtime-validation/20260921/a100-evidence-final2.tar.gz`，SHA-256
  `bb66e0fc5bf24bf69cb2e036a6d478a50ea50e1f3b1282dbae58b131d139c06b`。
- 模型获取：本轮优先使用两台机器既有模型；没有为了验证强制下载大权重。BOS 传输仅用于 probe、checkout snapshot 和证据归档，未上传权重。
