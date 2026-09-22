# 显存估计实现边界

本文把显存理论口径落到当前 MSV 代码和后续 runtime evidence 接口。实现原则是：Graph IR 负责模型事实，framework profile 负责来源可证明的 runtime 语义，运行时 evidence 负责实际预分配和物理驻留；三层不互相覆盖。

## 1. 当前实现映射

| 层 | 当前入口 | 当前职责 | 不负责 |
|---|---|---|---|
| Graph IR / normalized config | `frontend/src/structure` | shape、dtype、weight declaration、cache/state attributes | kernel workspace、调度、allocator |
| framework profile | `frontend/src/frameworkProfiles.js` | vLLM/SGLang 的 plan、dtype、cache ownership、DSA growth、speculative scratch | 启动 serving、经验校准 |
| theoretical accounting | `frontend/src/cost/memory.js` | weights、buffers、unique KV、state、speculative state、total | actual graph pool、driver overhead |
| parallel projection | `frontend/src/cost/parallel.js` | per-stage bytes、Fit、Max Context 输入 | runtime placement 细节和自动搜索 |
| communication / PD | `frontend/src/cost/comm.js` | logical communication、PD transfer；排除 local scratch transfer | NCCL allocation、overlap 后时延 |
| UI | `frontend/src/components/CostSummary.jsx` | 展示理论账本和 unknown fields | 直接拼 main/draft KV、隐藏 unknown |

`docs/details/framework_accounting.md` 是 profile 语义的来源边界；本文不重复替代其中的模型家族规则。

## 2. 统一数据模型

后续实现建议增加两个内部对象，仍不暴露为 serving backend：

```js
RuntimeMemoryRequest {
  framework: "neutral" | "vllm" | "sglang",
  frameworkVersion: string | null,
  modelRevision: string | null,
  device: { name, memoryBytes, driver, cuda },
  parallelPlan,
  workload: {
    maxModelLen,
    maxNumSeqs,
    maxNumBatchedTokens,
    batch,
    prefillTokens,
    decodeTokens,
    draftTokens,
    maxRunningRequests,
    stateSlots,
    captureSizes,
    graphMode,
    prefixCaching,
  },
  configFingerprint: string,
}

RuntimeMemoryComponent {
  id,
  phase: "startup" | "prefill" | "decode" | "resident",
  ownership: "weights" | "kv" | "state" | "graph" | "workspace" | "comm" | "allocator" | "driver",
  bytes: number | null,
  class: "T0" | "T1" | "T2" | "T3" | "U",
  method: "formula" | "source-profile" | "startup-profile" | "allocator-snapshot" | "nvml" | "unknown",
  source: string[],
  notes: string[],
}

RuntimeMemoryEvidence {
  request: RuntimeMemoryRequest,
  components: RuntimeMemoryComponent[],
  totals: {
    theoreticalResidentBytes,
    frameworkEstimateBytes,
    observedReservedBytes,
    observedUsedBytes,
  },
  capture: {
    graphCount,
    graphShapes,
    estimatedGraphPoolBytes,
    actualGraphPoolBytes,
  },
  unknownFields: string[],
  evidenceFiles: string[],
}
```

所有 component 必须有 `ownership` 和 `class`。同一 allocation 不能同时被两个 component 计入；同一个 pool 的不同 logical view 只能保留一个 backing storage component。

## 3. vLLM adapter

vLLM adapter 的输入和输出应分开：

```text
输入：
  max_num_seqs
  max_num_batched_tokens
  max_model_len
  speculative_config.num_speculative_tokens
  cudagraph_capture_sizes / max_cudagraph_capture_size
  compilation mode / graph mode
  TP/PP/EP/DP

输出：
  resolved scheduler limits
  resolved KV cache groups/specs
  block/page sizes and padding
  num_gpu_blocks / kv_cache_size_tokens
  capture descriptors and graph mode
  estimated graph pool bytes
  actual graph pool bytes（如果已启动）
```

实现规则：

1. graph shape 必须使用框架最终解析后的 capture sizes，而不是仅使用用户输入的最大值。
2. MTP 的 decode query length 要纳入 capture descriptor；不能按普通 decode batch 复用公式。
3. `max_num_batched_tokens` 与 `max_num_seqs` 分别作为 token cap 和 request cap；不能用 API `max_tokens` 替代。
4. `profile_cudagraph_memory` 的 estimate 和真正 `capture_model` 的 actual 分别记录。
5. MTP warning 只进入 `unknownFields` / source evidence，不得单独转换成 prefix-cache failure；必须结合 `prefix_cache_hits`、`cached_tokens` 或等价指标。
6. framework version、image digest、model revision、GPU UUID 必须进入 fingerprint。

本次 H20/A100 证据表明，vLLM 的 graph estimate 与 actual pool 都可以直接从日志和 runtime capture 获取，但两个容器版本不同，不能合并成一个硬件校准系数。

## 4. SGLang adapter

SGLang adapter 需要按 phase 和 backend 解析：

```text
cuda_graph_config.decode
cuda_graph_config.prefill
cuda_graph_max_bs_decode / prefill
cuda_graph_bs_decode / prefill
max_context_size
full_prefill_max_req
max_seq_len
max_running_requests
max_total_num_tokens
mamba cache ratio / pool cap
```

至少输出：

- decode/prefill graph bucket 和实际捕获 shape；
- request pool、token pool、Mamba state pool 的 effective capacity；
- DSA index capacity 与 dtype；
- KDA/GDN state dtype 和 persistent bytes；
- speculative SSM/conv scratch 的公式 bytes；
- dense/dedup backing storage；
- page reserve、workspace、allocator snapshot 的 unknown 或 observed evidence。

SGLang 的 `max_running_requests` 不能直接作为 state slots；必须先经过 token pool、Mamba pool 和 attention-DP 约束，得到每个 attention worker 的有效容量。

## 5. 证据采集协议

每次 runtime evidence 必须按下列顺序记录：

1. **环境指纹**：host、GPU、driver、CUDA、container image、framework version/commit、model path/revision。
2. **启动参数**：完整 argv、环境变量、parallel plan、workload、prefix-cache 和 graph 配置。
3. **理论账本**：同一 config 在本地生成 T0/T1 accounting，并保存 JSON。
4. **框架解析结果**：cache group/spec、block/page、effective capacity、capture descriptors。
5. **启动 profiling**：available KV memory、graph estimate、weights/non-KV profiling。
6. **真实 capture**：actual graph pool、captured graph count、graph mode。
7. **请求验证**：最小成功请求、prefix-cache cold/warm、必要的 MTP on/off 对照。
8. **清理证据**：本任务进程组、端口、GPU memory/process snapshot。
9. **分项对账**：weight、KV/token、state/request、draft KV、shared pool、graph pool、workspace、total。

缺一项不应把整个总量标成“完全对齐”；应按 component 降级为 T2 或 U。

## 6. 当前已验证和待补接口

### 已实现并有 GPU 证据的理论/语义

- main/draft/shared unique pool accounting；
- vLLM TP-only MoE 与显式 EP 的 plan 语义；
- vLLM DSA k-pool index growth；
- SGLang DSA/KDA/GDN dtype 语义；
- MTP/EAGLE 独立 draft KV；
- DSpark target/draft 私有 SWA ownership；
- SGLang speculative state 的显式 workload 公式；
- 理论 Fit / Max Context / per-stage / PD 使用同一账本。

### 尚未实现为产品输入的 runtime evidence

- `RuntimeMemoryRequest` 的持久化和 fingerprint；
- vLLM/SGLang capture descriptor 解析器；
- actual graph pool 与 theoretical ledger 的双层 UI；
- backend workspace、NCCL/EP buffer、allocator reserve 的 component parser；
- runtime evidence 与当前模型/硬件/框架不匹配时的 fail-closed 校验；
- evidence file 的导入、导出和脱敏。

下一步应先实现离线 JSON evidence 导入和只读展示，再考虑任何自动远程启动；MSV 不应把 serving runtime 嵌入浏览器。
