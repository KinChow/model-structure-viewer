# 显存估计理论口径

> 本文定义 MSV 的显存对象、分类和公式。它回答“从 Graph IR、规范化配置和并行计划能严格推出什么”，不把 serving runtime 的一次启动结果伪装成理论真值。
>
> 关联实现：[`framework_accounting.md`](framework_accounting.md)、[`memory_estimation_implementation.md`](memory_estimation_implementation.md)、[`memory_estimation_design.md`](memory_estimation_design.md)。

## 1. 三个不同问题

“显存”至少包含三个不同问题，不能用一个 `total VRAM` 混合回答：

1. **理论驻留容量（theoretical resident）**：模型在给定 workload 下，由权重、唯一 cache pool、持久 state 和显式 buffer 构成的闭式账本。
2. **框架启动预估（framework runtime estimate）**：框架根据最终解析后的参数、最小 KV cache、capture shape 和 profiling 估算将要预留的显存。
3. **运行时物理驻留（runtime observed resident）**：进程、CUDA allocator、CUDA Graph pool、通信库和驱动真正占用的显存快照。

三者关系不是恒等式：

```text
theoretical resident
  <= framework runtime estimate / reserved budget
  <= runtime observed resident       （通常，但不保证逐项单调）
```

原因是运行时估算可能包含保守 headroom，而实际 allocator 又可能复用 graph pool、释放临时块或受碎片影响。因而产品必须同时显示数值和证据等级，不能只显示一个更大的数字。

## 2. 五级证据分类

| 等级 | 名称 | 可否由 Graph IR 独立得到 | 允许进入默认 Fit | 典型内容 |
|---|---|---:|---:|---|
| T0 | 结构精确 | 是 | 是 | shape、dtype、参数元素数、显式 buffer |
| T1 | framework-conditioned closed form | 需要框架语义 profile | 是，但标记 profile | KV layout、GQA/MLA/DSA、state dtype、pool ownership、逻辑分片 |
| T2 | framework runtime estimate | 需要最终 runtime config 或启动 profiling | 默认不覆盖 T0/T1 | page padding、reserve blocks、effective slots、CUDA Graph 估算、backend capacity |
| T3 | runtime observed | 只能在目标版本/硬件/参数上测量 | 不覆盖理论 Fit | actual graph pool、allocator reserved、NVML、workspace、通信 buffer |
| U | opaque / unknown | 否 | 否 | 驱动 context、碎片、未暴露 workspace、未证明的压缩/页保留 |

规则：

- T0/T1 可以由公式产生确定数值；缺输入时输出 `unknown`，不能填 0。
- T2 必须携带框架、版本、硬件、启动参数和估算方法。
- T3 必须携带运行证据和 fingerprint；它是某次环境的观测，不是前端常数。
- U 必须在 accounting evidence 中列出缺项，不得用比例系数“校准”成已知量。

## 3. 统一的理论驻留账本

```text
TheoreticalResident {
  weights                 // 权重驻留字节
  buffers                 // Graph IR 显式声明的持久 buffer
  mainKv                  // main-owned unique cache pools
  draftKv                 // draft-only unique cache pools
  sharedKv                // shared-owned unique cache pools
  state                   // 持久 KDA/GDN/Mamba/压缩 state
  speculativeState        // worker-local target-verify scratch
  total
}
```

总量只能按唯一 storage 归属计算：

```text
totalKv   = mainKv + draftKv + sharedKv
total     = weights + buffers + totalKv + state + speculativeState
```

禁止：

```text
mainKv + draftKv                 // draft 与 main 可能是同一 pool
sharedPool + mainPool             // alias 未去重
logical view bytes + backing bytes // view 与 backing storage 重复
```

同一个 `cache_pool_id` 只代表同一 storage alias；共享 allocator、相同映射或相同窗口形状都不能自动去重。只有 Graph IR 或 framework profile 明确证明同一 backing storage 时，才可归入 `sharedKv`。

## 4. 组成项及准确性边界

### 4.1 权重

分成三个数：

```text
checkpointWeightBytes  // safetensors/index 或文件事实
logicalWeightBytes     // Graph IR shape × dtype
residentWeightBytes    // framework load/packing 后的物理驻留
```

当前 MSV 对 checkpoint 总字节和声明 shape 的逻辑字节可以做到精确；TP/PP/EP 只负责逻辑归属和每卡投影。以下情况转入 T2/U：

- runtime quantization packing 与 checkpoint 存储格式不同；
- fused weight、padding、kernel-specific scale/zero layout 未由 profile 证明；
- draft weight 无法按 checkpoint tensor 唯一归属；
- load-time temporary buffer 未释放或 allocator 复用情况未知。

不能把一次 `load_model` 快照中的临时 draft 参数总量直接当成最终 draft resident weight。

### 4.2 KV cache

逻辑 KV bytes/token 在以下条件满足时是 T1：

```text
layer count
× KV heads / latent width
× head dimension
× K/V components
× explicit cache dtype bytes
÷ effective sharding divisor
```

MLA、GQA、DSA index、SWA 和 MTP draft KV 必须使用各自 cache spec，不能统一套一个 BF16 fallback。当前已验证或实现的规则包括：

- standard GQA/MLA 的每 token logical bytes；
- 显式 DSA index dtype（包括 FP8/scale 布局）；
- vLLM DSA k-pool 的 index growth divisor；
- SGLang token-granular DSA capacity；
- MTP/EAGLE 独立 draft KV pool；
- DSpark target/draft 只有显式 alias 才共享。

以下属于 T2/U：

- runtime block/page padding；
- cache pool reserve slots；
- backend-specific page packing；
- sliding-window 的物理 pool 上限；
- external KV/offload tier 和 eviction 细节。

### 4.3 持久 state 与投机 scratch

持久 state 是跨 step 保留的 state；投机 scratch 是一次 worker-local target verify 所需的中间 buffer，二者必须分开：

```text
persistentState       // Fit / resident / stage HBM
speculativeState      // target worker resident；按 workload 增长
pdKvTransferBytes     // 不包含 worker-local speculative scratch
```

当 state shape、dtype、draft tokens 和 runtime cap 后的有效 request slots 已知时，SGLang GDN/KDA speculative scratch 可以按 allocation shape 做 T1/T2 公式计算；当前已验证 SSM sentinel row、conv-window dense/dedup 两类布局。没有有效 slots 或启用了尚未建模的 ReplaySSM 时保持 U。

### 4.4 Activation、workspace 和临时 buffer

这些通常不能从 Graph IR 独立得到：

- prefill activation peak 随 token shape、batch、kernel 和 fused path 变化；
- decode activation 与 speculative query length、capture shape 变化；
- FlashInfer/cuBLAS/cuDNN/DeepGEMM/Triton workspace 随版本和 backend 变化；
- NCCL/NVSHMEM/EP dispatch buffer 随通信实现和拓扑变化。

它们可以由 framework runtime profile 或启动 profiling 给出 T2 估计，但没有 runtime evidence 时不进入默认理论 Fit。

### 4.5 CUDA Graph pool

CUDA Graph 显存是 capture shape 的 memory pool，不是简单的“batch × token × 常数”：

```text
captureShapes = f(
  max_num_seqs,
  max_num_batched_tokens,
  max_model_len,
  speculative_tokens,
  cudagraph_capture_sizes,
  graph_mode,
  parallel_plan,
  backend
)

graphPoolEstimate
  = first_capture_pool
  + incremental_full_graph_pool
  + piecewise_graph_pool
  + speculator_graph_pool
  + encoder_graph_pool
```

多个 graph 可能共享 pool，不能把每个 graph 的 memory delta 无条件相加。vLLM 的成熟实现是先以最小 KV cache 做 profiling，再对 FULL graph 样本外推，并在真正 capture 后记录 actual graph pool；MSV 应保留这两个字段。

本次 H20/A100 实测说明：相同模型和大致 workload 下，改变 capture 上限会改变 graph pool，但 runtime actual 与 estimate 不相等。因此：

- `max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、draft tokens、capture sizes 都是必需输入；
- API 的单请求 `max_tokens` 不等价于 graph capture token budget；
- `--enforce-eager` 下 graph pool 为 0，但不能代表 CUDA Graph 开启路径。

### 4.6 allocator、driver 和碎片

下列项目只能观测或由运行时给出区间：

```text
CUDA context / driver reservation
PyTorch memory_reserved - memory_allocated
allocator fragmentation
NCCL communicator/bootstrap buffers
persistent runtime handles
JIT/autotune cache allocations
```

它们不属于 MSV 默认理论公式，也不允许通过 H20/A100 某次实测比例写入前端。

## 5. Fit 与 Max Context 的口径

默认 Fit 只使用 T0/T1 理论账本：

```text
fit_theoretical = theoreticalResident <= cardCapacity
```

如果用户提供了完整 T2/T3 evidence，则可以额外显示：

```text
fit_runtime_evidence = runtimeObserved <= cardCapacity
```

二者不覆盖彼此。Max Context 继续基于理论 growth：

```text
fixed = weights + buffers + state + speculativeState + boundedKv
 growth = kvBytesPerSequence
 maxContext = floor((cardCapacity - fixed) / growth)
```

runtime page reserve、Graph pool 和 workspace 只能作为附加的 runtime capacity warning，不能静默改变理论公式。

## 6. 当前结论

### 当前理论已经比较准确

- Graph IR 声明 shape、dtype、参数元素数；
- checkpoint 总权重字节；
- 逻辑 TP/PP/EP/DP 权重归属；
- unique cache pool 去重和 main/draft/shared roll-up；
- standard GQA/MLA KV logical bytes/token；
- explicit DSA dtype 与 vLLM k-pool growth rule；
- GDN/KDA state dtype 语义；
- MTP/EAGLE draft KV 的独立 pool 语义；
- SGLang speculative scratch（输入显式有效 slots/draft tokens 时）；
- 理论 Fit、Max Context、per-stage HBM 和 PD side accounting 的统一消费。

### 必须引入框架 profile 或 runtime evidence

- 最终 page/block size、padded page bytes、reserve slots；
- effective max requests / token cap；
- CUDA Graph capture shapes、graph pool estimate 和 actual pool；
- backend workspace、activation peak、NCCL/EP buffers；
- quantized resident packing 与 runtime temporary；
- allocator/driver overhead、fragmentation；
- 不同版本对 speculative cache-group、DSpark page reserve 和 ReplaySSM 的实现差异。

### 明确不做

- 不把实测显存变成默认常数；
- 不通过 magic multiplier 消除理论与实际的差异；
- 不在浏览器运行时启动 serving runtime；
- 不把 runtime observed total 伪装成 Graph IR 的精确结果；
- 不自动搜索使物理显存最小的并行方案。
