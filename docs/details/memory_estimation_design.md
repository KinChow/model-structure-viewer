# 显存估计系统设计

## 1. 目标

MSV 的目标不是预测某个 serving runtime 的全部物理行为，而是提供三层可解释答案：

```text
理论：从模型事实和 workload 能推出多少驻留显存？
实现：vLLM/SGLang 根据自己的实现会怎样解释这些字段？
运行时：目标版本和硬件实际预留/占用了多少？
```

设计必须支持三层并存，且用户能看出它们不是同一个数。

## 2. 架构边界

```text
Graph IR + normalized config + logical parallel plan
                    │
                    ▼
        Framework Runtime Profile
        ├── neutral: config-faithful
        ├── vLLM: source-backed semantics
        └── SGLang: source-backed semantics
                    │
                    ▼
        Theoretical Cost Accounting (T0/T1)
                    │
                    ├── Cost breakdown
                    ├── Fit / Max Context
                    ├── Roofline bytes
                    └── PD logical transfer

Optional offline Runtime Evidence
                    │
                    ▼
        Framework Runtime Estimate / Observed Ledger (T2/T3)
                    ├── graph pool
                    ├── page/reserve
                    ├── activation/workspace
                    ├── communication buffers
                    └── allocator/driver snapshots
```

Runtime Evidence 是可选的、离线的、带 fingerprint 的输入；它不是新的 serving backend，也不是浏览器运行时依赖。

## 3. 两本账与一份证据

### 3.1 理论账

理论账保持当前 `CostAccounting`：

```text
weights
+ declared buffers
+ unique main/draft/shared KV
+ persistent state
+ explicit speculative state
= theoretical resident
```

它继续驱动默认的：

- Cost breakdown；
- Total VRAM；
- Fit/no-fit；
- Max Context；
- per-stage HBM；
- PD prefill/decode logical fit；
- Roofline memory bytes。

### 3.2 运行时账

运行时账不覆盖理论账，而是增加生命周期和证据：

```text
startup resident
  = weights
  + theoretical cache/state
  + activation profile
  + graph pool
  + backend workspace
  + communication buffers
  + allocator/driver reserve
```

同一 allocator pool 可能在多个生命周期复用，因此 runtime ledger 必须支持：

- `exclusive`：不能与其他 component 重叠；
- `shared_pool`：多个 graph/storage 共享同一 pool；
- `view_of`：逻辑 view 不增加 backing bytes；
- `peak_only`：只在某 phase 出现，不计入 resident total；
- `unknown_overlap`：有证据但无法证明是否与另一项重叠。

### 3.3 证据

证据不是第三本“可加总的账”，而是每个数的 provenance：

```text
formula source
framework source revision
runtime resolved config
startup profile
actual capture
allocator/NVML snapshot
```

## 4. 用户可见模式

### 默认：理论模式

不加载 runtime evidence 时：

- 显示 theoretical resident；
- 显示 framework profile 名称；
- 显示 unknown fields；
- Fit 只判断理论账；
- 不显示伪造的 graph/workspace/allocator 常数。

### 可选：框架估计模式

用户显式填写 workload 或导入 T2 evidence 后：

- 显示 resolved graph shapes；
- 显示 page/reserve 和 graph estimate；
- 显示“估算，不是实测”；
- 如果缺少 source/config 字段，保留 unknown；
- 不改变理论账的来源。

### 可选：运行时证据模式

用户导入与当前 fingerprint 完全匹配的 T3 evidence 后：

- 同时显示 theoretical / framework estimate / observed；
- 显示每个 component 的差异；
- 可以显示 `runtime fit`，但不能把它替代为默认 Fit；
- fingerprint 不匹配时 fail-closed：只展示证据，不参与当前模型 Fit。

## 5. Fingerprint 与匹配

至少包含：

```text
model revision / weight manifest
Graph IR or catalog revision
framework name + version + commit
container image digest
GPU model + UUID or architecture
driver + CUDA
TP/PP/DP/EP plan
max_model_len
max_num_seqs / max_num_batched_tokens
speculative configuration
prefix-cache configuration
CUDA Graph configuration
```

以下任一变化都应使 T3 evidence 降级为参考：

- vLLM/SGLang commit 或 image digest 变化；
- GPU architecture、driver 或 CUDA 变化；
- model checkpoint / quantization 变化；
- capture sizes、MTP draft tokens、max batch 变化；
- parallel plan 或 cache dtype 变化。

## 6. 逐项对账界面

UI 不应该只显示“总显存差异 xx%”，而应显示：

| Component | Theory | Runtime estimate | Observed | Class | Difference reason |
|---|---:|---:|---:|---|---|
| weights | bytes | bytes | bytes | T0/T2/T3 | packing / padding |
| main KV | bytes | bytes | bytes | T1/T2/T3 | block/page/reserve |
| draft KV | bytes | bytes | bytes | T1/T2/T3 | pool ownership |
| shared KV | bytes | bytes | bytes | T1/T2/T3 | alias proof |
| persistent state | bytes | bytes | bytes | T1/T2/T3 | dtype/shape |
| speculative scratch | bytes | bytes | bytes | T1/T2/U | slots/draft tokens |
| CUDA Graph | — | bytes | bytes | T2/T3 | capture shape/pool reuse |
| workspace | — | bytes | bytes | T2/T3/U | backend/driver |
| comm buffers | — | bytes | bytes | T2/T3/U | NCCL/EP implementation |
| allocator/driver | — | bytes | bytes | T3/U | process/device |

差异如果不能归因到 dtype、shape、ownership、sharding、padding、pool reuse 或生命周期，就显示 `evidence gap`，不允许自动生成校准系数。

## 7. 分阶段落地

### Phase A：文档和当前实现收口

- 保持 `CostAccounting` 理论职责不变；
- 将 T0/T1/T2/T3/U 作为统一词汇；
- 统一 unknown fields 文案；
- 在验证状态中引用 H20/A100 graph/prefix evidence；
- 不新增 serving runtime。

### Phase B：离线 Runtime Evidence schema

- 增加 JSON schema 和 fingerprint 校验；
- 导入 vLLM/SGLang 的 startup/capture 摘要；
- 显示 component table；
- 未匹配 evidence 不参与 Fit。

### Phase C：framework parser

- vLLM：解析 resolved KV groups、graph descriptors、estimate/actual graph pool、KV capacity；
- SGLang：解析 phase graph buckets、token/request pools、Mamba/DSA state、reserve；
- parser 只读取现有日志、metrics、capture 和导出的 runtime config，不启动服务。

### Phase D：GPU 运行时采集器（范围外的独立工具）

- 在目标容器内运行 bounded probe；
- 输出脱敏 JSON evidence；
- 通过 checksum/BOS 或用户指定方式传回；
- MSV 只导入和解释 evidence，不管理远端生命周期。

## 8. 验收标准

理论层：

- 同一 Graph IR、config、plan 得到确定且可复现的 T0/T1 数值；
- main/draft/shared pool 不重复；
- Fit、Max Context、PD 使用同一个理论账。

实现层：

- vLLM/SGLang profile 规则都有 source anchor；
- 没有 runtime input 时不猜 state slots、reserve、workspace；
- profile 不包含 H20/A100 实测常数。

运行时层：

- 每个 T3 数值可追溯到 framework/version/image/GPU/workload；
- graph estimate 和 actual capture 分开；
- 逐项对账优先于 total 对账；
- 差异无法解释时保留 unknown/evidence gap。

## 9. 本次证据的设计结论

H20 和 A100 的 Qwen3.5-4B vLLM 运行已经证明：

- `max_num_seqs`、`max_num_batched_tokens`、MTP draft tokens、capture sizes 需要进入 workload；
- graph estimate 与 actual graph pool 必须分栏；
- MTP warning 不能替代 prefix-cache hit 指标；
- 两台机器的 `vllm-0920` 版本不同，不能直接产生硬件独立结论；
- 运行时证据应作为 T3 artifact 保存，而不是写入理论公式。
