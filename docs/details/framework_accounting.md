# Framework-conditioned cost accounting

> 2026-09-21 UTC runtime validation update:
> [H20/A100 report](evidence/memory/framework_runtime_validation_20260921.md)
> confirms state dtype/shape, private MTP KV and DSpark storage ownership.
> The follow-up fixes vLLM DSA k-pool growth and adds SGLang speculative state
> scratch with explicit effective workload inputs. Unspecified runtime
> allocations (DSpark physical reserve, backend packing and workspace) remain
> explicit unknowns rather than empirical corrections to theoretical capacity.

MSV keeps one Graph IR and one theoretical accounting path. A framework
profile selects implementation semantics; it does not emulate a serving
runtime and never contains measured GPU constants, throughput, latency, or
calibration multipliers.

## Profile boundary

`neutral` is config-faithful. `vllm` and `sglang` are source-backed profiles
used to resolve plan defaults, state dtype, communication options, and cache
ownership. If a relationship is not proven by the graph or the selected
profile, the result remains conservative and the accounting evidence lists the
unknown field.

The cache ledger is:

```text
unique cache pools
  -> main / draft / shared
  -> totalKv = mainKv + draftKv + sharedKv
  -> totalVram = weights + buffers + totalKv + totalState + speculativeState
```

The UI, Fit, Max Context, per-stage resident HBM and both PD fit projections
consume this same ledger. Roofline uses its weight bytes and the same state
dtype resolver, but **forward traffic is not resident capacity**: an inactive
draft has resident storage, not an extra executed forward. Sparse cache reads
remain operator action counts, not a sum of resident pools.
Speculative state scratch is a worker-local temporary buffer: it participates in
resident Fit, Max Context, per-stage HBM and PD-side Fit, but is not persistent
prefix-cache data and therefore is not included in `pdKvTransferBytes`.
The UI must not add `mainKv + draftKv` a second time.
An explicit `cache_pool_id` is an alias to one resident pool. A shared mapping
or shared allocator without the same storage alias is not enough to deduplicate
the pool.

`projectPlan` assigns each pool to its declared layer range; draft pools are
on the last PP stage. KV follows the existing GQA/MLA/DP sharding contract,
state follows attention TP, and declared buffers are included in Fit.
When an explicit SGLang speculative workload is supplied, source-backed
intermediate SSM/conv scratch is added as `speculativeStateBytes` and follows
the same attention-TP projection. The input must include `draftTokens` and the
effective per-attention-worker `stateSlots` after runtime caps; passing only the
raw CLI `max-running-requests` is not sufficient when SGLang caps the pool.
Absent those runtime settings, scratch remains unknown and is not guessed.
For a stage with affine growth:

```text
fixed = weightBytes + bufferBytes + stateBytes + speculativeStateBytes + boundedKvBytes
growth = (kvBytes - boundedKvBytes) / sequence
maxContext = floor((cardCapacity - fixed) / growth)
```

The current DP workload is per replica, as in the existing parallel protocol;
the UI does not divide a replica's request batch by DP a second time.
Max Context is a memory bound, not a promise beyond the model's context limit.

## Runtime choices and conservative cases

- vLLM: dense and EP-off MoE retain the user TP plan. Explicit EP enables
  `EP=TP×DP`, expert TP=1. These effective axes are not written back into the
  editable base plan, so disabling EP does not leave a stale `moeTp=1`.
- vLLM DSA k-pool divides the **index growth** by `index_kpool`; SGLang keeps
  its token-granular index-capacity rule. This is a source-backed cache-layout
  distinction, not a calibration multiplier.
- SGLang shared-expert fusion defaults off. Precedence is function option,
  camel-case plan option, snake-case plan option, then false. Only SGLang's
  explicit opt-in changes dispatched expert count.
- SGLang MTP/EAGLE scratch follows the upstream allocation shape:
  `(effectiveStateSlots + 1) × draftTokens × recurrentState` plus the BF16
  conv-window buffer. `effectiveStateSlots` is the per-attention-worker
  capacity after DP and mamba-cache caps. Linear-chain verification uses the
  deduplicated `[conv_window + draftTokens - 1]` layout for CUDA GDN;
  KDA (transpose), tree verification, CPU/NPU and explicit dedup-off use
  dense per-token windows. Draft workers and PD prefill do not execute
  target verification and skip this scratch. ReplaySSM is not represented
  by this formula; explicit ReplaySSM input yields an evidence gap.
- Runtime DSpark SWA is represented by the private logical window reserve
  `layers × window × head_dim × BF16 bytes × batch`. This is the uncompressed
  row bound, **not a claim about a paged backend's allocated pool size**.
  Main DeepSeek V4 window reserves use the same declared shape; compressed
  growth remains the graph's explicit FP4/FP8 declaration. Neutral retains
  the earlier full-context draft upper bound unless ownership is declared.
- Pool id aliases are deduplicated regardless of declaration order, including
  state. Conflicting sizes retain the larger bound and emit an evidence gap;
  an alias is not silently treated as two independent allocations.
- Checkpoint total weight is preserved. Main/draft attribution uses the
  existing graph-weight proportion, not a second checkpoint scan.

## Evidence boundary

- vLLM DeepSeek V4.1 DSpark exposes one SWA cache layer per draft layer via
  `get_draft_kv_cache_layer_names`; this is modeled as private draft cache
  storage unless Graph IR explicitly aliases a pool.
- SGLang's DSpark pool configurator allocates a draft ring (including the
  unified-fp8/bfloat16 branch); target mapping reuse is not treated as storage
  deduplication.
- vLLM GDN `auto` follows model dtype, while vLLM KDA `auto` uses FP32;
  SGLang defaults the SSM/temporal state to FP32 and reads an explicit config
  override. The profile changes only these formula inputs.
- DSA explicit cache dtypes, including the FP8 index + scale layout, override
  the generic “default KV bytes / element” fallback.

These rules are formula inputs, not measured values. GPU measurements are
component-level validation evidence: weight bytes, KV bytes/token, state
bytes/request, draft bytes/token, shared pool bytes, resident total, and
communication bytes must be reconciled separately.

The archived runtime shape replay is deterministic and read-only:
`node scripts/evidence/runtime/reconcile-accounting.mjs` checks the SGLang
speculative scratch total and the vLLM DSA/MLA per-token ledger against the
H20/A100 capture metadata. It does not start a container and does not turn
measured resident bytes into a frontend constant.

### Pinned source audit (2026-09-21 UTC / 2026-09-22 Asia/Shanghai)

Upstream files were read locally and the DSpark allocation anchors were also
retrieved from the pinned GitHub raw URLs. No sibling repository was modified.

| Project / revision | Source anchor | Formula implication |
|---|---|---|
| vLLM `8ec00ec2bc2d294b742f7814041d3a3043e0a90e` | `vllm/models/deepseek_v41/nvidia/dspark.py:310-313`, `vllm/models/deepseek_v4/nvidia/dspark.py:383-386` | Each draft layer names its own SWA cache; no unconditional target/draft storage alias |
| same | `vllm/v1/attention/backends/mla/sparse_swa.py:114-139` | SlidingWindowMLASpec has dtype, window, block/padding and packed-record inputs |
| same | `vllm/model_executor/layers/mamba/mamba_utils.py:98-146` | GDN and KDA have distinct auto dtype semantics |
| same | `vllm/model_executor/models/config.py:804-824` | Current Qwen3.5 path honors explicit HF mamba_ssm_dtype; older installed versions may differ |
| same | `vllm/v1/kv_cache_interface.py` (`tokens_per_state`), GLM5-Next indexer KV spec in the H20 capture | k-pool stores one index entry per pool, so index growth is divided by `index_kpool` |
| SGLang `11ecdbf39ff90308b67f057b9ab4c2402425afa6` | `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py:430-443` | Allocates draft worker memory rather than proving shared storage |
| same | `python/sglang/srt/model_executor/pool_configurator.py:1404-1434` | Unified SWA has request-scoped target/draft rings |
| same | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:1037-1042` | Shared full-to-SWA mapping is not the same as sharing KV tensors |
| same | `python/sglang/srt/configs/mamba_utils.py:47-81` | SSM defaults FP32 with config/env overrides |
| same | `python/sglang/srt/mem_cache/memory_pool.py:125-140,755-863` | GDN unique conv backing storage; KDA/tree dense fallback; SSM snapshots have a sentinel row |
| same | `python/sglang/srt/mem_cache/memory_pool.py:1114-1131` | `_NON_TRANSFER_STATE_FIELDS` excludes intermediate SSM/conv scratch from PD transfer |
| same | `python/sglang/srt/mem_cache/kv_cache_configurator.py:1123-1160,2316-2347` | PD prefill skips verify; effective request capacity is capped by token/mamba pools |

Reproducible raw-source templates:

```text
https://raw.githubusercontent.com/vllm-project/vllm/8ec00ec2bc2d294b742f7814041d3a3043e0a90e/{source-path}
https://raw.githubusercontent.com/sgl-project/sglang/11ecdbf39ff90308b67f057b9ab4c2402425afa6/{source-path}
```

**Not proven by this change:** page rounding/reserve slots, speculative
verification headroom, the active CUDA/backend cache packing, compressor
state allocation, CUDA graph/workspace, vLLM speculative cache-group reserve,
ReplaySSM scratch, or per-tensor draft weight attribution.
They must be reconciled against the exact runtime version/config. Existing H20
reports are historical evidence for these runtime-only boundaries, not a claim
that the theoretical ledger includes their preallocated bytes.
No measured value or empirical correction factor is shipped to the frontend.

The subsequent H20/A100 validation establishes why “uncompressed row bound”
above is only a per-logical-window bound, **not an upper bound on the physical
preallocated pool**: the tested DSpark draft uses 34,594,560 bytes/rank versus
393,216 bytes of logical window. Do not silently mark Fit as runtime-certified.
The new report also distinguishes SGLang's uncompressed DSA capacity reserve
from vLLM's `tokens_per_state=4` index growth. The latter is now represented by
the vLLM profile rule above; SGLang's page reserve remains runtime-unknown.
