# A100 CUDA Graph and MTP Prefix-Cache Validation

Generated: 2026-09-22T11:00:37+0800

## Boundary
- A100 only: `10.55.87.81`, container `vllm-0920`. H20 was not touched by this agent.
- No third graph-cap run was added.
- No source patch was applied; source files and the upstream stale-warning patch are evidence only.

## Environment
- Host: `gajl-bcc-offline-1323736.gajl.baidu.com`; GPU: A100-SXM4-80GB; driver `575.57.08`.
- Container image ID: `sha256:c3b2649067a2f3d24d0b42bbc8b181fd11dd8d87b9397c4fc143c2016aca53f7`; repo digest: `iregistry.baidu-int.com/hub-official/vllm-openai@sha256:31a59e7704a9c2fcd967b84f649442c7d8bd5884805c734dcdbb7b3794a822b3`.
- vLLM: `0.28.1rc1.dev278+g73029d424`; commit `73029d42441321b631779db3475031f5ec26dd6c`.
- Qwen3.5-4B; no `--enforce-eager`; prefix caching enabled.

## Matrix and runtime graph controls
- Both runs used utilization 0.30, max model length 4096, max sequences 4, max batched tokens 2048, and capture sizes `[1,2,3,4,6,8,12]`.
- Runtime-derived scheduler blocks: MTP-on 544 tokens, MTP-off 528 tokens. Each shared prefix was 3 blocks: 1632 and 1584 tokens respectively.
- Effective graph mode was `FULL_AND_PIECEWISE` in both runs.
- MTP-on native graph pool: 0.11 GiB actual vs 0.16 GiB estimated; available KV 12.88 GiB; GPU KV 141107 tokens.
- MTP-off native graph pool: 0.03 GiB actual vs 0.08 GiB estimated; available KV 10.96 GiB; GPU KV 198948 tokens.
- These graph figures are native pool/capture evidence, not NVML-total subtraction.

## Prefix-cache results
| Run | Request order hit-token deltas | Output validity |
|---|---|---|
| MTP-on | `cold-A 0; repeat-A 0; diff-B 1088; repeat-B 1088; diff-C 1088; cold-unrelated 0; repeat-A-after-unrelated 1088` | 7/7 valid |
| MTP-off | `cold-A 0; repeat-A 1584; diff-B 1584; repeat-B 1584; diff-C 1584; cold-unrelated 0; repeat-A-after-unrelated 1584` | 7/7 valid |

## Warning interpretation
- The old unannotated-eagle/Mamba warning was captured in MTP-on logs.
- It is not proof of zero prefix-cache reuse: the same run measured 1088-token local hits on later requests.
- The first MTP-on repeat-A miss is preserved as a sequence-level observation. It is not relabeled as global failure; a first-hit/retention/boundary effect remains a candidate, while genuine zero reuse is contradicted by later hits.
- H20/A100 differences are version/configuration-confounded because the H20 build is newer.

## Cleanup and artifacts
- All task-owned process groups were checked clean; unrelated historical processes were preserved.
- Final GPU inventory: all 8 A100s at 0 MiB, no compute apps, task port 18122 released.
- Full live artifact paths are listed in the final response.

## Exact memory evidence and limitations

| Metric | MTP-on | MTP-off |
|---|---:|---:|
| Native graph estimate bytes | 176,160,768 | 83,886,080 |
| Actual capture bytes | 121,634,816 | 35,651,584 |
| NVML after start, MiB | 23,085 | 23,057 |
| Available KV bytes | 13,828,452,660 | 11,771,146,548 |
| Prefix-query token delta | 11,514 | 11,178 |
| Prefix-hit token delta | 4,352 | 7,920 |
| Successful requests / generated tokens | 7 / 207 | 7 / 207 |

- `metrics-consistency.json`: every per-request query delta equals the prompt token
  count; `prefix_cache_hits_total`, `prompt_tokens_cached_total`, and
  `prompt_tokens_by_source_total{source="local_cache_hit"}` agree; prompt compute
  plus cached tokens equals total prompt tokens; per-request sums equal final
  minus initial counters; all 14 completion token counts and success counters agree.
- `usage.prompt_tokens_details` is null in responses, so the cached-token claims
  come from metrics, not invented response fields.
- Target actual capture tokens: PIECEWISE `[1,2,3,4,6,8,12]` in both runs;
  FULL `[3,6,9,12]` with MTP and `[1,2,3,4]` without it. MTP additionally captured
  draft-prefill PIECEWISE/FULL and draft-decode FULL `[1,2,3,4]`.
  Raw descriptor lines and exact return-value observations are retained.
- The active runner is MRv2 (`v1/worker/gpu/model_runner.py`). Its native profiler
  in `v1/worker/gpu/cudagraph_utils.py` samples FULL graphs, measures PIECEWISE and
  speculator captures, then extrapolates. Do not substitute the other runner's
  source formula for these observed results.
- Enabling `--cudagraph-metrics` did not produce per-request dispatch/replay
  counters in the collected metrics. Real graph capture is proven; per-request
  replay frequency is not quantified here.
- Native graph figures are not the difference between whole-process/whole-device
  memory totals. Compilation/profiling state also differs between cold/warm starts,
  so the KV or NVML differences are not an isolated MTP memory penalty.
- Runtime source snapshots: all 13 checked Python files byte-match `git show`
  at `73029d42441321b631779db3475031f5ec26dd6c` (`source-verification.json`).
- MTP-on group annotations are false for all four groups, while the coordinator
  fallback exposes eagle group IDs `[0,1,2,3]`. MTP-off has `[]`. Both use Mamba
  `align`, retention interval 0, and no fine-grained partial-hash hits.
- Output validity is a bounded functional smoke, not accuracy proof: 4 requests
  per run reached 48 tokens; B/C include incomplete reasoning. Repeated A
  returned Paris and the unrelated control returned BLUE.
- The first repeat-A miss is sequence evidence. Underregistration, retention,
  or boundary effects remain hypotheses; no internal per-step trace proves which.
- `EngineDeadError` appears during deliberate shutdown only, after all requests;
  both servers exited 0 and namespace-correct own-group cleanup checks passed.
- Initial optional `/server_info` 404 and an intermediate local harness syntax
  error are diagnostic attempts, not framework failures or successful controls.

## Transfer and delivery

The live archive was downloaded through the user's BOS prefix and verified locally:

```text
a100-live-evidence.tar.gz
SHA256 04a29cbbe7fbe7424b9f8c7318f62031d0dabd8211011947a84a3a34140ee401
```

Local root: `/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer/artifacts/framework-runtime-validation/20260922-cudagraph-prefix/a100`

Remote root: `/ssd2/zhouzijian01/msv-validation-20260922-cudagraph-prefix/a100`

SSH status: 2026-09-22 Asia/Shanghai: own SSH session 83399 closed after final artifact upload. Output: Connection to 10.55.87.81 closed; Shared connection to relay.baidu-int.com closed. Unified exec return code 0. vllm-0920 verified running immediately before exit; all 8 GPUs 0 MiB, no own service process-group members, no listener on 18122.
