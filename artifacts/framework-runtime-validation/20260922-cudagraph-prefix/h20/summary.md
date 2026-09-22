# H20 CUDA Graph / MTP Prefix Cache Runtime Validation

- Date: 2026-09-22
- Scope: H20 `10.98.95.16` only; A100 untouched.
- Container: `vllm-0920`, kept running at completion.
- Real model: `Qwen/Qwen3.5-4B` from authorized BOS, not reduced/dummy.

## Environment

- Host: `gajl-bcc-onlinec-com-1577862.gajl.baidu.com`; GPU 1 `NVIDIA H20-3e` UUID `GPU-ac19fc4c-e112-e799-4c48-1e0a76bf91a4`.
- Image ID: `sha256:38e8c0e60293aefc23edee5c074107e44b89d64f26d5399cd2f5956801b6ef24`; repo digest `iregistry.baidu-int.com/hub-official/vllm-openai@sha256:4cbfd34aac145fd1870381c030131c7f868fcad45448f401ecdb5fd4ed020b42`.
- vLLM: `0.29.1rc1.dev397+ga8d1aa9c9`; commit `a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b`.
- Controls: `--gpu-memory-utilization 0.30`, `--max-model-len 4096`, `--max-num-seqs 2`, `--max-num-batched-tokens 2048`, prefix caching on, no `--enforce-eager`, 24 output tokens.

## Results

| Run | Block | Prefix | Warm cached/hit tokens | Graph estimate / actual | Result |
|---|---:|---:|---:|---:|---|
| `mtp-on-cap8` | 544 | 1632 | [1088] / [1088.0] | 193986560 / 150994944 bytes | PASS |
| `mtp-off-cap8` | 528 | 1584 | [1584] / [1584.0] | 57671680 / 37748736 bytes | PASS |
| `mtp-on-cap32` | 544 | 1632 | [1088] / [1088.0] | 214958080 / 155189248 bytes | PASS |
| `mtp-off-cap8-sameprompt` | 528 | 1632 | [1584] / [1584.0] | 57671680 / 37748736 bytes | PASS |

### Prefix hit conclusion

- MTP-on warning was emitted, but it did **not** imply hit=0.
- MTP-on cap=8: cold `cached_tokens=0`; four warm requests, including identical full resend and different suffixes, each had `cached_tokens=1088` and metrics hit delta `1088`.
- Fourth run MTP-off same-prompt: the exact same request JSONs were resent; cold `0`, then four warm requests each had `cached_tokens=1584` and metrics hit delta `1584`.
- Mamba cache mode was `align`; the smaller retained hit than the 1632-token prefix is expected to be interpreted with block/scheduler/cache boundaries, not as a failure.

### CUDA Graph provenance

- Effective mode: `FULL_AND_PIECEWISE`.
- `cap=8` and optional `cap=32` were intentionally small bounded controls; results are not production graph-coverage claims.
- The estimate is from `GPUModelRunner.profile_cudagraph_memory()` and `cudagraph_utils._extrapolate_full_graph_memory()`; source excerpt is saved in `source-provenance.txt`.
- The runtime log separately reports actual pool memory versus estimate; both values are preserved per run in `server.log`, `extracted-summary.json`, and `summary.json`.

## Source and cleanup evidence

- Actual runner: V2 path `vllm.v1.worker.gpu.model_runner.GPUModelRunner`; all four server logs contain `Using V2 Model Runner`. Exact module and SHA are in `actual-runner.txt`; selection source is in `source/v1__worker__gpu_worker.py`.
- Warning/group annotation and coordinator condition: `source-provenance.txt` and `source/v1__core__kv_cache_utils.py` / `source/v1__core__kv_cache_coordinator.py`.
- Framework source SHA recheck: `framework-source-unchanged.txt` (`unchanged: true`).
- All owned harness processes exited; all test ports closed; final NVML showed 0 MiB on all H20 GPUs; `vllm-0920` remained running. Details: `cleanup-confirmation.txt` and each run's `result.json`.

## Artifact contents

- `mtp-on-cap8/`: core MTP-on prefix/cache/graph run.
- `mtp-off-cap8/`: MTP-off control with its naturally derived 528-token block.
- `mtp-on-cap32/`: bounded optional graph-capture-size comparison started before the fourth run.
- `mtp-off-cap8-sameprompt/`: fourth and final authorized run using identical prompt JSONs to MTP-on.
- `probe/`: harness copies and command inputs.
- `source/`: exact container Python source snapshots.
