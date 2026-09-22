# CUDA Graph / MTP prefix-cache runtime evidence

Date: 2026-09-22

This directory records the bounded real-model validation requested for the
framework-conditioned memory-accounting work. It is evidence, not a serving
benchmark and not a calibration dataset.

## Matrix

| Host | Container | vLLM | Model | CUDA Graph | Result |
|---|---|---|---|---|---|
| H20 `10.98.95.16` | `vllm-0920` | `0.29.1rc1.dev397+ga8d1aa9c9` | Qwen3.5-4B | enabled | MTP on/off and capture-size controls passed |
| A100 `10.55.87.81` | `vllm-0920` | `0.28.1rc1.dev278+g73029d424` | Qwen3.5-4B | enabled | MTP on/off and prefix-cache controls passed |

The container names are equal but the vLLM revisions differ. Results are
therefore per-environment validation, not a hardware-only A/B comparison.

## Main conclusions

- `max_num_batched_tokens`, `max_num_seqs`, `max_model_len`, speculative token
  count, and explicit capture sizes all affect the graph workload and memory
  profile.
- vLLM's native graph-pool estimate and actual captured pool are recorded
  separately; the estimate is not treated as a frontend constant.
- The unannotated MTP/Mamba warning did not imply zero prefix-cache reuse in
  these runs. Both H20 and A100 produced later local prefix-cache hits.
- All task-owned processes were cleaned up and the containers were preserved.

## Evidence packages

- H20: `h20/remote-evidence.tar.gz`, SHA-256 in
  `h20/remote-evidence.tar.gz.remote-sha256.txt`.
- A100: `a100/a100-live-evidence-final.tar.gz`, SHA-256 in
  `a100/a100-live-evidence-final.tar.gz.sha256`.
- Human-readable reports: `h20/summary.md`, `a100/summary.md`.

The expanded remote files are intentionally not duplicated in Git; the final
archives and their manifests are the reproducible evidence boundary.
