# 证据复现脚本（scripts/evidence）

与 `docs/details/evidence/` 的四维一一平行。每个探针**只做复现**：构造减层/真实件、跑真值、对前端 spec，产出结论回填对应 doc。

## 运行约定

- **路径全走环境变量，无机器硬编码**：
  - `SGLANG_MODELS_DIR` —— 本地 SGLang `python/sglang/srt/models` 源目录
  - `MSV_PROBE_MODEL` —— 探针用的 HF 模型路径或 id（如 `Qwen/Qwen3-0.6B`）
  - `V41_INFER_DIR` —— DeepSeek-V4.1-Flash 参考栈 `inference/` 目录（含 model.py）
  - `MODELS` —— 本地权重/减层件根（`$MODELS/_reduced/...`）
- 每个脚本头部 docstring 必含：目的 / 用法（含所需 env）/ 依赖 / **回填哪份 doc**。
- 输出确定性；原始 dump 不入库，跑脚本即重生。

## 目录

| 维度 | 脚本 | 依赖 | 回填 doc |
|---|---|---|---|
| structure/ | reconcile_reduced.py、deepseek_v41_module_tree.py、v4_proxy_backend.py、v4_proxy_frontend.mjs | transformers/torch、前端 | `evidence/structure/*` |
| memory/ | deepseek_v41_kv_shapes.py | 参考栈（V41_INFER_DIR） | `evidence/memory/*` |
| cost/ | operator_cost 系列、roofline、stage_vectors、flash_kernel、moe 系列 | torch FlopCounter、ncu | `evidence/cost/*` |
| parallelism/ | alltoall_bench.py、build_ep_ckpt.py、nsys_allreduce.py、allreduce_bench.py、rs_ag_bench.py、tp_projection.mjs | 多卡 SGLang、NCCL、nsys | `evidence/parallelism/*` |

- `_fixtures/` —— 共享减层 config（多探针复用）。
- `_lib/` —— 共享工具：减层器（缩层缩维）、named_modules/buffers dump、与前端 spec 的对账比对器。

> 具体脚本清单在迁移收口（Task 3.4）时补全。
