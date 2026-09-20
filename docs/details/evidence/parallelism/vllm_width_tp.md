# vLLM 跨框架有效宽度 / TP 折叠 —— 首次真机 vLLM 证据（A100, 2026-09-20）

> 复现：容器 `vllm-0920`（A100 `10.55.87.81`，8×A100-SXM4-80GB SM80），vLLM `0.28.1rc1.dev278+g73029d424`；
> `python3 -m vllm.entrypoints.openai.api_server --model /ssd2/models/Qwen/Qwen3-0.6B --tensor-parallel-size {1,2} --max-model-len 4096 --gpu-memory-utilization 0.4 --enforce-eager`。
> 此前全部并行/宽度真机证据均为 **SGLang 单框架**；本项补上 **vLLM 侧**，与 SGLang + MSV 三方对拍。

## 模型（Qwen3-0.6B，bf16）

num_hidden_layers=28、num_key_value_heads=8、num_attention_heads=16、head_dim=128、kv_cache_dtype=auto→bf16。

## 真机日志（关键行）

| 量 | TP1（单卡） | TP2（每卡） | 比值 | 期望 |
|---|---|---|---|---|
| 每卡权重（`Model loading took`） | 1.12 GiB | 0.57 GiB | ÷1.96 | ÷tp ✓ |
| GPU KV cache（池） | 282,224 tok | 575,600 tok | ×2.04 | ×tp ✓ |
| 每卡 KV 可用显存 | 30.15 GiB | 30.74 GiB | ≈ | ✓ |
| 最大并发 @4096 tok | 68.90× | 140.53× | ×2.04 | ×tp ✓ |

## 与 MSV / SGLang 对账（GQA KV 宽度）

- **KV/token（整模型）= 2·layers·kv_heads·head_dim·2B = 2·28·8·128·2 = 114,688 B ≈ 112 KiB**。
- vLLM TP1 实测：30.15 GiB / 282,224 = **114,712 B/token** → 与 MSV 口径一致（0.02%）。
- 与既有 SGLang 证据（`../memory/sglang_width_kv.md`：Qwen3-0.6B 每 token KV 112 KiB）**逐值一致** → **GQA KV 属两框架一致（frontend_problem_inventory 的 [C] 干净），MSV 预测对两框架都成立**。
- **每卡权重 ÷tp、KV 池 ×tp**：vLLM 与 SGLang（`tp_parallel.md`）同口径，验证 MSV `declaredWeightBytesPerCard` / `kvBytesPerCard`（GQA `min(tp,kv_heads)` 分片）对 vLLM 同样成立。

## 边界

- 本项为 TP=1/2 稠密 GQA；vLLM 的 MoE `EP=TP×DP`（无 moe_tp 轴，与 SGLang 分叉，frontend_problem_inventory [B]）仍待 vLLM MoE 真机（VL3）。
- vLLM build 不打印字面 `# GPU blocks`；权威池口径取 `GPU KV cache size: N tokens`（block size 16 → 17,639 blocks @TP1）。
