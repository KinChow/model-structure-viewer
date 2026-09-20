# SGLang 有效宽度 / KV / attention 执行 profile（Qwen3-0.6B, A100-80GB, TP=1）

> 口径范围（用户指定）：仅 SGLang、仅本地 `Qwen3-0.6B`；不跑 vLLM、不跑其它模型。
> 因此"跨框架 / 多轴有效宽度对照"退化为**单框架 TP=1 稠密**场景——
> TP=1 下模型不做张量并行切分，单卡有效宽度 == config 宽度（退化恒等）。
> 本文档验证的是：真实 runtime 的宽度/KV/attention 执行口径与 MSV 结构口径一致。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB（GPU 0，81920 MiB） |
| 框架 | SGLang（源码安装于 `/sgl-workspace/sglang`） |
| 模型 | `/ssd2/models/Qwen/Qwen3-0.6B`（qwen3, Qwen3ForCausalLM） |
| dtype | bfloat16（auto） |
| TP / attention backend | tp_size=1 / flashinfer |
| mem_fraction_static | 0.8 |
| max_total_num_tokens (KV pool) | 578608 |
| chunked_prefill_size / max_prefill_tokens | 8192 / 16384 |

启动命令（源码 `configs/qwen3_asr.py:167-168` 已用 `exist_ok=True` 修复 AutoConfig 重复注册，见 ../cost/bench_vs_roofline.md 说明）：

```bash
python -m sglang.launch_server \
  --model-path /ssd2/models/Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 30000 \
  --tp-size 1 --mem-fraction-static 0.8
```

就绪校验：`/get_model_info`、`/get_server_info` 返回上表字段（原始响应由该端点重取）。

## 有效宽度（config 口径，TP=1 恒等）

| 结构 | config | 单卡有效（TP=1） | MSV 结构口径 |
| --- | --- | --- | --- |
| hidden_size | 1024 | 1024 | 1024 |
| q 投影输出 (n_heads×head_dim) | 16×128=2048 | 2048 | 2048 |
| kv 投影输出 (n_kv×head_dim) | 8×128=1024 | 1024 | 1024 |
| GQA 分组 (q/kv) | 16/8 = 2:1 | 2:1 | 2:1 |
| FFN intermediate | 3072 | 3072 | 3072 |
| layers | 28 | 28 | 28 |

TP=1 下无 all-reduce/列切分，每卡承载全宽——"有效宽度"与 config 完全一致，
这是多卡有效宽度对照的**退化基线**（无切分即无宽度折减）。MSV 前端
`buildStructureFromConfig` 对 qwen3 产出的 q/kv/ffn 宽度与上表逐项相等。

## KV cache 口径（MSV byte 模型 vs runtime）

MSV 每 token KV 字节 = `L × n_kv × head_dim × 2(K+V) × dbytes`
= 28 × 8 × 128 × 2 × 2 = **114688 B = 112 KiB**。

runtime 交叉验证：SGLang KV 池 `max_total_num_tokens = 578608`，
按 mem_fraction_static=0.8 分配。若 KV 池物理占用
= 578608 × 114688 B = **66.36 GB ≈ 61.8 GiB**，与 80GB×0.8 扣除
权重(1.19GB)+激活/图缓存后的余量吻合。→ MSV 的每 token KV 口径与
runtime 完全一致（GQA 下按 n_kv=8 而非 n_q=16 计，正确）。

## Attention 执行

- backend = flashinfer（在机确认，见 `/get_server_info`）；
- GQA：16 query head 共享 8 组 KV（每组 2 个 q head），KV cache 只存 8 头，
  与 MSV 结构中 k_proj/v_proj 输出宽 1024（=8×128）一致；
- flash 类 kernel 在 prefill 融合 N×N scores、不落 HBM（口径证据见
  `../cost/flash_kernel_caliber.md`），本模型 prefill S=512 亦适用。

## 结论

TP=1 稠密场景下，SGLang runtime 的有效宽度、GQA 分组、每 token KV 字节
与 MSV 结构/成本口径逐项一致。跨框架（vLLM）与多卡（TP>1）有效宽度对照
不在本次范围（用户明确只跑 SGLang + Qwen3-0.6B）。
