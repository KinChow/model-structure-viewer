# Qwen3.5 GDN（Qwen3_5MoeForCausalLM）H20 运行时 —— KV 精确 / ssm fp32 方向确认 / GDN state-shape 残差（开）

> 复现：H20 `10.98.95.16`（8×H20-3e SM90）容器 `sglang-dev-20260918-20518d85`，减层件 `/ssd2/models/_reduced/qwen3_5_reduced`
> （8 层：linear×6 + full×2，`mamba_ssm_dtype=float32`，16 experts）；`python -m sglang.launch_server --load-format dummy
> --attention-backend fa3 --mem-fraction-static 0.3 --tp 1`。原始 dump 不入库，跑脚本重生。

## 真机（server fired up；SM90 GDN kernel）

- **GDN kernel 在 H20 跑通**：`FlashInfer GDN kernels loaded`、`decode=TritonGDNKernel / extend=FlashInferGDNKernel`、
  `fused GDN decode QKVZ/BA unpack + indexed Conv1D`；端到端出 token。
- `Mamba Cache is allocated. max_mamba_cache_size: 4, conv_state size: 0.00GB, ssm_state size: 0.06GB`
- `KV Cache is allocated. dtype: torch.bfloat16, #tokens: 10,097,100, K size: 19.26 GB, V size: 19.26 GB`

## 对账 MSV（本地 `linearStateBytesPerSequence` / `kvBytesPerToken`，post-fix）

| 量 | 真机 | MSV | 判定 |
|---|---|---|---|
| GQA KV / token（2 full 层） | K 19.26GB/10,097,100 = **2048 B**（K）→ K+V **4096 B/token** | `kvBytesPerToken=4096` | ✅ **0.0%**（2·2层·kv_heads2·head_dim256·2B） |
| 线性 recurrent state dtype | ssm 0.06GB/4 slots ≈ **14.8–16.1 MB/seq**（fp32 方向） | fp32 recurrent 12.0 MiB（bf16 仅 6.0 MiB） | ✅ **fp32 方向确认**（Bug2：远超 bf16 口径） |
| 线性 state 总量/seq | ssm ≈ 15 MB + conv≈0 | `linearStateBytesPerSequence=12.28 MiB`（conv bf16 24,576×2 + rec fp32 524,288×4，×6 层） | 🟡 **GDN state-shape 残差 ~1.2–1.3×（开）** |

## 结论

- **KV（GQA）与 ssm fp32 方向**：MSV 与真机一致，Bug2（ssm 按 fp32）方向坐实（真机 ssm ≈15MB ≫ bf16 6MB）。
- **开项（诚实登记，未强凑）**：MSV 的 GDN recurrent state 用 KDA 式 `num_value_heads·value_dim·key_dim=524,288 elem/层`，
  真机 ssm_state 每层每 slot 元素数偏大 ~1.2–1.3×（0.06GB 为 2 位有效数字，粗口径；需精确 ssm 字节定位）。
  根因待查：Qwen3.5 GatedDeltaNet 的 state shape 可能含 MSV 未建的分量（如额外 per-head 归一/衰减 state），非 dtype 问题。
  **处置：不为凑数改前端公式；下一步取精确 ssm 字节（SGLang GDN state 形状源码 / 放大 mem 使 ssm 位数更细）后再定夺是否修 `linearStateResidentDecl` 的 GDN 分支。**
