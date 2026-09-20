# A2 kernel 口径实测（flash-attention scores/probs 是否落 HBM）

> 在机：A100-SXM4-80GB / CUDA 13.0 / torch 2.13.0+cu130 / ncu（GPU 计数器可采）。
> 被测 kernel：`pytorch_flash::flash_fwd_kernel`（torch SDPA FLASH 后端，bf16, causal）。
> 合成张量（无需权重），代表 MHA 形状；ncu `--kernel-name regex:flash_fwd_kernel --launch-count 1`。

## 实测 vs MSV A2 理论口径

MSV A2 假设：把 attention 的 scores/probs 中间量按**理论上限**计（物化 scores+probs、读写 ~4×）。
flash kernel 的真实行为：分块在 SRAM 融合单遍，**N×N scores 不落 HBM**，HBM 仅搬 Q/K/V/O。

| 形状 (B,H,S,D) | 实测 DRAM 读+写 | 理论 Q/K/V/O | 理论 A2 causal 4× 物化 | 实测 / A2-4× |
|---|---|---|---|---|
| 1,32,4096,128 | 148.0 + 44.1 = **192 MiB** | 128 MiB | 2048 MiB | **≈ 1/10.7** |
| 1,32,8192,128 | 388.8 + 94.5 = **483 MiB** | 256 MiB | 8192 MiB | **≈ 1/17** |

## 判定

- **实证结论**：flash kernel 的 HBM 访存与 **Q/K/V/O（O(H·S·D)）同阶**，而非 scores 的 O(H·S²)；
  N×N scores/probs 从不写回 HBM。MSV A2 的"4× 物化"是**保守理论上限**，与真实 flash kernel 差约
  **1 个数量级**，且**随 S 增大而扩大**（实测∝S，A2-4×∝S²：S 4096→8192，比值 10.7×→17×）。
- **处置（保持理论口径 + 标注 kernel 级实证）**：MSV 成本模型延续"理论/下界"口径（本就标注为估计/上界），
  A2 不改公式；在 `cost_counts.md` A2 假设旁标注本实证——**flash kernel 下 scores/probs 不落 HBM，
  真实 attention HBM ≈ Q/K/V/O；A2 的物化上限仅作 roofline 上界，长上下文下显著高估**。
  稀疏变体（MLA/DSA/dsv4 top-k）只会进一步降低有效 KV/score 访存，同向不反向，故上界结论仍成立。

## 复现

```bash
# 理论参考 + 一次 flash SDPA
python3 scripts/evidence/cost/flash_kernel_attn.py 4096
# ncu 采 flash_fwd_kernel 的 DRAM 字节
ncu --kernel-name "regex:flash_fwd_kernel" --launch-count 1 \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum python3 scripts/evidence/cost/flash_kernel_attn.py 4096
```
（`scripts/evidence/cost/flash_kernel_attn.py`：bf16 causal SDPA FLASH 后端，形状 B=1 H=32 S∈{4096,8192} D=128。）
