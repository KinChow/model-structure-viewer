# Tier3 运行时（engram/DSpark/真实 KV footprint）——A100 硬件边界

## 结论：A100(SM80) 无 fp8/fp4 张量核，参考栈 fp8 GEMM 无法执行 → Tier3 行为验证需 Hopper/Ada 级 GPU

参考栈自测 `python model.py`（8×A100 之一，`CUDA_LAUNCH_BLOCKING=1`）的确定性失败点：

```
cute/arch/mma_sm89.hpp:88 SM89_16x8x32_F32E4M3E4M3F32_TN::fma(...):
  Assertion `0 && "Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN
  without CUTE_ARCH_MMA_F32_SM89_ENABLED"` failed.
调用链：model.py:198 linear → kernel.py:306 fp8_gemm → CUDALaunch CUDA_ERROR_ASSERT
```

- 失败 kernel 是 **SM89**（Ada：L40/L40S/RTX4090）的 fp8 tensor-core MMA `E4M3×E4M3→F32`。
- 本机是 **A100-SXM4-80GB = SM80（Ampere）**，**无 fp8 MMA**（fp8 张量核自 SM89/Hopper SM90 起；config 的
  `expert_dtype=fp4` 更需 Blackwell SM100）。故参考栈的 fp8 dense GEMM / fp4 expert 路径在 A100 上必然
  device-assert，**与模型大小无关**（减到 1 层同样触发）——**减层兜底不能绕过**，是硬件架构边界。
- 旁证：纯量化 kernel `act_quant`（fp8 量化、逐元素、无 MMA）在 A100 上实测可跑（[256,1024]/[4096,512] OK）；
  仅 **fp8 GEMM**（用 fp8 张量核）不可跑。

## 影响与取舍

- **Tier3 行为指标**（engram 接受/命中、DSpark 投机接受率、真实每 token KV footprint、吞吐/时延）依赖完整
  权重的 fp8/fp4 前向 → **A100 上不可得**。`convert.py` 转 TP8（~500GB 产物）后 `torchrun generate.py` 会在
  同一 fp8 GEMM 处失败，故**不执行**该转换（避免 ~500GB + 数小时后撞同一硬件墙）。
- 需在 **Hopper（H100/H200/H20，SM90）或 Ada（L40S，SM89）** 上重跑本项；届时用完整权重取行为真值。

## A100 上已达成（本次实际收口）

- **减层真实模块树 + KV shape 真值**（构造/读 buffer 不触发 GEMM）：由 `scripts/evidence/memory/deepseek_v41_kv_shapes.py` 重生、
  `deepseek_v41_module_tree.md`——compress_kv_cache 仅 kv_source 层、index k_cache 仅 index_source 层，
  Reuse 层 0 常驻，`csa2_kv_share_ok=true`。
- **前端 CSA2 跨层 KV 共享过计数修复 + 逐层层位对账**（`../memory/deepseek_v41_csa2_kv_bytes.md`）。
- **kernel plumbing 部分验证**：fp8 量化 kernel 在 A100 可跑；fp8/fp4 **MMA** 需 SM89+/SM90（硬件边界，登记）。

## 后续：H20（Hopper SM90）复跑计划（用户 2026-09-18 认领）

用户将在 **H20** 上重试完整推理。硬件对照与预期：

- **fp8 GEMM 解除**：H20 = Hopper(SM90)，有 fp8 张量核 → A100(SM80) 上失败的
  `SM89_16x8x32_F32E4M3E4M3F32_TN`（fp8 MMA）在 H20 上可跑（tilelang 会走 SM90 wgmma）。参考栈自测
  `python model.py` 与 `convert.py`→`torchrun generate.py` 应能前向。
- **fp4 expert 仍是风险**：`float4_e2m1` MMA 是 **Blackwell(SM100)** 才原生；H20 无 fp4 张量核，config 的
  `expert_dtype=fp4` 路径可能需 dequant→bf16/fp8 回退或 emulation。若参考栈 fp4 kernel 无 H20 回退，仍会卡在
  expert GEMM——届时可先减层 / 或临时把 expert 转 fp8 跑通行为验证。
- **环境复现**：项目 `.venv`（Py3.12/torch2.14+cu130/tilelang0.1.8）+ **`apache-tvm-ffi==0.1.8.post2`**
  （0.1.14 与 tilelang0.1.8 py3.12 反射冲突，须降级）。

**H20 上要取的证据 + 待收口差异**：
1. **engram/DSpark 运行时**：engram（层 1/14）命中/接受、DSpark（层 37/38/39、128 草稿专家）投机接受率、显存、吞吐。
2. **真实逐层 KV footprint** → 与前端边际模型对拍，**用于二次校准 V4.1 的 +18.7% 残差**（下）。

## 待收口差异（记录，H20 复跑时校准）

前端边际 + 逐 dtype KV-per-token（spec `dsv4-kv-marginal-bytes`）对官方
「Global KV Cache Per Token」：**V4-Flash 3,440 vs 3,514（−2.1%）**、**V4.1-Flash 1,056 vs 890（+18.7%）**。
V4.1 残差的静态未定项（H20 真值可定夺）：① 压缩 latent 的 rope 尾（qk_rope_head_dim=64）是否高精存储而非 fp4；
② Full 模式（ratio=1）index k_cache 是否全长计入官方口径；③ fp4 scale（E4M3/16、E8M0/32）摊销；④ 官方图取整。
拿到 H20 真实逐层 KV 字节后，据①–④调 `F4` 有效字节 / index 计法，把 V4.1 收敛到 890 容差内。详见
`../memory/deepseek_v41_csa2_kv_bytes.md §追加`。
