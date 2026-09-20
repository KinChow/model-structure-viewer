# DeepEP 源码重编验证尝试（A100 sm_80 / CUDA 13）——结论：环境依赖链未就绪，留 Hopper 环境

## 动机

之前 DeepEP 在框架层生效但预编译 kernel 报 `layout.cu:128 'named symbol not found'`（`etp_deepep.md`）。
诊断为**预编译 wheel（`sgl-deep-ep 0.1.2` / `deep_ep 2.1.0`）与本机 NVSHMEM/CUDA 符号不匹配**——源码 rebuild 对症。
本机 NVSHMEM 已就位（`nvidia-nvshmem-cu13 3.4.5`），CUDA 13.0 / driver 575.57.08 / A100 sm_80。

## 尝试（全程临时目录 + 符号链接，已清理，未动生产）

从容器 overlay 取 DeepEP 源码（含 `csrc/kernels/{intranode,internode,internode_ll,runtime}.cu`）复制到临时目录，逐层解依赖：

| 步骤 | 处置 | 结果 |
|---|---|---|
| ① Hopper arch 硬编码 | setup.py `TORCH_CUDA_ARCH_LIST '9.0'→'8.0'`；确认 `#define __CUDA_ARCH__ 900` 只在 `#ifdef __CLION_IDE__`（非编译期）；kernels 无 wgmma/tma/mbarrier/cluster 等 Hopper 专属指令 | ✓ 非硬阻塞 |
| ② NVSHMEM 链接布局 | pip NVSHMEM 是拆分布局（`libnvshmem_device.a`+`libnvshmem_host.so.3`+`nvshmem_bootstrap_uid.so.3`），非源码构建的 `libnvshmem.a`；patch `extra_link_args`/`nvcc_dlink` + 补 `libnvshmem_host.so` 符号链接 | ✓ 已适配 |
| ③ **NVSHMEM 头依赖 CCCL** | `nvshmem.h → nvshmem_tensor.h:37` `#include <cuda/std/tuple>` **fatal error: No such file** —— 本机 CUDA 13 工具链**缺 CCCL/libcu++ 头**；`pip install nvidia-cuda-cccl-cu13` **wheel 构建失败** | ✗ **硬阻塞** |
| ④ torch dlink | device-link 步 `/bin/sh: 1: Bad substitution`（torch cpp_extension 在 dash 下的 dlink 命令模板问题） | 次要，未及处理 |

## 结论（据实）

- **未撞 sm_80 架构墙**：编译停在 `deep_ep.cpp` 的 **NVSHMEM 头依赖 `cuda/std/tuple`（CCCL/libcu++）缺失**，
  而非 kernel 的 Hopper 指令——即在本环境 DeepEP 连编到 kernel 都没到，先卡在**构建依赖链**。
- **根因**：DeepEP(+NVSHMEM) 是 **CUDA-12 / Hopper** 验证过的栈；本机是 **CUDA-13 / Ampere**，nvidia 包拆分且缺
  CCCL 头（`nvidia-cuda-cccl-cu13` 装不上）。继续需手动补 CCCL 头 + 修 dash dlink + 可能的 NVSHMEM 3.4.5 API 差异
  + 之后才轮到 sm_80 kernel 能否过——**开放式工具链补齐**，非快速修复，且与本机 CUDA13 无官方适配。
- **建议**：DeepEP 源码验证放到**匹配的 CUDA-12 Hopper 镜像（H20）**做——与其他 fp8/fp4 项一并留 H20。
- **不影响口径**：all-to-all dispatch **字节口径 `B·T·topk·H·b` 已 NCCL 直测收口**（`scripts/evidence/parallelism/alltoall_bench.py` 比值 1.000，
  与传输后端解耦）；DeepEP 重编只是把前端 DeepEP 路径对到其真实 kernel，属增量、非口径关键。

清理：临时构建目录、pip NVSHMEM lib 里补的符号链接均已删除；生产 `deep_ep 2.1.0` import 不受影响。
