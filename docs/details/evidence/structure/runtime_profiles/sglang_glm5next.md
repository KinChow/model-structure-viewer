# glm5_next（GLM-5.3-Flash）DSA 稀疏前向 + hybrid cache 运行时 profile（H20/SM90）

> 复现脚本：`scripts/evidence/structure/glm5_next_reduce.py`（减层器）+ `sglang.launch_server --load-format dummy --tp1`。
> 环境：8× H20-3e（SM90 Hopper）/ CUDA 13 / sglang `0.0.0.dev1+g20518d851` / torch 2.13.0+cu130。原始 dump 不入库，跑脚本重生。

## 目的

A100(SM80) 上 `assembleGlm5Next`（DSA + 线性 KDA hybrid）只能验到 cache 分配、**DSA 稀疏前向 kernel 需 SM90a/SM100f**
（见 `sglang_dsa.md`）。本项在 H20 补上稀疏**前向**端到端，并逐字节核 hybrid 三 cache（kv/state/index）对前端。

## 减层件

`glm5_next_reduce.py`：8 层（DSA 层 [3,7] + KDA 层 [0,1,2,4,5,6]）、16 experts、去 fp8→bf16、保全部 per-head 维度；
`--load-format dummy` 随机权重（验 kernel 路径 + cache 口径，非输出正确性）。

## 结论

- **DSA 稀疏前向在 H20 端到端跑通**：`prefill=flashmla_sparse` / `decode=fa3` / KDA `TritonKDAKernel`；长 prompt 3001 tok
  （> 本轮 `index_topk=2048`）触发稀疏 top-k，`Prefill batch #new-token 3001` 成功出 token。**收口“DSA 稀疏前向留 H20”。**
- **三 cache 元素口径对前端 `assembleGlm5Next` 逐点 0.0%**：KDA state 1,122,304（conv 73,728 + temporal 1,048,576）、
  MLA latent 512、DSA index 128（对 `Glm5NextTextConfig` / `KimiLinearStateShape` / `index_key_cache` 真值）。
- **逐字节挖出前端 bug（Bug 1）**：`dsa_sparse_mla` 把 DSA index 键缓存按 bf16 计（256 B/层），实测 SGLang 存
  **fp8(uint8 1B) + E8M0 尺度(4B/128) = 132 B/层**（1.94×）→ glm5_next/deepseek_v32(9 模型) **KV-per-token 高估 +10.7%**
  （H20 实测 KV 池 73.71 GB / 34,233,856 tok = **2312 B/token**，前端 bf16-index 给 2560）。`dsv4_sparse_mla`
  （`ops/index.js:768-779`）正确传了 index dtype，此分支（`ops/index.js:1146`）漏传。详见 `../memory/cache_dtype_audit.md`。

## 边界

- dummy 随机权重：验 kernel 路径 + cache 分配口径，非输出正确性；全权重忠实前向未做。
- 同一 `flashmla_sparse` kernel 亦解除 `assembleDeepseekV32`(DSA) 的稀疏前向硬件前置（同 kernel）；其减层全前向可同法补（本轮未跑，仅登记）。
- Bug 1 的前端修复须在有 node 的开发机做（`node --test` / `verify:models` / 重生成 golden）。
