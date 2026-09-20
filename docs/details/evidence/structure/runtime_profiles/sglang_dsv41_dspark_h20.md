# DeepSeek-V4.1-Flash + DSpark 投机 —— H20 fp8 前向端到端（最后一个 fp8-only builder 运行时收口）

> 复现（2026-09-20，本人现跑非复用旧服务）：H20 `10.98.95.16`（8×H20-3e SM90）容器 `dsv41_zzj_deploy`
> （`lmsysorg/sglang:dev-dsv41`，含 dsv41 支持）。模型 `/ssd1/models/DeepSeek-V4.1-Flash-Attn-W8A8-MoE-W4A8-INT8-Dynamic`。
> `python -m sglang.launch_server --tp-size 8 --ep-size 8 --attention-backend dsv4 --kv-cache-dtype fp8_e4m3
> --speculative-algorithm DSPARK --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
> --speculative-dspark-block-size 5 --moe-runner-backend marlin --mem-fraction-static 0.6 --context-length 8192`。

## 结论：`assembleDeepseekV41`(dsv4) 运行时前向 + DSpark 在 H20 跑通（此前 ❌ 未验 fp8→H20）

- **fp8 前向端到端**：`Using DeepseekV4AttnBackend for dsv4 attention backend (CUDA)`、MoE `CompressedTensorsWNA16MarlinMoEMethod`
  （W4A8 marlin）、kv `fp8_e4m3`；`The server is fired up and ready to roll!`；`max_total_num_tokens=4,451,072`（tp8/ep8，ctx 8192）。
- **DSpark 投机跑通**：`speculative_algorithm=DSPARK`（draft 权重打包在 target ckpt）、target verify CUDA graph `num_tokens_per_req=6`
  + draft verify `num_tokens_per_req=5` 全部 capture 成功；`/generate` 端到端出正确文本（"The capital of France is Paris."）。
  → **HB1（V4/V4.1 fp8 dense 前向）+ HB6（DSpark 投机）在 H20 运行时验证通过**（A100 SM80 无 fp8 张量核，此项此前留 H20）。
- **EP 行为坐实**（frontend_problem_inventory [B]）：`Expert parallelism keeps only a slice of the routed experts on each rank
  ... Shared experts fusion optimization is disabled` —— EP 开时 SGLang **不融合 shared expert**（每 rank 只留 routed 分片），
  与 vLLM「shared expert 独立 MLP」一致 → 该 [B] 分叉在 EP 场景两框架收敛。每卡权重+KV 载入后 avail≈46.5 GB（mem-fraction 0.6 of 140GB）。

## 边界（诚实登记，KV 数值口径）

- 本次是 **W8A8(attn)/W4A8-INT8-Dynamic(MoE)** 量化 build，`enable_deepseek_v4_fp4_indexer=False` → KV/index 走 **fp8 口径**，
  footprint 大于 MSV 建模的 **fp4 设计口径（890 B/token，静态已精确闭合）**。这是「前端对齐模型 fp4 设计 vs 本 build 未启 fp4」的
  **参考基准差异**（见 `../../memory/cache_dtype_audit.md` dsv4 澄清），非前端错。**MSV 890 的运行时数值对拍需 fp4-indexer 启用的 build**（留后续）。
- engram（层 1/14）命中/吞吐本次未单列（W8A8 量化 build + DSpark 已覆盖 fp8 前向与投机；engram 行为量化需专用探针，留后续）。
- 本轮为现跑取证（GPU 现从 0 起、跑后清零），非复用主机上既有 dsv41 部署。

## 收口

**至此 11 个 builder 全部有 H20/A100 运行时证据**：A100 上 9 个（真机/减层 cache 口径）+ H20 上补的 2 个 fp8-only（dsv4/deepseek_v41 前向 + DSpark）。
`builder_coverage.md` 的 `assembleDeepseekV4/assembleDeepseekV41` 由「❌ 未验(fp8)→H20」升级为「✅ H20 fp8 前向 + DSpark」。
