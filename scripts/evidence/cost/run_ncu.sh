#!/usr/bin/env bash
# NV-3 深化 · 逐算子 ncu 采集驱动。对每个 op 跑 bench_op.py，ncu 圈定 cudaProfiler 范围
# 采 DRAM/指令/缓存指标，CSV 存 ncu_raw/<op>.csv。
set -u
cd "$(dirname "$0")"
mkdir -p ncu_raw
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,sm__inst_executed_pipe_tensor.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
lts__t_bytes.sum,l1tex__t_bytes.sum"
for op in gate_proj down_proj o_proj q_proj attention rmsnorm swiglu rope; do
  echo "=== ncu $op ==="
  ncu --profile-from-start off --csv --metrics "$METRICS" \
      python3 bench_op.py "$op" > "ncu_raw/${op}.csv" 2>/dev/null
done
echo "done -> ncu_raw/"
