# MiniMax-M2.7 / GLM-4.7 separate-QKV repair

- 日期：2026-09-24
- 范围：`MiniMaxAI/MiniMax-M2.7`、`zai-org/GLM-4.7`
- Graph IR：保持 `version=2` / `schema_version=2`

## 证据

两条模型的本地 `source-ref.json` 都列出了每层独立的：

- `root.layers.0.self_attn.q_proj`
- `root.layers.0.self_attn.k_proj`
- `root.layers.0.self_attn.v_proj`

MiniMax-M2.7 的随附 Transformers 实现 `models/MiniMaxAI/MiniMax-M2.7/modeling_minimax_m2.py` 也分别构造并调用 `q_proj`、`k_proj`、`v_proj`；GLM-4.7 的 source-ref 使用同一组独立 checkpoint 叶路径。运行时 kernel 是否融合，不改变 checkpoint/架构语义。

## 修复

此前配方把两条模型标为 `fusedQkv`，前端因此显示虚构的 `qkv_proj → qkv_split` 权重节点。现改为 `separateQkvQkNorm`：

- 分别显示 Q、K、V projection，并各自绑定权重容量；
- 保留真实的 Q/K RMSNorm；
- 保留 GQA 的 KV 头数、partial RoPE、SDPA 和输出投影；
- GLM-4.7 的 attention bias 继续从配置透传；
- 没有新增融合权重，也没有升级 Graph IR。

## 验证

机制断言在旧实现上会失败，当前确认：

- MiniMax-M2.7：`q projection / k projection / v projection / Q RMSNorm / K RMSNorm / partial rotary position embedding`；
- GLM-4.7：同样的独立 QKV 和 Q/K norm；
- `source-ref`、配置归一化、builder 声明和物化 Graph IR 的 checkpoint 路径一致。

本修复不代表已经完成两条模型的 GPU kernel 融合、量化 scale traffic 或真实性能验证；这些仍保持 unknown。
