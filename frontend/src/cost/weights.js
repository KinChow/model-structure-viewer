// weights.js —— 从 safetensors header 读 checkpoint 真值（零权重下载，KB 级 range read）。
//
// 依据 evolution_design.md §4.2：参数量换算必须用 @huggingface/hub 的 parseSafetensorsMetadata，
// 不手写（子字节量化打包容器宽度、bitsandbytes__ 前缀、exponent-only dtype 等边界都在库里）。
// 建树（skeleton.js）自己写，换算（本文件）用库 —— 两件事分开。

import { parseSafetensorsMetadata } from "@huggingface/hub";

/** 过滤非张量 key（safetensors 元数据 / bitsandbytes 量化状态）。 */
function isTensorKey(name) {
  if (name.startsWith("__")) return false;
  if (name.startsWith("bitsandbytes__")) return false;
  return true;
}

/**
 * 读取一个 HF repo 的 safetensors 元数据，归一化为：
 * - tensors: [{name, dtype, shape}]（跨分片合并，含每张量的真实 dtype/shape）
 * - parameterCount: 逐 dtype 精确参数量（库计算，处理打包容器/排除项）
 * - parameterTotal: 模型级精确总参数量
 *
 * 失败（无 safetensors / gated / 网络）时抛错，由调用方降级为模板路径。
 */
export async function fetchCheckpointTruth({ modelId, revision = "main", fetchImpl = fetch }) {
  const parsed = await parseSafetensorsMetadata({
    repo: { type: "model", name: modelId },
    revision,
    computeParametersCount: true,
    fetch: fetchImpl,
  });

  const headers = parsed.sharded ? Object.values(parsed.headers) : [parsed.header];
  const tensors = [];
  for (const header of headers) {
    for (const [name, info] of Object.entries(header)) {
      if (!isTensorKey(name)) continue;
      tensors.push({ name, dtype: info.dtype, shape: info.shape });
    }
  }
  tensors.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  return {
    tensors,
    parameterCount: parsed.parameterCount ?? null,
    parameterTotal: parsed.parameterTotal ?? null,
  };
}
