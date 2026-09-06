import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { tensorDims } from "../dims.js";

export function attentionResidualModule(id, normalized, { layerIndex = 0 } = {}) {
  const dims = tensorDims(normalized);
  const hidden = dims.hidden;
  const projection = [-1, -1, 1];
  const blockSize = normalized.attnResBlockSize;
  const blockIndex = blockSize ? Math.floor(layerIndex / blockSize) : 0;
  const blockWrite = Boolean(blockSize && layerIndex % blockSize === 0);
  const previousBlocks = blockSize ? Math.floor(layerIndex / blockSize) : 0;
  const mlpValidBlocks = previousBlocks + (blockWrite ? 1 : 0);
  return withShapeDims(moduleSpec(
    id,
    "Attention Residual",
    "residual",
    {
      class: "AttentionResidual",
      block_size: blockSize,
      block_index: blockIndex,
      block_write: blockWrite,
      previous_blocks: previousBlocks,
      bank_shape: `[batch, sequence, snapshot blocks=${previousBlocks + (blockWrite ? 1 : 0)}, hidden size=${normalized.hiddenSize}]`,
      ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[batch, sequence, hidden size]" },
    },
    [
      operatorSpec(`${id}.aggregate`, "attention residual aggregation", "attention_residual", {
        input_shape: "[snapshot bank, prefix, batch, sequence, hidden size]",
        output_shape: "[batch, sequence, hidden size]",
        aggregation_points: ["pre_attention", "pre_mlp"],
        block_write: blockWrite,
        previous_blocks: previousBlocks,
      }, { input: hidden, output: hidden }),
      operatorSpec(`${id}.self_attention_res_norm`, "attention residual norm", "rmsnorm", {
        ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[residual states, batch, sequence, hidden size]" },
        aggregation_point: "pre_attention",
        snapshot_write: blockWrite,
        valid_snapshot_blocks: previousBlocks,
      }),
      operatorSpec(`${id}.self_attention_res_proj`, "attention residual projection", "linear", {
        ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[residual states, batch, sequence, 1]" },
        aggregation_point: "pre_attention",
        snapshot_write: blockWrite,
        valid_snapshot_blocks: previousBlocks,
      }, { input: hidden, output: projection }),
      operatorSpec(`${id}.mlp_res_norm`, "MLP residual norm", "rmsnorm", {
        ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[residual states, batch, sequence, hidden size]" },
        aggregation_point: "pre_mlp",
        snapshot_write: false,
        valid_snapshot_blocks: mlpValidBlocks,
      }),
      operatorSpec(`${id}.mlp_res_proj`, "MLP residual projection", "linear", {
        ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[residual states, batch, sequence, 1]" },
        aggregation_point: "pre_mlp",
        snapshot_write: false,
        valid_snapshot_blocks: mlpValidBlocks,
      }, { input: hidden, output: projection }),
    ],
  ), dims.hidden, dims.hidden);
}

export function outputAttentionResidualModule(id, normalized) {
  const dims = tensorDims(normalized);
  const snapshotBlocks = normalized.attnResBlockSize && normalized.layers
    ? Math.ceil(normalized.layers / normalized.attnResBlockSize)
    : undefined;
  return withShapeDims(moduleSpec(
    id,
    "Output Attention Residual",
    "residual",
    {
      class: "OutputAttentionResidual",
      block_size: normalized.attnResBlockSize,
      snapshot_blocks: snapshotBlocks,
      aggregation_point: "output",
    },
    [
      operatorSpec(`${id}.norm`, "output residual norm", "rmsnorm"),
      operatorSpec(`${id}.proj`, "output residual projection", "linear", {}, { input: dims.hidden, output: [-1, -1, 1] }),
    ],
  ), dims.hidden, dims.hidden);
}
