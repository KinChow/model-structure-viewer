import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec, weightMatrixDecl } from "../operators/ops/index.js";
import { tensorDims } from "../config/dims.js";
import { formulaForOperator } from "../operators/formulas/index.js";

// Frozen states, never a mutable bank node with back-edges. Embedding is
// written at layer 0; a completed prefix is appended at 12,24,... .
export function attentionResidualStage(normalized, layerIndex) {
  const size = normalized.attnResBlockSize;
  return { before: Math.ceil(layerIndex / size), after: Math.floor(layerIndex / size) + 1,
    write: layerIndex % size === 0, block: Math.floor(layerIndex / size) };
}

export function residualBankState(id, normalized, snapshots, write = false) {
  const shape = [-1, -1, snapshots, normalized.hiddenSize];
  return operatorSpec(id, write ? "snapshot bank append" : "snapshot bank reference", write ? "attn_res_snapshot" : "identity", {
    checkpoint_module: false, semantic_role: "attnres_depth_state", snapshot_count: snapshots,
    previous_snapshot_count: Math.max(0, snapshots - 1),
    snapshot_write: write, state_lifetime: "current forward depth; not autoregressive KV",
    depth_state_elements_per_token: snapshots * normalized.hiddenSize,
    storage_semantics: "dense token-major bank; reference torch.cat append",
    activation_materialization: write ? "reference_torch_cat" : "reference",
    formula: write ? "bank_out = append(bank_in, prefix_in)" : "bank_out = bank_in",
    explanation: "Depth-history state version. The reference implementation appends a dense snapshot with torch.cat; optimized allocator reuse and transient peak are implementation-specific.",
  }, { input: write ? [-1, -1, snapshots - 1, normalized.hiddenSize] : shape, output: shape });
}

export function attentionResidualModule(id, normalized, { candidates, point, normId, projId, skipped = false }) {
  const hidden = tensorDims(normalized).hidden, H = normalized.hiddenSize;
  const states = [-1, -1, candidates, H], scores = [-1, -1, candidates, 1];
  const formula = formulaForOperator("attention_residual");
  const children = [
    operatorSpec(`${id}.candidates`, "history and prefix candidates", "identity", {
      checkpoint_module: false, candidate_states: candidates, activation_materialization: "unknown",
      formula: "V = concat(bank, prefix)",
    }, { input: hidden, output: states }),
    operatorSpec(normId, "residual score RMSNorm", "rmsnorm", {
      candidate_states: candidates, aggregation_point: point,
    }, { input: states, output: states }),
    operatorSpec(projId, "residual score projection", "linear", {
      weightMatrices: [weightMatrixDecl("tp", { shape: [1, H], split: "output" })],
      candidate_states: candidates, aggregation_point: point,
    }, { input: states, output: scores }),
    operatorSpec(`${id}.probabilities`, "softmax over depth candidates", "softmax", {
      checkpoint_module: false, softmax_axis: "candidate_states", candidate_states: candidates,
      formula: "p = softmax(scores, dim=depth)",
    }, { input: scores, output: scores }),
    operatorSpec(`${id}.weighted_sum`, "weighted depth aggregation", "matmul", {
      checkpoint_module: false, candidate_states: candidates,
      formula: "y = sum_i p_i V_i (unnormalized values)",
    }, { input: states, output: hidden }),
  ];
  for (const child of children) {
    child.attributes.execution_skipped = skipped;
    // These are explanatory steps inside one aggregate. FP32 intermediates
    // need not each materialize in HBM; don't display fabricated allocations.
    child.attributes.activation_materialization = "unknown";
  }
  if (skipped) {
    children.at(-1).operatorId = "identity";
    children.at(-1).name = "first-layer aggregation bypass";
    children.at(-1).attributes.formula = "y = prefix (empty history; no scoring)";
  }
  return withShapeDims(moduleSpec(id, point === "output" ? "Output Attention Residual" : `Attention Residual (${point})`, "residual", {
    class: "AttentionResidual", operator_id: "attention_residual", checkpoint_module: false,
    formula: formula.formula, explanation: formula.explanation, inputs: formula.inputs, outputs: formula.outputs,
    aggregation_point: point, candidate_states: candidates, execution_skipped: skipped,
    activation_compute_dtype: "float32", activation_materialization: "unknown",
    dataflow_edges: skipped ? [[`${id}.candidates`, `${id}.weighted_sum`]]
      : [[`${id}.candidates`, normId], [normId, projId], [projId, `${id}.probabilities`],
        [`${id}.probabilities`, `${id}.weighted_sum`], [`${id}.candidates`, `${id}.weighted_sum`]],
    dataflow_edge_relations: skipped ? [{ from: `${id}.candidates`, to: `${id}.weighted_sum`, label: "first-layer identity; scoring skipped" }]
      : [{ from: `${id}.probabilities`, to: `${id}.weighted_sum`, label: "depth weights" },
        { from: `${id}.candidates`, to: `${id}.weighted_sum`, label: "unnormalized values" }],
  }, children), hidden, hidden);
}

export function outputAttentionResidualModule(id, normalized) {
  return attentionResidualModule(id, normalized, {
    candidates: Math.ceil(normalized.layers / normalized.attnResBlockSize) + 1,
    point: "output", normId: "output_attn_res_norm", projId: "output_attn_res_proj",
  });
}
