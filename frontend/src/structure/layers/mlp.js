import { moduleSpec, withShapeDims } from "./base.js";
import { mlpOperatorSpecs } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { hfNamedClass } from "../archs/index.js";

export function mlpModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    id,
    "MLP",
    "mlp",
    {
      class: hfNamedClass(normalized, "mlpClass", "MLP"),
      hidden_size: normalized.hiddenSize,
      intermediate_size: normalized.intermediateSize,
      dataflow_edges: [["gate_proj", "swiglu"], ["up_proj", "swiglu"], ["swiglu", "down_proj"]],
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        intermediate_shape: shapes.intermediate,
      }),
    },
    mlpOperatorSpecs(id, normalized),
  ), dims.hidden, dims.hidden);
}
