import { moduleSpec } from "./base.js";
import { mlpOperatorSpecs } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function mlpModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  return moduleSpec(
    id,
    "MLP",
    "mlp",
    {
      class: "MLP",
      hidden_size: normalized.hiddenSize,
      intermediate_size: normalized.intermediateSize,
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        intermediate_shape: shapes.intermediate,
      }),
    },
    mlpOperatorSpecs(id, normalized),
  );
}
