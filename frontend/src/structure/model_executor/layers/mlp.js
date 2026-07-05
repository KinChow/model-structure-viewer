import { moduleSpec } from "./base.js";
import { mlpOperatorSpecs } from "../ops/index.js";

export function mlpModule(id, normalized) {
  return moduleSpec(
    id,
    "MLP",
    "mlp",
    {
      class: "MLP",
      hidden_size: normalized.hiddenSize,
    },
    mlpOperatorSpecs(id),
  );
}
