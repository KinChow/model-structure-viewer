import { moduleSpec } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function rmsNormModule(id, name = "RMSNorm", normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.hidden, shapes.hidden) : {};
  return moduleSpec(id, name, "normalization", { class: "RMSNorm", ...flow }, [
    operatorSpec(`${id}.rmsnorm`, "RMSNorm", "rmsnorm", flow),
  ]);
}
