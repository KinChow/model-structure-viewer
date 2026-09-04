import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function rmsNormModule(id, name = "RMSNorm", normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.hidden, shapes.hidden) : {};
  const dims = normalized ? tensorDims(normalized) : null;
  return withShapeDims(moduleSpec(id, name, "normalization", { class: "RMSNorm", ...flow }, [
    operatorSpec(`${id}.rmsnorm`, "RMSNorm", "rmsnorm", flow, { input: dims?.hidden, output: dims?.hidden }),
  ]), dims?.hidden, dims?.hidden);
}
