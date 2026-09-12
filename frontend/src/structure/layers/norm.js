import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { hfNamedClass, recipeNormMode } from "../archs/index.js";

export function rmsNormModule(id, name = "RMSNorm", normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.hidden, shapes.hidden) : {};
  const dims = normalized ? tensorDims(normalized) : null;
  const gemmaStyle = recipeNormMode(normalized) === "gemma_rmsnorm";
  const formulaId = gemmaStyle ? "gemma_rmsnorm" : "rmsnorm";
  const className = hfNamedClass(normalized, "rmsNormClass", "RMSNorm", "RMSNorm");
  return withShapeDims(moduleSpec(id, name, "normalization", { class: className, ...flow }, [
    operatorSpec(`${id}.rmsnorm`, gemmaStyle ? "Gemma RMSNorm" : "RMSNorm", formulaId, flow, { input: dims?.hidden, output: dims?.hidden }),
  ], undefined), dims?.hidden, dims?.hidden);
}
