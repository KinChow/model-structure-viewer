import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function lmHeadModule(id = "lm_head", normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.hidden, shapes.logits) : {};
  const dims = normalized ? tensorDims(normalized) : null;
  return withShapeDims(moduleSpec(id, "lm head", "output", { class: "Linear", vocab_size: normalized?.vocabSize, ...flow }, [
    operatorSpec(`${id}.linear`, "output projection", "linear", flow, { input: dims?.hidden, output: dims?.logits }),
  ]), dims?.hidden, dims?.logits);
}
