import { moduleSpec } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function lmHeadModule(id = "lm_head", normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.hidden, shapes.logits) : {};
  return moduleSpec(id, "lm head", "output", { class: "Linear", vocab_size: normalized?.vocabSize, ...flow }, [
    operatorSpec(`${id}.linear`, "output projection", "linear", flow),
  ]);
}
