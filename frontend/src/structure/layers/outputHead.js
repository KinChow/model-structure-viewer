import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec, weightMatrixDecl } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";

export function lmHeadModule(id = "lm_head", normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.hidden, shapes.logits) : {};
  const dims = normalized ? tensorDims(normalized) : null;
  const tied = Boolean(normalized?.tieWordEmbeddings);
  return withShapeDims(moduleSpec(id, "lm head", "output", { class: "Linear", vocab_size: normalized?.vocabSize, tied_word_embeddings: tied || undefined, ...flow }, [
    operatorSpec(`${id}.linear`, "output projection", "linear", {
      ...flow,
      ...(tied && dims ? {
        weightMatrices: [weightMatrixDecl("vocab", {
          shape: [normalized.vocabSize || 0, normalized.hiddenSize || 0],
          shared: true,
          quantizable: false,
        })],
      } : {}),
    }, { input: dims?.hidden, output: dims?.logits }),
  ], undefined, "output"), dims?.hidden, dims?.logits);
}
