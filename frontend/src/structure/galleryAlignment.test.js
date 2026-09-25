import assert from "node:assert/strict";
import test from "node:test";
import { galleryAlignmentForGraph } from "./galleryAlignment.js";

test("gallery alignment summarizes a mixed attention schedule without changing graph facts", () => {
  const result = galleryAlignmentForGraph({
    root_id: "root",
    nodes: [
      { id: "root", type: "model" },
      { id: "root.0", parent_id: "root", type: "decoder" },
      { id: "root.0.0", parent_id: "root.0", type: "attention", repeat: 69, attributes: { attention_kind: "linear" } },
      { id: "root.0.1", parent_id: "root.0", type: "attention", repeat: 24, attributes: { attention_kind: "mla" } },
    ],
  }, "KimiK3ForConditionalGeneration");
  assert.equal(result.decoder_type, "Decoder-only");
  assert.equal(result.attention_mix, "69 KDA + 24 Gated MLA");
});

test("gallery alignment exposes V4.1 and IndexShare features", () => {
  const result = galleryAlignmentForGraph({
    root_id: "root",
    nodes: [
      { id: "root", type: "model" },
      { id: "root.0", parent_id: "root", type: "causal-encoder" },
      { id: "root.1", parent_id: "root", type: "decoder" },
      { id: "root.1.0", parent_id: "root.1", type: "attention", repeat: 3, attributes: { attention_kind: "qsa", index_source_layer: 1 } },
    ],
  }, "DeepseekV41ForCausalLM");
  assert.equal(result.decoder_type, "Causal encoder-decoder");
  assert.equal(result.attention_mix, "3 QSA");
  assert.deepEqual(result.special_features, ["IndexShare", "Sparse attention"]);
});
