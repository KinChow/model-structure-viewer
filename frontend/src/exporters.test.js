import assert from "node:assert/strict";
import test from "node:test";
import { exportStructure } from "./exporters.js";

const structure = {
  summary: { strategy: "frontend-architecture-template" },
  source: {},
  root: {
    id: "model",
    name: "Model",
    type: "model",
    attributes: {},
    children: [
      {
        id: "decoder",
        name: "Decoder Layers",
        type: "decoder",
        repeat: 2,
        attributes: {},
        children: [],
      },
    ],
  },
};

test("exports structure as JSON on the frontend", () => {
  const text = exportStructure(structure, "json");
  assert.equal(JSON.parse(text).root.name, "Model");
});

test("exports structure as Mermaid on the frontend", () => {
  const text = exportStructure(structure, "mermaid");
  assert.match(text, /^flowchart TD/);
  assert.match(text, /Decoder Layers x2/);
});

test("exports structure as DOT on the frontend", () => {
  const text = exportStructure(structure, "dot");
  assert.match(text, /^digraph ModelStructure/);
  assert.match(text, /rankdir=LR/);
});
