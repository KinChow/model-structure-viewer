import assert from "node:assert/strict";
import test from "node:test";

import { childRepeatMultiplier, walkStructure } from "../traverse.js";

test("walkStructure applies a parent repeat to ordinary children", () => {
  const rows = [];
  walkStructure({ repeat: 3, children: [{ children: [] }] }, (row) => rows.push(row));

  assert.equal(rows[1].path, "root.0");
  assert.equal(rows[1].multiplier, 3);
});

test("explicit child repeats replace an informational parent repeat", () => {
  const rows = [];
  walkStructure({ repeat: 4, children: [{ repeat: 3, children: [{ children: [] }] }] }, (row) => rows.push(row));

  assert.equal(rows[1].multiplier, 1);
  assert.equal(rows[2].multiplier, 3);
  assert.equal(childRepeatMultiplier({ repeat: 4 }, 2, { repeatHandled: true }), 2);
});

test("walkStructure consumes Graph IR nodes and preserves repeat multipliers", () => {
  const rows = [];
  walkStructure({ children: [] }, ({ path, multiplier }) => rows.push({ path, multiplier }), {
    version: 2,
    schema_version: 2,
    root_id: "root",
    nodes: [
      { id: "root", module_id: "model", parent_id: null, order: 0, type: "model", repeat: null },
      { id: "root.0", module_id: "decoder.layers", parent_id: "root", order: 0, type: "layer", repeat: 4 },
      { id: "root.0.0", module_id: "decoder.layers.attention", parent_id: "root.0", order: 0, type: "attention", repeat: null },
    ],
    edges: [],
  });
  assert.deepEqual(rows.map(({ path, multiplier }) => [path, multiplier]), [["root", 1], ["root.0", 1], ["root.0.0", 4]]);
});
