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
