import assert from "node:assert/strict";
import test from "node:test";

import { childRepeatMultiplier, walkStructure } from "../traverse.js";

// P7（步骤 7）：walkStructure 签名收窄为 walkStructure(graph, visit)——
// tree root 入参与树回退分支退役；以下夹具全部为 Graph IR。
const graph = (nodes) => ({ version: 2, schema_version: 2, root_id: "root", nodes, edges: [] });

test("walkStructure applies a parent repeat to ordinary children", () => {
  const rows = [];
  walkStructure(graph([
    { id: "root", parent_id: null, order: 0, type: "model", repeat: null },
    { id: "root.0", parent_id: "root", order: 0, type: "layer", repeat: 3 },
    { id: "root.0.0", parent_id: "root.0", order: 0, type: "operator", repeat: null },
  ]), (row) => rows.push(row));

  assert.equal(rows[2].path, "root.0.0");
  assert.equal(rows[2].multiplier, 3);
});

test("explicit child repeats replace an informational parent repeat", () => {
  const rows = [];
  walkStructure(graph([
    { id: "root", parent_id: null, order: 0, type: "model", repeat: null },
    { id: "root.0", parent_id: "root", order: 0, type: "layer", repeat: 4 },
    { id: "root.0.0", parent_id: "root.0", order: 0, type: "layer", repeat: 3 },
    { id: "root.0.0.0", parent_id: "root.0.0", order: 0, type: "operator", repeat: null },
  ]), (row) => rows.push(row));

  assert.equal(rows[1].multiplier, 1);
  assert.equal(rows[3].multiplier, 3);
  assert.equal(childRepeatMultiplier({ repeat: 4 }, 2, { repeatHandled: true }), 2);
});

test("residentRepeat keeps MTP repeat=0 resident", () => {
  const rows = [];
  walkStructure(graph([
    { id: "root", parent_id: null, order: 0, type: "model", repeat: null },
    { id: "root.mtp", parent_id: "root", order: 0, type: "mtp", repeat: 0, attributes: { modules: 1 } },
    { id: "root.mtp.eh_proj", parent_id: "root.mtp", order: 0, type: "operator", repeat: null },
  ]), (row) => rows.push(row));
  assert.equal(rows[2].multiplier, 0);
  assert.equal(rows[2].resident, 1);
});

test("residentRepeat multiplies a template MTP, not expanded DSpark stages", () => {
  const template = [];
  walkStructure(graph([
    { id: "root", parent_id: null, order: 0, type: "model", repeat: null },
    { id: "root.mtp", parent_id: "root", order: 0, type: "mtp", repeat: 0, attributes: { modules: 3 } },
    { id: "root.mtp.layer", parent_id: "root.mtp", order: 0, type: "decoder", repeat: null },
  ]), (row) => template.push(row));
  assert.equal(template[2].multiplier, 0);
  assert.equal(template[2].resident, 3);

  const expanded = [];
  walkStructure(graph([
    { id: "root", parent_id: null, order: 0, type: "model", repeat: null },
    { id: "root.mtp", parent_id: "root", order: 0, type: "dspark", repeat: 0, attributes: { modules: 3 } },
    { id: "root.mtp.0", parent_id: "root.mtp", order: 0, type: "decoder", repeat: null },
    { id: "root.mtp.1", parent_id: "root.mtp", order: 1, type: "decoder", repeat: null },
    { id: "root.mtp.2", parent_id: "root.mtp", order: 2, type: "decoder", repeat: null },
  ]), (row) => expanded.push(row));
  assert.equal(expanded[2].resident, 1);
  assert.equal(expanded[3].resident, 1);
  assert.equal(expanded[4].resident, 1);
});

test("walkStructure consumes Graph IR nodes and preserves repeat multipliers", () => {
  const rows = [];
  walkStructure(graph([
    { id: "root", module_id: "model", parent_id: null, order: 0, type: "model", repeat: null },
    { id: "root.0", module_id: "decoder.layers", parent_id: "root", order: 0, type: "layer", repeat: 4 },
    { id: "root.0.0", module_id: "decoder.layers.attention", parent_id: "root.0", order: 0, type: "attention", repeat: null },
  ]), ({ path, multiplier }) => rows.push({ path, multiplier }));
  assert.deepEqual(rows.map(({ path, multiplier }) => [path, multiplier]), [["root", 1], ["root.0", 1], ["root.0.0", 4]]);
});
