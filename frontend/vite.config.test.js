import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";

import { isPathInside } from "./vite.config.js";

test("static model assets stay inside the configured root", () => {
  const root = path.resolve("/workspace/models");

  assert.equal(isPathInside(root, path.join(root, "catalog.json")), true);
  assert.equal(isPathInside(root, path.join(root, "provider", "config.json")), true);
  assert.equal(isPathInside(root, root), false);
  assert.equal(isPathInside(root, path.resolve(root, "..", "models-copy", "config.json")), false);
  assert.equal(isPathInside(root, path.resolve(root, "..", "config.json")), false);
});
