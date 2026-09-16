import assert from "node:assert/strict";
import test from "node:test";
import { foldedLayerName } from "./foldedLayerName.js";

test("单层折叠名保留单个层号（对标 PyTorch ModuleList 单块 repr）", () => {
  assert.equal(foldedLayerName(2, 2, "DecoderLayer"), "2 (DecoderLayer)");
  assert.equal(foldedLayerName(0, 0, "VisionLayer"), "0 (VisionLayer)");
});

test("多层折叠名用区间，避免只显示起始层号", () => {
  assert.equal(foldedLayerName(0, 1, "DecoderLayer"), "0\u20131 (DecoderLayer)");
  assert.equal(foldedLayerName(3, 59, "DecoderLayer"), "3\u201359 (DecoderLayer)");
  assert.equal(foldedLayerName(0, 31, "VisionLayer"), "0\u201331 (VisionLayer)");
});
