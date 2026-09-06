import assert from "node:assert/strict";
import test from "node:test";
import {
  formatBytes,
  formatCount,
  formatMacs,
  formatMetric,
  formatQuantity,
  formatRate,
  formatSeconds,
} from "./formatters.js";

test("shared count formatter preserves summary and inspector precision", () => {
  assert.equal(formatCount(1_234_567), "1.2M");
  assert.equal(formatCount(1_234_567, { largeDigits: 2 }), "1.23M");
  assert.equal(formatCount(999), "999");
  assert.equal(formatCount(Number.NaN), null);
});

test("shared byte formatter supports compact diagram KiB output", () => {
  assert.equal(formatBytes(1_536), "1536 B");
  assert.equal(formatBytes(1_536, { includeKib: true }), "1.5 KiB");
  assert.equal(formatBytes(2 * 1024 ** 2), "2.0 MiB");
  assert.equal(formatBytes(Number.POSITIVE_INFINITY), "-");
});

test("shared cost formatters keep units stable across panels", () => {
  assert.equal(formatMetric(1.2e9), "1.20G");
  assert.equal(formatQuantity(1.2e9), "1.20 G");
  assert.equal(formatMacs(1.2e6), "1.2 M");
  assert.equal(formatSeconds(0.25), "250.00 ms");
  assert.equal(formatRate(2e9), "2.0 GB/s");
});
