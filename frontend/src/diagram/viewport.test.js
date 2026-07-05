import assert from "node:assert/strict";
import test from "node:test";

import { fitDiagramViewport, sameDiagramViewport } from "./viewport.js";

test("fitDiagramViewport centers a smaller diagram inside the available viewport", () => {
  const result = fitDiagramViewport({
    contentWidth: 400,
    contentHeight: 200,
    viewportWidth: 1000,
    viewportHeight: 600,
  });

  assert.equal(result.scale, 1);
  assert.equal(result.offsetX, 300);
  assert.equal(result.offsetY, 200);
  assert.equal(result.canvasWidth, 1000);
  assert.equal(result.canvasHeight, 600);
});

test("fitDiagramViewport scales a wide diagram down with balanced padding", () => {
  const result = fitDiagramViewport({
    contentWidth: 1800,
    contentHeight: 600,
    viewportWidth: 900,
    viewportHeight: 500,
    padding: 30,
  });

  assert.ok(Math.abs(result.scale - 0.4667) < 0.0002);
  assert.ok(result.offsetX >= 30);
  assert.ok(result.offsetX < 31);
  assert.equal(result.canvasWidth, 900);
  assert.ok(result.offsetY > 100);
});

test("fitDiagramViewport treats zoom as a multiplier over the fitted scale", () => {
  const fitted = fitDiagramViewport({
    contentWidth: 1800,
    contentHeight: 600,
    viewportWidth: 900,
    viewportHeight: 500,
    padding: 30,
  });
  const zoomed = fitDiagramViewport({
    contentWidth: 1800,
    contentHeight: 600,
    viewportWidth: 900,
    viewportHeight: 500,
    padding: 30,
    zoom: 1.4,
  });

  assert.ok(Math.abs(zoomed.baseScale - fitted.scale) < 0.0002);
  assert.ok(Math.abs(zoomed.scale - fitted.scale * 1.4) < 0.0002);
  assert.ok(zoomed.canvasWidth > fitted.canvasWidth);
  assert.equal(zoomed.canvasHeight, fitted.canvasHeight);
});

test("sameDiagramViewport suppresses duplicate viewport updates", () => {
  const viewport = fitDiagramViewport({
    contentWidth: 400,
    contentHeight: 200,
    viewportWidth: 1000,
    viewportHeight: 600,
  });

  assert.equal(sameDiagramViewport(viewport, { ...viewport }), true);
  assert.equal(sameDiagramViewport(viewport, { ...viewport, offsetX: viewport.offsetX + 1 }), false);
});

test("fitDiagramViewport keeps small diagrams within the visible frame", () => {
  const result = fitDiagramViewport({
    contentWidth: 440,
    contentHeight: 120,
    viewportWidth: 802,
    viewportHeight: 470,
  });

  assert.equal(result.scale, 1);
  assert.equal(result.canvasWidth, 802);
  assert.equal(result.canvasHeight, 470);
  assert.equal(result.offsetX, 181);
  assert.equal(result.offsetY, 175);
});
