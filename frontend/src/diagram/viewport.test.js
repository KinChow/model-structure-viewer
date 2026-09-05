import assert from "node:assert/strict";
import test from "node:test";

import { fitDiagramViewport, sameDiagramViewport, scrollForZoomAnchor, zoomForWheel } from "./viewport.js";

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

test("fitDiagramViewport scales a wide diagram down and aligns it for scanning", () => {
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
  assert.equal(result.offsetY, 30);
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

test("zoomForWheel keeps modifier-wheel zoom inside the canvas bounds", () => {
  assert.equal(zoomForWheel(1, -100), 1.1);
  assert.equal(zoomForWheel(2.5, -100), 2.5);
  assert.equal(zoomForWheel(0.25, 100), 0.25);
});

test("scrollForZoomAnchor keeps the pointer over the same graph content", () => {
  const position = scrollForZoomAnchor({ contentX: 400, contentY: 120, pointerX: 100, pointerY: 80, viewport: { scale: 0.5, offsetX: 32, offsetY: 32 } });
  assert.deepEqual(position, { left: 132, top: 12 });
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
