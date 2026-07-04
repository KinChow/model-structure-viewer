const DEFAULT_PADDING = 32;
const DEFAULT_MIN_SCALE = 0.25;
const DEFAULT_MAX_SCALE = 1;

function floor4(value) {
  return Math.floor(value * 10000) / 10000;
}

export function fitDiagramViewport({
  contentWidth,
  contentHeight,
  viewportWidth,
  viewportHeight,
  zoom = 1,
  padding = DEFAULT_PADDING,
  minScale = DEFAULT_MIN_SCALE,
  maxScale = DEFAULT_MAX_SCALE,
}) {
  const safeContentWidth = Math.max(1, contentWidth);
  const safeContentHeight = Math.max(1, contentHeight);
  const safeViewportWidth = Math.max(1, viewportWidth);
  const safeViewportHeight = Math.max(1, viewportHeight);
  const availableWidth = Math.max(1, safeViewportWidth - padding * 2);
  const availableHeight = Math.max(1, safeViewportHeight - padding * 2);
  const rawBaseScale = Math.min(
    maxScale,
    availableWidth / safeContentWidth,
    availableHeight / safeContentHeight
  );
  const baseScale = floor4(Math.max(minScale, rawBaseScale));
  const scale = floor4(Math.max(minScale, baseScale * Math.max(0.1, zoom)));
  const scaledWidth = safeContentWidth * scale;
  const scaledHeight = safeContentHeight * scale;
  const canvasWidth = Math.max(safeViewportWidth, Math.ceil(scaledWidth + padding * 2));
  const canvasHeight = Math.max(safeViewportHeight, Math.ceil(scaledHeight + padding * 2));

  return {
    baseScale,
    scale,
    offsetX: floor4(Math.max(padding, (canvasWidth - scaledWidth) / 2)),
    offsetY: floor4(Math.max(padding, (canvasHeight - scaledHeight) / 2)),
    canvasWidth,
    canvasHeight,
  };
}

export function sameDiagramViewport(left, right) {
  return (
    left.baseScale === right.baseScale &&
    left.scale === right.scale &&
    left.offsetX === right.offsetX &&
    left.offsetY === right.offsetY &&
    left.canvasWidth === right.canvasWidth &&
    left.canvasHeight === right.canvasHeight
  );
}
