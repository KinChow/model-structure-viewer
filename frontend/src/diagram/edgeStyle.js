function knownElements(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return 0;
  return shape.reduce((total, value) => {
    const dimension = Number(value);
    return Number.isFinite(dimension) && dimension > 0 ? total * dimension : 0;
  }, 1);
}

export function edgeStrokeWidth(edge, source) {
  if (edge?.kind !== "dataflow") return 1.5;
  if (edge?.evidence === "module-order") return 2.4;
  const elements = knownElements(source?.node?.output_shape);
  if (elements <= 0) return 1.8;
  return Math.min(2.8, 1.4 + Math.max(0, (Math.log10(elements) - 3.5) * 0.45));
}
