export function moduleSpec(id, name, type, attributes = {}, children = [], repeat = undefined, role = undefined) {
  return {
    kind: "module",
    id,
    name,
    type,
    role,
    repeat,
    attributes: Object.fromEntries(
      Object.entries(attributes).filter(([, value]) => value !== undefined && value !== null),
    ),
    children,
  };
}

export function withShapeDims(spec, inputShape, outputShape) {
  return { ...spec, input_shape: inputShape, output_shape: outputShape };
}
