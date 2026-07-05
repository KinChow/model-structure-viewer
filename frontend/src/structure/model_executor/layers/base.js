export function moduleSpec(id, name, type, attributes = {}, children = [], repeat = undefined) {
  return {
    kind: "module",
    id,
    name,
    type,
    repeat,
    attributes: Object.fromEntries(
      Object.entries(attributes).filter(([, value]) => value !== undefined && value !== null),
    ),
    children,
  };
}
