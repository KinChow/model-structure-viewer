export function moduleSpec(id, name, type, attributes = {}, children = [], repeat = undefined) {
  return {
    kind: "module",
    id,
    name,
    type,
    repeat,
    attributes,
    children,
  };
}
