export function nodeMatches(node, term) {
  if (!term) return false;
  const lower = term.trim().toLowerCase();
  if (!lower) return false;
  const cls = node.attributes?.class;
  return (
    String(node.name || "").toLowerCase().includes(lower) ||
    String(node.type || "").toLowerCase().includes(lower) ||
    String(cls || "").toLowerCase().includes(lower)
  );
}

export function computeMatches(root, term) {
  const result = new Set();
  if (!root || !term || !term.trim()) return result;
  function visit(node, path) {
    if (nodeMatches(node, term)) result.add(path);
    node.children?.forEach((child, index) => {
      visit(child, `${path}.${index}`);
    });
  }
  visit(root, "root");
  return result;
}
