// 收集 IR 节点上的公式绑定，供公式索引与图节点双向联动。

export function collectFormulaLinks(root) {
  const links = [];
  function visit(node, path) {
    const formulaId = node?.attributes?.formula_id;
    if (formulaId) links.push({ path, formulaId, explanation: node.attributes.explanation || "" });
    node?.children?.forEach((child, index) => visit(child, `${path}.${index}`));
  }
  if (root) visit(root, "root");
  return links;
}
