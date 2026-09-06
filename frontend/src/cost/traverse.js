export function childRepeatMultiplier(node, inheritedMultiplier = 1, { repeatHandled = false } = {}) {
  const repeat = Number.isFinite(node?.repeat) ? node.repeat : 1;
  const childHasExplicitRepeat = (node?.children || []).some((child) => Number.isFinite(child?.repeat));
  return inheritedMultiplier * (repeatHandled || childHasExplicitRepeat ? 1 : repeat);
}

export function walkStructure(root, visit) {
  function walk(node, path = "root", multiplier = 1) {
    visit({ node, path, multiplier });
    const childMultiplier = childRepeatMultiplier(node, multiplier);
    (node?.children || []).forEach((child, index) => {
      walk(child, `${path}.${index}`, childMultiplier);
    });
  }
  if (root) walk(root);
}
