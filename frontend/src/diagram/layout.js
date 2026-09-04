import { metaForNode, typeClass } from "./meta";

export const NODE_WIDTH = 220;
const NODE_HEIGHTS = [56, 76, 96];
const NODE_GAP_Y = 18;
const NODE_GAP_X = 60;
const LAYOUT_TOP = 28;
const LAYOUT_LEFT = 28;

export function layoutDiagram(root, expandedGroups) {
  const expanded = expandedGroups instanceof Set ? expandedGroups : new Set();
  const items = [];

  function measure(node, depth, path) {
    const metaLines = metaForNode(node);
    const height = NODE_HEIGHTS[Math.min(metaLines.length, NODE_HEIGHTS.length - 1)];
    const isCollapsible = node.children?.length > 0;
    const isExpanded = isCollapsible ? expanded.has(path) : true;
    const item = {
      node,
      path,
      depth,
      children: [],
      width: NODE_WIDTH,
      height,
      typeClass: typeClass(node.type),
      repeat: node.repeat,
      fullName: node.name,
      displayName: node.repeat > 1 && String(node.type).includes("layer")
        ? "Decoder layer group"
        : node.name,
      metaLines,
      isCollapsible,
      isExpanded,
    };
    items.push(item);
    if (isCollapsible && isExpanded) {
      item.childItems = (node.children || []).map((child, index) => measure(child, depth + 1, `${path}.${index}`));
      item.children = item.childItems.map((child) => child.path);
    }
    const childHeight = item.childItems?.length
      ? item.childItems.reduce((sum, child) => sum + child.subtreeHeight, 0) + (item.childItems.length - 1) * NODE_GAP_Y
      : 0;
    item.subtreeHeight = Math.max(height, childHeight);
    return item;
  }

  const tree = measure(root, 0, "root");
  function place(item, x, y) {
    item.x = x;
    item.y = y + (item.subtreeHeight - item.height) / 2;
    let childY = y;
    (item.childItems || []).forEach((child) => {
      place(child, x + NODE_WIDTH + NODE_GAP_X, childY);
      childY += child.subtreeHeight + NODE_GAP_Y;
    });
  }
  place(tree, LAYOUT_LEFT, LAYOUT_TOP);

  items.forEach((item) => {
    if (!item.isCollapsible || !item.isExpanded) return;
    const descendants = items.filter((candidate) => candidate.path.startsWith(`${item.path}.`));
    if (descendants.length === 0) return;
    const left = Math.min(...descendants.map((candidate) => candidate.x)) - 14;
    const top = Math.min(...descendants.map((candidate) => candidate.y)) - 22;
    const right = Math.max(...descendants.map((candidate) => candidate.x + candidate.width)) + 14;
    const bottom = Math.max(...descendants.map((candidate) => candidate.y + candidate.height)) + 14;
    item.containerFrame = {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
      label: [
        item.displayName,
        item.repeat > 1 ? `×${item.repeat}` : null,
        item.node?.attributes?.range || null,
      ].filter(Boolean).join(" · "),
      depth: item.depth,
    };
  });
  return items;
}
