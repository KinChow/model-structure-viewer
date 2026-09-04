import { metaForNode, typeClass } from "./meta";

export const NODE_WIDTH = 220;
const NODE_HEIGHTS = [56, 76, 96];
const NODE_GAP_Y = 18;
const NODE_GAP_X = 60;
const LAYOUT_TOP = 28;
const LAYOUT_LEFT = 28;

export function layoutDiagram(root, expandedGroups) {
  const expanded = expandedGroups instanceof Set ? expandedGroups : new Set();
  const levels = [];
  const items = [];

  function visit(node, depth, path) {
    if (!levels[depth]) levels[depth] = [];
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
    levels[depth].push(item);
    items.push(item);
    const shouldRecurse = !isCollapsible || isExpanded;
    if (shouldRecurse) {
      node.children?.forEach((child, index) => {
        const childPath = `${path}.${index}`;
        item.children.push(childPath);
        visit(child, depth + 1, childPath);
      });
    }
  }
  visit(root, 0, "root");

  const columnHeights = levels.map((level) =>
    level.reduce((acc, item) => acc + item.height + NODE_GAP_Y, -NODE_GAP_Y)
  );
  const maxColumnHeight = Math.max(0, ...columnHeights);

  levels.forEach((level, depth) => {
    const columnHeight = columnHeights[depth];
    const startY = LAYOUT_TOP + Math.max(0, (maxColumnHeight - columnHeight) / 2);
    let cursorY = startY;
    level.forEach((item) => {
      item.x = LAYOUT_LEFT + depth * (NODE_WIDTH + NODE_GAP_X);
      item.y = cursorY;
      cursorY += item.height + NODE_GAP_Y;
    });
  });

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
      label: item.displayName,
      depth: item.depth,
    };
  });
  return items;
}
