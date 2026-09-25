// Decompose an exact visible edge at each compound boundary. ELK can then
// lay out every container independently (RIGHT for the model, DOWN within
// modules), while routing each segment against its owner's visible children.
// All generated IDs are layout-only; Graph IR endpoints remain untouched.
function parentPath(path) {
  const index = path.lastIndexOf(".");
  return index < 0 ? null : path.slice(0, index);
}

function commonAncestor(left, right) {
  const a = left.split(".");
  const b = right.split(".");
  let index = 0;
  while (index < a.length && index < b.length && a[index] === b[index]) index += 1;
  return a.slice(0, index).join(".");
}

export function buildElkHierarchyEdges(graph) {
  const visible = new Set(graph.nodes.map((node) => node.path));
  const partsByOwner = new Map();
  const portsByNode = new Map();
  const segmentsByEdge = new Map();
  const append = (owner, segment) => {
    const list = partsByOwner.get(owner) || [];
    list.push(segment);
    partsByOwner.set(owner, list);
  };
  const port = (node, edgeId, direction) => {
    const id = `${node}::bridge::${edgeId}::${direction}`;
    const list = portsByNode.get(node) || [];
    list.push({ id, direction });
    portsByNode.set(node, list);
    return id;
  };
  for (const edge of graph.edges) {
    if (edge.kind !== "dataflow" || !visible.has(edge.source) || !visible.has(edge.target)) continue;
    const ancestor = commonAncestor(edge.source, edge.target);
    if (!ancestor || edge.source === edge.target) continue;
    const outgoing = [];
    const incoming = [];
    let sourcePath = edge.source;
    let sourceRef = sourcePath === ancestor
      ? port(ancestor, edge.id, "out")
      : `${sourcePath}::out`;
    while (sourcePath !== ancestor && parentPath(sourcePath) !== ancestor) {
      const owner = parentPath(sourcePath);
      if (!owner) break;
      const boundary = port(owner, edge.id, "out");
      const id = `${edge.id}::out::${owner}`;
      append(owner, { id, source: sourceRef, target: boundary });
      outgoing.push(id);
      sourceRef = boundary;
      sourcePath = owner;
    }
    let targetPath = edge.target;
    let targetRef = targetPath === ancestor
      ? port(ancestor, edge.id, "in")
      : `${targetPath}::in`;
    while (targetPath !== ancestor && parentPath(targetPath) !== ancestor) {
      const owner = parentPath(targetPath);
      if (!owner) break;
      const boundary = port(owner, edge.id, "in");
      const id = `${edge.id}::in::${owner}`;
      append(owner, { id, source: boundary, target: targetRef });
      incoming.unshift(id);
      targetRef = boundary;
      targetPath = owner;
    }
    const core = `${edge.id}::core`;
    append(ancestor, { id: core, source: sourceRef, target: targetRef });
    segmentsByEdge.set(edge.id, [...outgoing, core, ...incoming]);
  }
  return { partsByOwner, portsByNode, segmentsByEdge };
}
