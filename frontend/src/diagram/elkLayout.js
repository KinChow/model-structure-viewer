import ELK from "elkjs/lib/elk.bundled.js";

const elk = new ELK();

/** Lay out the visible graph independently from the source model tree. */
export async function layoutGraphWithElk(graph) {
  const result = await elk.layout({
    id: "model-graph",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "RIGHT",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.layered.spacing.nodeNodeBetweenLayers": "72",
      "elk.spacing.nodeNode": "28",
      "elk.padding": "32",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    },
    children: graph.nodes.map((node) => ({
      id: node.path,
      width: node.width,
      height: node.height,
    })),
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  });

  const positions = new Map((result.children || []).map((child) => [child.id, child]));
  const nodes = graph.nodes.map((node) => {
    const position = positions.get(node.path);
    return position ? { ...node, x: position.x, y: position.y } : node;
  });
  return {
    ...graph,
    nodes,
    // ELK positions all visible nodes directly; old hierarchy frames are not
    // used because they would imply a tree-shaped canvas.
    containerFrames: [],
  };
}
