import assert from "node:assert/strict";
import test from "node:test";
import { layoutGraph } from "./layout.js";
import { layoutGraphWithElk } from "./elkLayout.js";
import { materializeStructureGraph } from "../structure/graph/materializeStructureGraph.js";
import { graphViewNode } from "../structure/graph/selectors.js";

function structureFrom(tree) {
  return { graph: materializeStructureGraph(tree) };
}

test("independent multimodal branches retain separate rows before their join", async () => {
  const structure = structureFrom({
    id: "model", type: "model",
    attributes: { dataflow_edges: [["visual", "merge"], ["embed", "merge"], ["merge", "decoder"]] },
    children: ["visual", "embed", "merge", "decoder"].map(id => ({
      id, name: id, type: id === "embed" ? "embedding" : "module", children: [],
    })),
  });
  const graph = await layoutGraphWithElk(layoutGraph(structure, new Set(["root"])));
  const visual = graph.nodes.find(n => n.path === "root.0");
  const embed = graph.nodes.find(n => n.path === "root.1");
  const merge = graph.nodes.find(n => n.path === "root.2");
  assert.ok(Math.abs(visual.y - embed.y) >= Math.min(visual.height, embed.height));
  assert.ok(merge.x > visual.x && merge.x > embed.x);
});

test("cross-container dependencies constrain their common ancestor without changing endpoints", async () => {
  const structure = structureFrom({
    id: "model", type: "model",
    attributes: { dataflow_edges: [["source.index", "target.reuse"]] },
    // Deliberately reverse tree order: dependency, not array order, decides placement.
    children: ["target", "source"].map(id => ({
      id, name: id, type: "module", attributes: { dataflow_edges: [] },
      children: [{ id: `${id}.${id === "source" ? "index" : "reuse"}`, name: id, type: "operator" }],
    })),
  });
  const view = layoutGraph(structure, new Set(["root", "root.0", "root.1"]));
  const graph = await layoutGraphWithElk(view);
  assert.ok(graph.nodes.find(n => n.path === "root.1").x < graph.nodes.find(n => n.path === "root.0").x);
  assert.deepEqual(graph.edges.map(e => [e.source, e.target]), [["root.1.0", "root.0.0"]]);
  assert.equal(graph.edges.some(e => e.id.startsWith("__constraint__")), false);
  assert.ok(graph.edges[0].routePoints?.length >= 2, "展开后的精确跨容器端点应得到完整路线");
  assert.deepEqual(graph.edges[0].routePoints[0], {
    x: graph.nodes.find(n => n.path === "root.1.0").x + graph.nodes.find(n => n.path === "root.1.0").width / 2,
    y: graph.nodes.find(n => n.path === "root.1.0").y + graph.nodes.find(n => n.path === "root.1.0").height,
  });
});

test("ELK routes a child-to-ancestor edge through the expanded frame border", async () => {
  const structure = structureFrom({
    id: "model", type: "model",
    children: [{
      id: "compound", type: "module", attributes: { dataflow_edges: [["leaf", "compound"]] },
      children: [{ id: "leaf", type: "operator", children: [] }],
    }],
  });
  const view = layoutGraph(structure, new Set(["root", "root.0"]));
  const graph = await layoutGraphWithElk(view);
  const edge = graph.edges.find(e => e.source === "root.0.0" && e.target === "root.0");
  assert.ok(edge?.routePoints?.length >= 2);
  const frame = graph.containerFrames.find(f => f.id === "root.0");
  const end = edge.routePoints.at(-1);
  assert.ok(end.y >= frame.y && end.y <= frame.y + frame.height);
});

test("layoutGraph exposes independent visible nodes and edges", () => {
  const graph = layoutGraph(structureFrom({
    name: "model",
    type: "model",
    children: [
      { name: "embed", type: "embedding", children: [] },
      { name: "layers", type: "layer-group", repeat: 2, children: [
        { name: "attention", type: "attention", children: [] },
      ] },
    ],
  }), new Set(["root", "root.1"]));

  assert.equal(graph.nodes.find((node) => node.path === "root.1").children, undefined);
  assert.equal(graph.nodes.find((node) => node.path === "root.0").stage, "input");
  assert.equal(graph.nodes.find((node) => node.path === "root.1.0").stage, "decoder");
  assert.deepEqual(graph.edges.filter((edge) => edge.kind === "dataflow" && edge.evidence === "module-order").map(({ source, target }) => [source, target]), [["root.0", "root.1"]]);
  assert.ok(graph.nodes.find((node) => node.path === "root.0").x < graph.nodes.find((node) => node.path === "root.1").x);
  assert.equal(graph.nodes.find((node) => node.path === "root.1").x, graph.nodes.find((node) => node.path === "root.1.0").x);
  assert.ok(graph.containerFrames.some((frame) => frame.id === "root.1"));
});

test("layoutGraph consumes explicit IR edges without inferring replacements", () => {
  const graph = layoutGraph({
    graph: {
      version: 2,
      schema_version: 2,
      root_id: "root",
      nodes: [
        { id: "root", name: "model", type: "model" },
        { id: "root.0", parent_id: "root", order: 0, name: "first", type: "module" },
        { id: "root.1", parent_id: "root", order: 1, name: "second", type: "module" },
      ],
      edges: [{ id: "explicit", source: "root.1", target: "root.0", kind: "dataflow", evidence: "declared" }],
    },
  }, new Set(["root"]));

  assert.equal(graph.graphVersion, 2);
  assert.deepEqual(graph.edges, [
    { id: "explicit", source: "root.1", target: "root.0", kind: "dataflow", evidence: "declared" },
  ]);
});

test("layoutGraph builds the canvas view from Graph IR and ignores stale legacy fields", () => {
  const graphRoot = {
    id: "model",
    name: "Graph Model",
    type: "model",
    children: [{ id: "graph.decoder", name: "Graph Decoder", type: "decoder", children: [] }],
  };
  const graph = materializeStructureGraph(graphRoot);
  const layout = layoutGraph({ graph }, new Set(["root"]));

  assert.equal(layout.nodes.find((node) => node.path === "root").node.name, "Graph Model");
  assert.equal(layout.nodes.find((node) => node.path === "root.0").node.name, "Graph Decoder");
});

test("layoutGraph consumes builder-declared edges without using display names", () => {
  const graph = layoutGraph(structureFrom({
    name: "model",
    type: "model",
    children: [{
      name: "opaque module",
      type: "mlp",
      attributes: { dataflow_edges: [["left", "right"]] },
      children: [
        { id: "opaque.left", name: "first branch", type: "operator", children: [] },
        { id: "opaque.right", name: "second branch", type: "operator", children: [] },
      ],
    }],
  }), new Set(["root", "root.0"]));

  assert.deepEqual(graph.edges.filter((edge) => edge.evidence === "declared").map(({ source, target }) => [source, target]), [["root.0.0", "root.0.1"]]);
  assert.equal(graph.edges.some((edge) => edge.evidence === "module-order" && edge.source.startsWith("root.0.")), false);
});

test("graph view projection preserves hierarchy and node facts", () => {
  const root = {
    id: "model",
    name: "Model",
    type: "model",
    attributes: { class: "Model" },
    children: [{
      id: "decoder",
      name: "Decoder",
      type: "decoder",
      repeat: 4,
      attributes: { range: "0..3" },
      children: [{ id: "decoder.attention", name: "Attention", type: "attention", children: [] }],
    }],
  };
  const graph = materializeStructureGraph(root);
  const projected = graphViewNode(graph, graph.root_id);
  assert.equal(projected.id, "model");
  assert.equal(projected.children[0].id, "decoder");
  assert.equal(projected.children[0].repeat, 4);
  assert.equal(projected.children[0].attributes.range, "0..3");
  assert.equal(projected.children[0].children[0].id, "decoder.attention");
});

test("ELK lays out the graph without changing stable node paths", async () => {
  const graph = layoutGraph(structureFrom({
    name: "model", type: "model", children: [
      { name: "a", type: "module", children: [
        { name: "a-child", type: "operator", children: [] },
        { name: "a-next", type: "operator", children: [] },
      ] },
      { name: "b", type: "module", children: [] },
    ],
  }), new Set(["root", "root.0"]));
  const laidOut = await layoutGraphWithElk(graph);
  assert.deepEqual(laidOut.nodes.map((node) => node.path), ["root", "root.0", "root.0.0", "root.0.1", "root.1"]);
  assert.ok(laidOut.nodes.every((node) => Number.isFinite(node.x) && Number.isFinite(node.y)));
  assert.ok(laidOut.nodes.find((node) => node.path === "root.0.0").y > laidOut.nodes.find((node) => node.path === "root.0").y);
  assert.ok(laidOut.nodes.find((node) => node.path === "root.0.0").y < laidOut.nodes.find((node) => node.path === "root.0.1").y);
  assert.deepEqual(JSON.parse(JSON.stringify(laidOut.edges.map(({ id, source, target, kind, evidence, source_canonical_id, target_canonical_id }) => ({ id, source, target, kind, evidence, source_canonical_id, target_canonical_id })))), graph.edges);
  assert.ok(laidOut.edges.every((edge) => edge.sections === undefined));
  assert.deepEqual(laidOut.containerFrames.map((frame) => frame.id), ["root", "root.0"]);
  assert.equal(laidOut.containerFrames.find((frame) => frame.id === "root.0").edgeAnchorOffset, 70);
});

test("keeps the output head inside the model compound", async () => {
  // HF/vLLM 语义：lm_head 是 XxxForCausalLM 的直接成员，应落在 model 容器内，
  // 与 decoder 等顶层子节点并列，而非被拎到容器外。
  const graph = layoutGraph(structureFrom({
    name: "DeepseekV3ForCausalLM", type: "model", children: [
      { name: "decoder", type: "module", children: [
        { name: "norm", type: "normalization", children: [] },
      ] },
      { name: "lm head", type: "output", children: [] },
    ],
  }), new Set(["root", "root.0"]));
  const laidOut = await layoutGraphWithElk(graph);
  assert.deepEqual(graph.edges.filter((edge) => edge.evidence === "module-order").map(({ source, target }) => [source, target]), [["root", "root.1"]]);
  const modelFrame = laidOut.containerFrames.find((frame) => frame.id === "root");
  assert.ok(modelFrame, "model 容器 frame 存在");
  const lmHead = laidOut.nodes.find((node) => node.path === "root.1");
  const decoder = laidOut.nodes.find((node) => node.path === "root.0");
  // lm_head 在 decoder 右侧，且横向落在 model 容器框内
  assert.ok(lmHead.x > decoder.x);
  assert.ok(lmHead.x >= modelFrame.x);
  assert.ok(lmHead.x <= modelFrame.x + modelFrame.width);
});

test("旁挂草稿分支保持独立且不与主干节点重叠", async () => {
  // 生产布局的顶层语义流统一按输入→输出从左到右；DSpark/MTP 是旁挂
  // 分支，不要求再强行压到主干下方，但必须保持独立且不能覆盖主干节点。
  const graph = layoutGraph(structureFrom({
    name: "DeepseekV4ForCausalLM", type: "model", children: [
      { name: "embed tokens", type: "embedding", input_shape: [1, 2], output_shape: [1, 4], children: [] },
      { name: "decoder", type: "decoder", input_shape: [1, 4], output_shape: [1, 4], children: [
        { name: "layer", type: "module", children: [] },
      ] },
      { name: "dspark", type: "dspark", input_shape: [1, 4], output_shape: [1, 4], attributes: {
        dataflow_edges: [],
      }, children: [
        { name: "markov head", type: "dspark-markov", children: [] },
      ] },
      { name: "final norm", type: "normalization", input_shape: [1, 4], output_shape: [1, 4], children: [] },
      { name: "lm head", type: "output", input_shape: [1, 4], output_shape: [1, 8], children: [] },
    ],
  }), new Set(["root"]));
  const laidOut = await layoutGraphWithElk(graph);
  const top = laidOut.nodes.filter((node) => node.path.split(".").length === 2);
  const draft = top.find((node) => node.path === "root.2");
  const trunk = top.filter((node) => node.path !== "root.2");
  assert.ok(draft.x > Math.min(...trunk.map((node) => node.x)),
    "draft branch should remain downstream of the main input");
  // 无任何顶层节点两两重叠
  for (const a of top) {
    for (const b of top) {
      if (a.path >= b.path) continue;
      const ox = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
      const oy = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
      assert.ok(!(ox > 2 && oy > 2), `${a.path} 与 ${b.path} 重叠 ${Math.round(ox)}x${Math.round(oy)}`);
    }
  }
});

test("多模态输入分支存在时仍保持 decoder 到 lm_head 左到右主干", async () => {
  const graph = layoutGraph(structureFrom({
    name: "multimodal", type: "model", attributes: {
      dataflow_edges: [["vision", "merge"], ["text", "merge"], ["merge", "decoder"], ["decoder", "final norm"], ["final norm", "lm head"], ["decoder", "dspark"], ["dspark", "lm head"]],
    }, children: [
      { id: "vision", name: "vision", type: "vision-encoder", children: [] },
      { id: "text", name: "text", type: "embedding", children: [] },
      { id: "merge", name: "merge", type: "operator", children: [] },
      { id: "decoder", name: "Decoder", type: "decoder", children: [] },
      { id: "dspark", name: "DSpark", type: "dspark", children: [{ name: "stage", type: "operator", children: [] }] },
      { id: "final norm", name: "final norm", type: "normalization", children: [] },
      { id: "lm head", name: "lm head", type: "output", children: [] },
    ],
  }), new Set(["root", "root.4"]));
  const laidOut = await layoutGraphWithElk(graph);
  const node = path => laidOut.nodes.find(item => item.path === path);
  assert.ok(node("root.3").x < node("root.5").x, "decoder should precede final norm");
  assert.ok(node("root.5").x < node("root.6").x, "final norm should precede lm head");
  assert.ok(node("root.4").x > node("root.3").x, "DSpark should remain downstream of decoder");
});

test("展开 DSpark 后 root 级旁挂边使用最终坐标的避障路线", async () => {
  const graph = layoutGraph(structureFrom({
    name: "DeepseekV41ForCausalLM", type: "model", children: [
      { name: "decoder", type: "decoder", children: [
        { name: "layer", type: "module", children: [] },
      ] },
      { name: "dspark", type: "dspark", children: [
        { name: "main projection", type: "operator", children: [] },
        { name: "confidence head", type: "operator", children: [] },
      ] },
      { name: "final norm", type: "normalization", children: [] },
      { name: "lm head", type: "output", children: [] },
    ],
  }), new Set(["root", "root.0", "root.1"]));
  const laidOut = await layoutGraphWithElk(graph);
  const rootEdges = laidOut.edges.filter((edge) => edge.source.split(".").length === 2);
  const draftEdges = rootEdges.filter((edge) => edge.source === "root.1" || edge.target === "root.1");
  assert.equal(draftEdges.length, 2);
  assert.ok(draftEdges.every((edge) => edge.routePoints?.length >= 2), "旁挂边必须有最终坐标路线");
  assert.ok(draftEdges.every((edge) => edge.routePoints.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))));
});

test("ELK lane 中展开容器不会压缩主干间距", async () => {
  // Layout IR 把输入、主干和辅助分支建模为 ELK compound lane。展开 decoder
  // 只改变 decoder compound 的尺寸，不应压缩 main lane 中 decoder → final norm
  // 的有效间距。
  const tree = {
    name: "DeepseekV4ForCausalLM", type: "model", children: [
      { name: "embed tokens", type: "embedding", input_shape: [1, 2], output_shape: [1, 4], children: [] },
      { name: "decoder", type: "decoder", input_shape: [1, 4], output_shape: [1, 4], children: [
        { name: "attn", type: "attention", children: [] },
        { name: "mlp", type: "mlp", children: [] },
      ] },
      { name: "final norm", type: "normalization", input_shape: [1, 4], output_shape: [1, 4], children: [] },
      { name: "lm head", type: "output", input_shape: [1, 4], output_shape: [1, 8], children: [] },
    ],
  };
  const effGap = async (expanded) => {
    const laidOut = await layoutGraphWithElk(layoutGraph(structureFrom(tree), expanded));
    const frameById = new Map(laidOut.containerFrames.map((frame) => [frame.id, frame]));
    const box = (path) => {
      const frame = frameById.get(path);
      if (frame) return { x: frame.x, w: frame.width, y: frame.y, h: frame.height };
      const node = laidOut.nodes.find((n) => n.path === path);
      return { x: node.x, w: node.width, y: node.y, h: node.height };
    };
    const decoder = box("root.1");
    const finalNorm = box("root.2");
    return finalNorm.x - (decoder.x + decoder.w);
  };
  const collapsedGap = await effGap(new Set(["root"]));
  const expandedGap = await effGap(new Set(["root", "root.1"]));
  assert.ok(collapsedGap >= 40, `折叠主干间距 ${Math.round(collapsedGap)} 过窄`);
  assert.ok(expandedGap >= 40, `展开主干间距 ${Math.round(expandedGap)} 过窄`);
});

test("展开的草稿子树完整落在 model 容器 frame 内（不越界）", async () => {
  // 手动把草稿挪到主干下方后，必须同步撑高 model 容器高度，否则草稿展开更高时
  // 会溢出容器框底部（展示层越界回归）。
  const graph = layoutGraph(structureFrom({
    name: "DeepseekV4ForCausalLM", type: "model", children: [
      { name: "embed tokens", type: "embedding", input_shape: [1, 2], output_shape: [1, 4], children: [] },
      { name: "decoder", type: "decoder", input_shape: [1, 4], output_shape: [1, 4], children: [
        { name: "layer", type: "module", children: [] },
      ] },
      { name: "dspark", type: "dspark", input_shape: [1, 4], output_shape: [1, 4], attributes: { dataflow_edges: [] }, children: [
        { name: "s0", type: "operator", children: [] },
        { name: "s1", type: "normalization", children: [] },
        { name: "s2", type: "decoder", children: [] },
        { name: "s3", type: "decoder", children: [] },
        { name: "s4", type: "operator", children: [] },
        { name: "s5", type: "normalization", children: [] },
      ] },
      { name: "final norm", type: "normalization", input_shape: [1, 4], output_shape: [1, 4], children: [] },
      { name: "lm head", type: "output", input_shape: [1, 4], output_shape: [1, 8], children: [] },
    ],
  }), new Set(["root", "root.2"]));
  const laidOut = await layoutGraphWithElk(graph);
  const modelFrame = laidOut.containerFrames.find((frame) => frame.id === "root");
  assert.ok(modelFrame, "model 容器 frame 存在");
  const draftKids = laidOut.nodes.filter((node) => node.path.startsWith("root.2"));
  const frameBottom = modelFrame.y + modelFrame.height;
  for (const node of draftKids) {
    assert.ok(
      node.y + node.height <= frameBottom + 2,
      `草稿子节点 ${node.path} 底部 ${Math.round(node.y + node.height)} 越界 model frame 底部 ${Math.round(frameBottom)}`,
    );
  }
});

test("layoutGraph adds dataflow edges only when tensor shapes match", () => {
  const graph = layoutGraph(structureFrom({
    name: "block", type: "module", children: [
      { name: "gate", type: "operator", input_shape: [1, 4], output_shape: [1, 8], children: [] },
      { name: "activation", type: "operator", input_shape: [1, 8], output_shape: [1, 8], children: [] },
      { name: "same-shape", type: "operator", input_shape: [1, 8], output_shape: [1, 8], children: [] },
      { name: "other", type: "operator", input_shape: [1, 16], output_shape: [1, 16], children: [] },
    ],
  }), new Set(["root"]));
  assert.deepEqual(graph.edges.filter((edge) => edge.kind === "dataflow").map(({ source, target }) => [source, target]), [
    ["root.0", "root.1"],
    ["root.1", "root.2"],
    ["root.2", "root.3"],
  ]);
});

test("layoutGraph models MLA as a branched attention graph", () => {
  const graph = layoutGraph(structureFrom({
    name: "model", type: "model", children: [
      { name: "MLA Attention", type: "attention", attributes: { dataflow_edges: [["q_proj", "rope"], ["k_proj", "rope"], ["rope", "scores"], ["scores", "softmax"], ["softmax", "context"], ["v_proj", "context"], ["context", "o_proj"]] }, children: [
        { id: "q_proj", name: "q projection", type: "operator", children: [] },
        { id: "k_proj", name: "k projection", type: "operator", children: [] },
        { id: "v_proj", name: "v projection", type: "operator", children: [] },
        { id: "rope", name: "rotary position embedding", type: "operator", children: [] },
        { id: "scores", name: "attention scores", type: "operator", children: [] },
        { id: "softmax", name: "attention probabilities", type: "operator", children: [] },
        { id: "context", name: "weighted value", type: "operator", children: [] },
        { id: "o_proj", name: "output projection", type: "operator", children: [] },
      ] },
    ],
  }), new Set(["root", "root.0"]));
  assert.deepEqual(graph.edges.filter((edge) => edge.evidence === "declared").map(({ source, target }) => [source, target]), [
    ["root.0.0", "root.0.3"],
    ["root.0.1", "root.0.3"],
    ["root.0.3", "root.0.4"],
    ["root.0.4", "root.0.5"],
    ["root.0.5", "root.0.6"],
    ["root.0.2", "root.0.6"],
    ["root.0.6", "root.0.7"],
  ]);
});

test("ELK keeps MLA input projections upstream of their consumers", async () => {
  const graph = layoutGraph(structureFrom({
    name: "model", type: "model", children: [
      { name: "MLA Attention", type: "attention", attributes: { dataflow_edges: [["q_proj", "rope"], ["k_proj", "rope"], ["rope", "scores"], ["scores", "softmax"], ["softmax", "context"], ["v_proj", "context"], ["context", "o_proj"]] }, children: [
        { id: "q_proj", name: "q projection", type: "operator", children: [] },
        { id: "k_proj", name: "k projection", type: "operator", children: [] },
        { id: "v_proj", name: "v projection", type: "operator", children: [] },
        { id: "rope", name: "rotary position embedding", type: "operator", children: [] },
        { id: "scores", name: "attention scores", type: "operator", children: [] },
        { id: "softmax", name: "attention probabilities", type: "operator", children: [] },
        { id: "context", name: "weighted value", type: "operator", children: [] },
        { id: "o_proj", name: "output projection", type: "operator", children: [] },
      ] },
    ],
  }), new Set(["root", "root.0"]));
  const laidOut = await layoutGraphWithElk(graph);
  const get = (name) => laidOut.nodes.find((node) => node.node.name === name);
  const inputs = [get("q projection"), get("k projection"), get("v projection")];
  assert.ok(inputs.slice(0, 2).every((node) => node.y < get("rotary position embedding").y));
  assert.ok(get("v projection").y < get("weighted value").y);
});

test("layoutGraph models MLP as a gated branch instead of a sequential chain", () => {
  const graph = layoutGraph(structureFrom({
    name: "model", type: "model", children: [
      { name: "MLP", type: "mlp", attributes: { dataflow_edges: [["gate_proj", "swiglu"], ["up_proj", "swiglu"], ["swiglu", "down_proj"]] }, children: [
        { id: "gate_proj", name: "gate projection", type: "operator", children: [] },
        { id: "up_proj", name: "up projection", type: "operator", children: [] },
        { id: "swiglu", name: "SwiGLU activation", type: "operator", children: [] },
        { id: "down_proj", name: "down projection", type: "operator", children: [] },
      ] },
    ],
  }), new Set(["root", "root.0"]));
  assert.deepEqual(graph.edges.filter((edge) => edge.evidence === "declared").map(({ source, target }) => [source, target]), [
    ["root.0.0", "root.0.2"],
    ["root.0.1", "root.0.2"],
    ["root.0.2", "root.0.3"],
  ]);
  assert.equal(graph.edges.some((edge) => edge.source === "root.0.0" && edge.target === "root.0.1"), false);
});

test("layoutGraph models MoE routing and combine branches semantically", () => {
  const graph = layoutGraph(structureFrom({
    name: "model", type: "model", children: [
      { name: "Routed MoE", type: "moe", attributes: { dataflow_edges: [["router", "topk"], ["topk", "dispatch"], ["dispatch", "expert_mlp"], ["topk", "combine"], ["expert_mlp", "combine"]] }, children: [
        { id: "router", name: "router logits", type: "operator", children: [] },
        { id: "topk", name: "top-k expert routing", type: "operator", children: [] },
        { id: "dispatch", name: "expert dispatch", type: "operator", children: [] },
        { id: "expert_mlp", name: "expert MLP", type: "operator", children: [] },
        { id: "combine", name: "expert combine", type: "operator", children: [] },
      ] },
    ],
  }), new Set(["root", "root.0"]));
  assert.deepEqual(graph.edges.filter((edge) => edge.evidence === "declared").map(({ source, target }) => [source, target]), [
    ["root.0.0", "root.0.1"],
    ["root.0.1", "root.0.2"],
    ["root.0.2", "root.0.3"],
    ["root.0.1", "root.0.4"],
    ["root.0.3", "root.0.4"],
  ]);
  assert.equal(graph.edges.some((edge) => edge.source === "root.0.0" && edge.target === "root.0.1" && edge.evidence === "module-order"), false);
});
