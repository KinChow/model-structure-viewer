import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { aggregateCost } from "../../cost/aggregate.js";
import { actionsByFormulaGroup } from "../../cost/ui.js";
import { buildNodeLens } from "../../diagram/lens.js";
import { buildSkeleton } from "../truth/skeleton.js";

const root = new URL("../../../../models/", import.meta.url);
const read = url => fs.existsSync(url) ? JSON.parse(fs.readFileSync(url, "utf8")) : null;
const catalog = read(new URL("catalog.json", root));
// Frozen expectations from primary forward implementations, not the builder.
const semantics = {
  qwen3_5: "placeholder_scatter", qwen3_5_moe: "placeholder_scatter",
  qwen4_exp: "placeholder_scatter", glm5_next: "placeholder_scatter",
  minimax_m3_vl: "placeholder_scatter", kimi_k25: "placeholder_expand",
  kimi_k3: "placeholder_expand", deepseek_v41: "image_span_overwrite",
  deepseek_v4: "image_span_overwrite",
};

for (const loading of ["config", "artifacts"]) {
  test(`39 multimodal entries have independent text/vision branches (${loading})`, () => {
    let checked = 0;
    for (const entry of catalog.models) {
      const config = read(new URL(entry.config_path, root));
      const n = normalizeConfig(config);
      const dir = new URL("./", new URL(entry.config_path, root));
      const s = loading === "config" ? buildStructureFromConfig(config, { modelId: entry.model_id })
        : buildStructureFromArtifacts({ config, modelId: entry.model_id,
          checkpointTruth: read(new URL("skeleton-truth.json", dir)) || read(new URL("header-truth.json", dir)),
          sourceRef: read(new URL("source-ref.json", dir)) });
      const g = s.graph;
      const fusion = g.nodes.find(node => node.canonical_id === "multimodal_fusion");
      if (!n.hasVision) {
        assert.equal(fusion, undefined, entry.model_id);
        continue;
      }
      checked++;
      assert.ok(fusion, entry.model_id);
      assert.equal(fusion.attributes.fusion_semantics, semantics[n.modelType], entry.model_id);
      assert.equal(fusion.attributes.sequence_policy, "workload_is_post_fusion");
      const embed = g.nodes.find(node => node.canonical_id === "embed_tokens");
      const vision = g.nodes.find(node => node.type === "vision-encoder");
      const projector = g.nodes.find(node => node.type === "projector");
      const imageInput = g.nodes.find(node => node.canonical_id === "image_input");
      const textInput = g.nodes.find(node => node.canonical_id === "text_input");
      const incoming = target => g.edges.filter(e => e.target === target.id).map(e => e.source);
      assert.deepEqual(incoming(embed), [textInput.id], "visual features are never embedding lookup indices");
      assert.deepEqual(incoming(vision), [imageInput.id]);
      assert.deepEqual(new Set(incoming(fusion)), new Set([embed.id, (projector || vision).id, textInput.id]));
      const entryStack = g.nodes.find(node => node.type === (n.modelType === "deepseek_v41" ? "encoder" : "decoder"));
      const expand = g.nodes.find(node => node.attributes?.semantic_role === "gr_expand" && node.parent_id === g.root_id);
      assert.ok(g.edges.some(e => e.source === fusion.id && e.target === (expand || entryStack).id));
      if (expand) {
        assert.equal(expand.input_shape.at(-1), n.hiddenSize);
        assert.equal(expand.output_shape.at(-1), n.hiddenSize * n.hyperConnectionCount);
        assert.ok(g.edges.some(e => e.source === expand.id && e.target === entryStack.id));
      }
      assert.equal(g.version, 2);
      assert.equal(g.schema_version, 2);
      assert.ok(!fusion.attributes.weightMatrices?.length);
      assert.ok(!fusion.tensor_names?.length);
      const decode = countsForNode(fusion, { config: n, options: { phase: "decode", batch: 2, sequence: 8 }, bytesPerElement: 2 });
      assert.equal(decode.matrix, 0);
      assert.equal(decode.bytes.weights, 0);
      assert.equal(decode.bytes.actIn, 0, "cached vision is not scattered again every decode token");
      const prefill = countsForNode(fusion, { config: n, options: { phase: "prefill", batch: 2, sequence: 8 }, bytesPerElement: 2 });
      assert.equal(prefill.matrix, 0);
      assert.equal(prefill.bytes.weights, 0);
      assert.equal(prefill.bytes.actIn, null, "placeholder occupancy and materialization are not supplied by sequence alone");
      if (n.modelType === "deepseek_v41") {
        assert.ok(g.edges.some(e => e.relation === "kv-projection"));
        const draft = g.nodes.find(node => node.type === "dspark");
        assert.ok(g.edges.some(e => e.target === draft.id && g.nodes.find(n => n.id === e.source)?.type === "decoder"));
      }
    }
    assert.equal(checked, 39);
  });
}

test("fusion unknown traffic survives totals, per-group UI and node lens; compute stays known", () => {
  const config = read(new URL("Qwen/Qwen3.5-0.8B/config.json", root));
  const s = buildStructureFromConfig(config);
  const n = normalizeConfig(config);
  const cost = aggregateCost({ graph: s.graph, config: n, batch: 1, sequence: 16 });
  assert.equal(cost.computeComplete, true);
  assert.ok(cost.totalMacs > 0);
  assert.equal(cost.actions.actIn, null);
  assert.equal(cost.actions.actOut, null);
  const memory = actionsByFormulaGroup(cost).find(group => group.group === "memory");
  assert.equal(memory.actions.bytes.actIn, null);
  const fusion = s.graph.nodes.find(node => node.canonical_id === "multimodal_fusion");
  const lens = buildNodeLens(s, { peak_flops: { bf16: 1000 }, memory_bandwidth: 100 });
  assert.equal(lens.nodes[fusion.id].times.memory, null);
  assert.equal(lens.nodes[fusion.id].metrics.vramBytes, null);
});

test("semantic fusion cannot bind a checkpoint module with a coincidentally matching path", () => {
  const config = read(new URL("Qwen/Qwen3.5-0.8B/config.json", root));
  const truth = { tensors: [{ name: "model.multimodal_fusion.weight", shape: [2, 2], dtype: "BF16" }] };
  const s = buildStructureFromArtifacts({ config, checkpointTruth: truth });
  const fusion = s.graph.nodes.find(node => node.canonical_id === "multimodal_fusion");
  assert.ok(!fusion.tensor_names?.length);
  assert.equal(s.graph.nodes.filter(node => node.tensor_names?.includes(truth.tensors[0].name)).length, 1,
    "unmatched checkpoint fact must survive as a gap, not be discarded");
});

test("multimodal projector canonical IDs follow published module paths", () => {
  const expected = [
    ["MiniMaxAI/MiniMax-M3", [
      "multi_modal_projector", "multi_modal_projector.linear_1",
      "multi_modal_projector.act", "multi_modal_projector.linear_2",
      "patch_merge_mlp", "patch_merge_mlp.linear_1",
      "patch_merge_mlp.act", "patch_merge_mlp.linear_2",
    ]],
    ["moonshotai/Kimi-K2.5", [
      "mm_projector", "mm_projector.pre_norm", "mm_projector.proj.0",
      "mm_projector.proj.1", "mm_projector.proj.2",
    ]],
    ["moonshotai/Kimi-K3", [
      "mm_projector", "mm_projector.pre_norm", "mm_projector.proj.0",
      "mm_projector.proj.1", "mm_projector.proj.2",
    ]],
    ["deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", [
      "aligner", "aligner.w1", "aligner.activation", "aligner.w2",
    ]],
    ["deepseek-ai/DeepSeek-V4.1-Flash", [
      "aligner", "aligner.w1", "aligner.activation", "aligner.w2",
    ]],
  ];
  for (const [modelId, paths] of expected) {
    const config = read(new URL(`${modelId}/config.json`, root));
    const structure = buildStructureFromConfig(config, { modelId });
    const ids = new Set(structure.graph.nodes.map((node) => node.canonical_id));
    for (const path of paths) assert.ok(ids.has(path), `${modelId}: missing ${path}`);
  }
});

test("released MiniMax-M3 tensors split projection and patch merge into independent modules", () => {
  const fixture = read(new URL("../models/__fixtures__/minimax-m3-projector-header.json", import.meta.url));
  const config = read(new URL("MiniMaxAI/MiniMax-M3/config.json", root));
  const structure = buildStructureFromArtifacts({
    config, modelId: fixture.model_id, revision: fixture.revision,
    checkpointTruth: { skeleton: buildSkeleton(fixture.tensors), tensor_count: fixture.tensors.length },
  });
  const graph = structure.graph;
  const byCanonical = new Map(graph.nodes.map(node => [node.canonical_id, node]));
  const expected = ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2",
    "patch_merge_mlp.linear_1", "patch_merge_mlp.linear_2"];
  for (const path of expected) assert.ok(byCanonical.has(path), `missing released module ${path}`);
  for (const tensor of fixture.tensors) {
    const owner = graph.nodes.filter(node => node.tensor_names?.includes(tensor.name));
    assert.equal(owner.length, 1, `${tensor.name}: exactly one weight owner`);
    assert.equal(owner[0].canonical_id, tensor.name.replace(/\.(weight|bias)$/, ""));
    assert.equal(owner[0].value_source, "checkpoint");
    assert.ok(Object.values(owner[0].weight_shapes || {}).some(shape => shape.join(",") === tensor.shape.join(",")),
      `${tensor.name}: released shape must be bound`);
  }
  const expectedElements = fixture.tensors.reduce((sum, tensor) =>
    sum + tensor.shape.reduce((size, dim) => size * dim, 1), 0);
  const projected = graph.nodes.filter(node => /^(multi_modal_projector|patch_merge_mlp)\./.test(node.canonical_id));
  const declaredElements = projected.flatMap(node => node.attributes.weightMatrices || [])
    .reduce((sum, matrix) => sum + matrix.shape.reduce((size, dim) => size * dim, 1), 0);
  assert.equal(declaredElements, expectedElements, "released projector tensors and graph capacity reconcile exactly");
  assert.equal(projected.filter(node => node.tensor_names?.length)
    .reduce((sum, node) => sum + node.params, 0), expectedElements);
  assert.equal(graph.nodes.some(node => node.canonical_id?.includes("merge_linear_")), false,
    "the fused library class is not the released checkpoint's module layout");
  assert.ok(graph.edges.some(edge => edge.source_canonical_id === "multi_modal_projector"
    && edge.target_canonical_id === "patch_merge_mlp"));
  assert.ok(graph.edges.some(edge => edge.source_canonical_id === "patch_merge_mlp"
    && edge.target_canonical_id === "multimodal_fusion"));
});

test("both published MiniMax-M3 indexes have the same split projector keys", () => {
  const published = read(new URL("../models/__fixtures__/minimax-m3-published-projector-index.json", import.meta.url));
  assert.equal(published.models.length, 2);
  for (const variant of published.models) {
    assert.match(variant.index_sha256, /^[0-9a-f]{64}$/);
    assert.equal(variant.tensor_names.length, 8);
    assert.ok(variant.tensor_names.every(name => /^(multi_modal_projector|patch_merge_mlp)\.linear_[12]\.(weight|bias)$/.test(name)));
    const config = read(new URL(`${variant.model_id}/config.json`, root));
    const graph = buildStructureFromConfig(config, { modelId: variant.model_id }).graph;
    for (const name of variant.tensor_names) {
      assert.ok(graph.nodes.some(node => node.canonical_id === name.replace(/\.(weight|bias)$/, "")),
        `${variant.model_id}: missing released weight path ${name}`);
    }
    assert.equal(graph.nodes.some(node => node.canonical_id?.includes("merge_linear_")), false);
  }
});
