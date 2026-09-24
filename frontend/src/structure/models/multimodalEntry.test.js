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
import { walkStructure } from "../../cost/traverse.js";
import { computeNodeCosts } from "../../cost/compute.js";

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
      "mm_projector", "mm_projector.proj.0",
      "mm_projector.proj.1", "mm_projector.proj.2", "mm_projector.post_norm",
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

for (const variant of ["MiniMax-M3", "MiniMax-M3-MXFP8"]) {
  test(`${variant} vision tower follows the published separate-QKV CLIP path`, () => {
    const config = read(new URL(`MiniMaxAI/${variant}/config.json`, root));
    const sourceRef = read(new URL(`MiniMaxAI/${variant}/source-ref.json`, root));
    const graph = buildStructureFromArtifacts({
      modelId: `MiniMaxAI/${variant}`,
      config,
      sourceRef,
      checkpointTruth: read(new URL(`MiniMaxAI/${variant}/header-truth.json`, root)),
    }).graph;
    const ids = new Set(graph.nodes.map(node => node.canonical_id));
    for (const id of [
      "vision_tower.embeddings.proj",
      "vision_tower.pre_layrnorm",
      "vision_tower.layers.0.self_attn.q_proj",
      "vision_tower.layers.0.self_attn.k_proj",
      "vision_tower.layers.0.self_attn.v_proj",
      "vision_tower.layers.0.self_attn.out_proj",
      "vision_tower.layers.0.layer_norm1",
      "vision_tower.layers.0.layer_norm2",
      "vision_tower.layers.0.mlp.fc1",
      "vision_tower.layers.0.mlp.fc2",
      "vision_tower.rotary_emb",
    ]) assert.ok(ids.has(id), `${variant}: missing ${id}`);
    assert.equal([...ids].some(id => id === "vision_tower.layers.0.qkv_proj"), false);
    assert.equal([...ids].some(id => id === "vision_tower.layers.0.self_attn.qkv_proj"), false);
    const node = id => graph.nodes.find(candidate => candidate.canonical_id === id);
    assert.equal(node("vision_tower.layers.0.layer_norm1").attributes.affine_bias, true);
    assert.equal(node("vision_tower.layers.0.self_attn.q_proj").attributes.bias, true);
    assert.equal(normalizeConfig(config).visionTemporalPatchSize, 2,
      "published Conv3d kernel is temporal 2 × spatial 14², not a single frame");
    assert.deepEqual(node("vision_tower.embeddings.proj").attributes.weightMatrices[0].shape,
      [1280, 3 * 2 * 14 * 14]);
    // Published Conv3d + pre-LN + 32 independent CLIP blocks. The source
    // defines four biased attention linears, two affine LayerNorms and two
    // biased GELU-MLP linears per block; no learned absolute position table.
    let visualParams = 0;
    walkStructure(graph, ({ node: part, multiplier }) => {
      if (!part.id.startsWith("vision_tower.")) return;
      for (const group of part.attributes?.weightMatrices || []) {
        visualParams += group.shape.reduce((product, dim) => product * dim, 1)
          * (group.count || 1) * (group.matrices || 1) * multiplier;
      }
    });
    assert.equal(visualParams, 631185920);
    assert.equal(node("vision_tower.layers.0.self_attn.rope").attributes.position_encoding, "rope_axial_3d");
    assert.equal(node("vision_tower.layers.0.self_attn.sdpa").attributes.attention_mask_kind, "bidirectional");
    assert.ok(graph.edges.some(edge =>
      edge.source_canonical_id === "vision_tower.layers.0.self_attn.q_proj"
      && edge.target_canonical_id === "vision_tower.layers.0.self_attn.q_reshape"));
    assert.ok(graph.edges.some(edge =>
      edge.source_canonical_id === "vision_tower.layers.0.self_attn.v_reshape"
      && edge.target_canonical_id === "vision_tower.layers.0.self_attn.sdpa"));
  });
}

test("Kimi-K3 vision tower follows the published MoonViT checkpoint layout", () => {
  const config = read(new URL("moonshotai/Kimi-K3/config.json", root));
  const sourceRef = read(new URL("moonshotai/Kimi-K3/source-ref.json", root));
  const graph = buildStructureFromArtifacts({
    modelId: "moonshotai/Kimi-K3",
    config,
    sourceRef,
  }).graph;
  const ids = new Set(graph.nodes.map(node => node.canonical_id));
  for (const id of [
    "vision_tower.patch_embed.proj",
    "vision_tower.patch_embed.pos_emb",
    "vision_tower.encoder.blocks.0.wqkv",
    "vision_tower.encoder.blocks.0.wo",
    "vision_tower.encoder.blocks.0.norm0",
    "vision_tower.encoder.blocks.0.norm1",
    "vision_tower.encoder.blocks.0.mlp.fc0",
    "vision_tower.encoder.blocks.0.mlp.fc1",
    "vision_tower.encoder.final_layernorm",
  ]) assert.ok(ids.has(id), `Kimi-K3: missing ${id}`);
  assert.equal(ids.has("vision_tower.0.qkv_proj"), false);
  assert.equal(ids.has("vision_tower.encoder.blocks.0.q_proj"), false);
  const node = id => graph.nodes.find(candidate => candidate.canonical_id === id);
  assert.equal(normalizeConfig(config).visionQkvHiddenSize, 1536);
  assert.equal(normalizeConfig(config).visionPatchTokens, 4096);
  assert.equal(normalizeConfig(config).visionTokens, 1024);
  assert.deepEqual(node("vision_tower.patch_embed.proj").attributes.weightMatrices[0].shape,
    [1024, 3 * 14 * 14]);
  assert.deepEqual(node("vision_tower.patch_embed.pos_emb").attributes.weightMatrices[0].shape,
    [64, 64, 1024]);
  assert.deepEqual(node("vision_tower.encoder.blocks.0.wqkv").attributes.weightMatrices[0].shape,
    [3 * 1536, 1024]);
  assert.equal(node("vision_tower.encoder.blocks.0.wqkv").attributes.bias, false);
  assert.equal(node("vision_tower.encoder.blocks.0.wo").attributes.weightMatrices[0].shape[1], 1536);
  assert.equal(node("vision_tower.encoder.blocks.0.sdpa").attributes.attention_mask_kind, "bidirectional");
  assert.equal(node("vision_tower.encoder.blocks.0.rope").attributes.position_encoding, "rope_2d");
  assert.ok(graph.edges.some(edge =>
    edge.source_canonical_id === "vision_tower.patch_embed.proj"
    && edge.target_canonical_id === "vision_tower.patch_embed.pos_emb"));
  assert.ok(graph.edges.some(edge =>
    edge.source_canonical_id === "vision_tower.encoder.blocks.0.qkv_reshape"
    && edge.target_canonical_id === "vision_tower.encoder.blocks.0.sdpa"));
  assert.ok(graph.edges.some(edge =>
    edge.source_canonical_id === "vision_tower.encoder.rope_2d"
    && edge.target_canonical_id === "vision_tower.encoder.blocks.0"
    && edge.relation === "index-control"));
  let visionParams = 0;
  walkStructure(graph, ({ node: part, multiplier }) => {
    if (!part.id.startsWith("vision_tower.")) return;
    for (const group of part.attributes?.weightMatrices || []) {
      visionParams += group.shape.reduce((product, dim) => product * dim, 1)
        * (group.count || 1) * (group.matrices || 1) * multiplier;
    }
  });
  assert.equal(visionParams, 401214464, "published 27 blocks, patch, position and final norm");

  const rows = computeNodeCosts(graph, normalizeConfig(config), {
    batch: 1, sequence: 16, visionTokens: 1024, phase: "prefill",
  });
  const row = id => rows.find(candidate => candidate.node.id === id);
  assert.equal(row("vision_tower.patch_embed.proj").actions.matrix,
    4096 * 1024 * (3 * 14 * 14));
  assert.equal(row("vision_tower.encoder.blocks.0.wqkv").actions.matrix,
    27 * 4096 * (3 * 1536) * 1024);
  assert.equal(row("vision_tower.encoder.blocks.0.sdpa").actions.matrix,
    27 * 12 * 4096 * 4096 * (128 + 128));
  assert.equal(row("vision_tower.patch_embed.pos_emb").actions.bytes.weights,
    64 * 64 * 1024 * 2);
  assert.equal(row("mm_projector.proj.0").actions.matrix, 1024 * 4096 * 4096,
    "projector consumes merged tokens, not full 4096 patch positions");
  const half = computeNodeCosts(graph, normalizeConfig(config), {
    batch: 1, sequence: 16, visionTokens: 512, phase: "prefill",
  });
  const halfRow = id => half.find(candidate => candidate.node.id === id);
  assert.equal(halfRow("vision_tower.patch_embed.proj").actions.matrix,
    2048 * 1024 * (3 * 14 * 14));
  assert.equal(halfRow("mm_projector.proj.0").actions.matrix, 512 * 4096 * 4096);
});

for (const variant of ["Qwen3.8-27B", "Qwen3.8-Flash-Next", "Qwen3.5-122B-A10B"]) {
  test(`${variant} vision tower follows the published visual.blocks layout`, () => {
    const modelId = `Qwen/${variant}`;
    const config = read(new URL(`${modelId}/config.json`, root));
    const sourceRef = read(new URL(`${modelId}/source-ref.json`, root));
    const graph = buildStructureFromArtifacts({ modelId, config, sourceRef }).graph;
    const ids = new Set(graph.nodes.map(node => node.canonical_id));
    for (const id of [
      "visual.patch_embed.proj",
      "visual.pos_embed",
      "visual.rotary_pos_emb",
      "visual.blocks.0.norm1",
      "visual.blocks.0.norm2",
      "visual.blocks.0.attn.qkv",
      "visual.blocks.0.attn.proj",
      "visual.blocks.0.mlp.linear_fc1",
      "visual.blocks.0.mlp.linear_fc2",
      "visual.merger.norm",
      "visual.merger.linear_fc1",
      "visual.merger.linear_fc2",
    ]) assert.ok(ids.has(id), `${modelId}: missing ${id}`);
    assert.equal(ids.has("visual.0.qkv_proj"), false);
    assert.equal(ids.has("visual.0.qkv_split"), false);
    const node = id => graph.nodes.find(candidate => candidate.canonical_id === id);
    const n = normalizeConfig(config);
    assert.equal(n.visionPatchTokens, n.visionPositionCount);
    assert.deepEqual(node("visual.pos_embed").attributes.weightMatrices[0].shape,
      [n.visionPositionCount, n.visionHiddenSize]);
    assert.deepEqual(node("visual.patch_embed.proj").attributes.weightMatrices[0].shape,
      [n.visionHiddenSize, 3 * n.visionTemporalPatchSize * n.visionPatchSize * n.visionPatchSize]);
    assert.equal(node("visual.blocks.0.norm1").attributes.affine_bias, true);
    assert.equal(node("visual.blocks.0.attn.qkv").attributes.bias, true);
    assert.equal(node("visual.blocks.0.attn.rope").attributes.position_encoding, "rope_3d");
    assert.equal(node("visual.blocks.0.sdpa").attributes.attention_mask_kind, "bidirectional");
    assert.deepEqual(node("visual.merger.norm").input_shape, [-1, -1, n.visionHiddenSize]);
    assert.deepEqual(node("visual.merger.patch_merge").output_shape, [-1, -1,
      n.visionHiddenSize * n.visionMergeSize * n.visionMergeSize]);
    assert.ok(graph.edges.some(edge =>
      edge.source_canonical_id === "visual.merger.norm"
      && edge.target_canonical_id === "visual.merger.patch_merge"));
    assert.ok(graph.edges.some(edge =>
      edge.source_canonical_id === "visual.blocks.0.attn.qkv_reshape"
      && edge.target_canonical_id === "visual.blocks.0.sdpa"));
    assert.ok(graph.edges.some(edge =>
      edge.source_canonical_id === "visual.rotary_pos_emb"
      && edge.target_canonical_id === "visual.blocks.0"
      && edge.relation === "index-control"));
  });
}
