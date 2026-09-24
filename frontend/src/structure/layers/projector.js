import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { recipeValue } from "../archs/index.js";

/**
 * 多模态投影器。两种形态，按 `mm_projector_type` 分：
 *
 * - `patchmerger`（Kimi 系 MoonViT，models/moonshotai/Kimi-K2.5/
 *   modeling_kimi_k25.py:737-751 `PatchMergerMLP`）：
 *     pre_norm = LayerNorm(mm_hidden)            // 有 bias，权重 2·mm_hidden
 *     proj[0]  = Linear(mm_hidden·merge² → 同宽)  // 有 bias
 *     GELU
 *     proj[2]  = Linear(mm_hidden·merge² → text_hidden) // 有 bias
 *   此前只发射一条 `mm_hidden → text_hidden` 的 linear，少了合并后的宽度与两处
 *   bias/LN，K2.5 视觉侧因此少 4,600 万参数（2026-09-09 权重字节逐层归因抓出）。
 * - 其它（简单线性投影）：单条 `visionOutput → hidden`。
 */
export function projectorModule(normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.visionOutput, shapes.hidden) : {};
  const dims = normalized ? tensorDims(normalized) : null;
  const projectorType = String(normalized?.visionProjectorType || "");
  const mmHidden = normalized?.visionOutputSize || normalized?.visionHiddenSize || 0;
  const mergeSize = normalized?.visionMergeSize || 1;
  const mergedWidth = mmHidden * mergeSize * mergeSize;
  const textHidden = normalized?.hiddenSize || 0;
  // The display role remains "projector", but the canonical path follows the
  // published module name (mm_projector, multi_modal_projector, or aligner).
  const baseId = recipeValue(normalized, "visionProjectorPath") || "projector";
  const child = suffix => `${baseId}.${suffix}`;
  const projectorKind = recipeValue(normalized, "visionProjectorKind")
    || (projectorType.includes("patchmerger") ? "patchmerger" : "linear");

  if (projectorKind === "deepseek_aligner") {
    const downsample = normalized?.visionDownsampleRatio || 1;
    const alignWidth = mmHidden * downsample * downsample;
    const alignShape = `[batch, merged visual tokens, aligner width=${alignWidth}]`;
    const outputShape = shapes ? shapes.hidden : `[batch, tokens, hidden=${textHidden}]`;
    const children = [
      operatorSpec(child("w1"), "aligner first projection", "linear", {
        ...shapeFlow(alignShape, outputShape), modality: "vision", bias: true,
      }, { input: [-1, -1, alignWidth], output: dims?.hidden }),
      operatorSpec(child("activation"), "aligner GELU", "vision_activation", {
        ...shapeFlow(outputShape, outputShape), modality: "vision",
      }, { input: dims?.hidden, output: dims?.hidden }),
      operatorSpec(child("w2"), "aligner output projection", "linear", {
        ...shapeFlow(outputShape, outputShape), modality: "vision", bias: true,
      }, { input: dims?.hidden, output: dims?.hidden }),
    ];
    return withShapeDims(moduleSpec(baseId, "Multi-modal Projector", "projector", {
      class: "Aligner",
      implementation: ["Aligner"],
      dataflow_edges: [[child("w1"), child("activation")], [child("activation"), child("w2")]],
      ...flow,
    }, children), dims?.visionOutput, dims?.hidden);
  }

  if (projectorKind === "two_stage_patch_merge") {
    const projectorHidden = textHidden || mmHidden;
    // MiniMax first projects each patch to the text width, then reshapes
    // spatial_merge_size² patches into one channel vector.  Its published
    // `merged_hidden_size` is therefore text_hidden × merge², not the raw
    // vision width × merge² used by Kimi PatchMerger.
    const minimaxMergedWidth = textHidden * mergeSize * mergeSize;
    const mergedShape = `[batch, merged visual tokens, merged width=${minimaxMergedWidth}]`;
    const projectedShape = `[batch, visual tokens, projector hidden=${projectorHidden}]`;
    const firstChildren = [
      operatorSpec(child("linear_1"), "projector first projection", "linear", {
        ...shapeFlow(`[batch, visual tokens, vision hidden=${mmHidden}]`, projectedShape),
        modality: "vision", bias: true,
      }, { input: [-1, -1, mmHidden], output: [-1, -1, projectorHidden] }),
      operatorSpec(child("act"), "projector GELU", "vision_activation", {
        ...shapeFlow(projectedShape, projectedShape), modality: "vision",
      }, { input: [-1, -1, projectorHidden], output: [-1, -1, projectorHidden] }),
      operatorSpec(child("linear_2"), "projector text projection", "linear", {
        ...shapeFlow(projectedShape, `[batch, visual tokens, hidden=${textHidden}]`),
        modality: "vision", bias: true,
      }, { input: [-1, -1, projectorHidden], output: dims?.hidden }),
    ];
    const mergeBaseId = recipeValue(normalized, "visionProjectorMergePath") || "patch_merge_mlp";
    const mergeChild = suffix => `${mergeBaseId}.${suffix}`;
    const mergeChildren = [
      operatorSpec(mergeChild("linear_1"), "patch merge projection", "linear", {
        ...shapeFlow(mergedShape, projectedShape), modality: "vision", bias: true,
      }, { input: [-1, -1, minimaxMergedWidth], output: [-1, -1, projectorHidden] }),
      operatorSpec(mergeChild("act"), "patch merge GELU", "vision_activation", {
        ...shapeFlow(projectedShape, projectedShape), modality: "vision",
      }, { input: [-1, -1, projectorHidden], output: [-1, -1, projectorHidden] }),
      operatorSpec(mergeChild("linear_2"), "patch merge output projection", "linear", {
        ...shapeFlow(projectedShape, `[batch, merged visual tokens, hidden=${textHidden}]`),
        modality: "vision", bias: true,
      }, { input: [-1, -1, projectorHidden], output: dims?.hidden }),
    ];
    const first = withShapeDims(moduleSpec(baseId, "Multi-modal Projector", "projector-stage", {
      checkpoint_layout: "separate_patch_merge_mlp",
      dataflow_edges: [[child("linear_1"), child("act")], [child("act"), child("linear_2")]],
      ...flow,
    }, firstChildren), dims?.visionOutput, dims?.hidden);
    const merge = withShapeDims(moduleSpec(mergeBaseId, "Patch Merge MLP", "projector", {
      checkpoint_layout: "separate_patch_merge_mlp",
      dataflow_edges: [
        [mergeChild("linear_1"), mergeChild("act")],
        [mergeChild("act"), mergeChild("linear_2")],
      ],
      ...shapeFlow(`[batch, visual tokens, hidden=${textHidden}]`, `[batch, merged visual tokens, hidden=${textHidden}]`),
    }, mergeChildren), [-1, -1, textHidden], dims?.hidden);
    return [first, merge];
  }

  if (projectorKind === "patchmerger" && mergedWidth > 0) {
    const mergedShape = `[batch, merged visual tokens, merged width=${mergedWidth}]`;
    const patchChildren = baseId === "mm_projector"
      ? { norm: "pre_norm", fc1: "proj.0", activation: "proj.1", fc2: "proj.2" }
      : { norm: "pre_norm", fc1: "fc1", activation: "activation", fc2: "fc2" };
    const children = [
      operatorSpec(child(patchChildren.norm), "projector LayerNorm", "rmsnorm", {
        ...shapeFlow(`[batch, visual tokens, mm hidden=${mmHidden}]`, `[batch, visual tokens, mm hidden=${mmHidden}]`),
        modality: "vision", affine_bias: true,
      }, { input: [-1, -1, mmHidden], output: [-1, -1, mmHidden] }),
      operatorSpec(child(patchChildren.fc1), "projector first projection", "linear", {
        ...shapeFlow(mergedShape, mergedShape), modality: "vision", bias: true,
      }, { input: [-1, -1, mergedWidth], output: [-1, -1, mergedWidth] }),
      operatorSpec(child(patchChildren.activation), "projector GELU", "vision_activation", {
        ...shapeFlow(mergedShape, mergedShape), modality: "vision",
      }, { input: [-1, -1, mergedWidth], output: [-1, -1, mergedWidth] }),
      operatorSpec(child(patchChildren.fc2), "vision-text projection", "linear", {
        ...shapeFlow(mergedShape, shapes ? shapes.hidden : `[batch, tokens, hidden=${textHidden}]`),
        modality: "vision", bias: true,
      }, { input: [-1, -1, mergedWidth], output: dims?.hidden }),
    ];
    return withShapeDims(moduleSpec(baseId, "Multi-modal Projector", "projector", {
      class: "PatchMergerMLP",
      mm_hidden_size: mmHidden,
      merge_kernel_size: mergeSize,
      merged_width: mergedWidth,
      // 出处见文件头注释（MoonViT 的 PatchMergerMLP）。这里只留类名，不写家族名 ——
      // §8.1：家族名不得出现在非注释代码里。
      implementation: ["PatchMergerMLP"],
      dataflow_edges: [
        [child(patchChildren.norm), child(patchChildren.fc1)],
        [child(patchChildren.fc1), child(patchChildren.activation)],
        [child(patchChildren.activation), child(patchChildren.fc2)],
      ],
      ...flow,
    }, children), dims?.visionOutput, dims?.hidden);
  }

  return withShapeDims(moduleSpec(baseId, "Multi-modal Projector", "projector", { class: "Projector", ...flow }, [
    operatorSpec(child("linear"), "vision-text projection", "linear", { ...flow, modality: "vision" }, { input: dims?.visionOutput, output: dims?.hidden }),
  ]), dims?.visionOutput, dims?.hidden);
}
