import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";

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

  if (projectorType.includes("patchmerger") && mergedWidth > 0) {
    const mergedShape = `[batch, merged visual tokens, merged width=${mergedWidth}]`;
    const children = [
      operatorSpec("projector.pre_norm", "projector LayerNorm", "rmsnorm", {
        ...shapeFlow(`[batch, visual tokens, mm hidden=${mmHidden}]`, `[batch, visual tokens, mm hidden=${mmHidden}]`),
        modality: "vision", affine_bias: true,
      }, { input: [-1, -1, mmHidden], output: [-1, -1, mmHidden] }),
      operatorSpec("projector.fc1", "projector first projection", "linear", {
        ...shapeFlow(mergedShape, mergedShape), modality: "vision", bias: true,
      }, { input: [-1, -1, mergedWidth], output: [-1, -1, mergedWidth] }),
      operatorSpec("projector.activation", "projector GELU", "vision_activation", {
        ...shapeFlow(mergedShape, mergedShape), modality: "vision",
      }, { input: [-1, -1, mergedWidth], output: [-1, -1, mergedWidth] }),
      operatorSpec("projector.fc2", "vision-text projection", "linear", {
        ...shapeFlow(mergedShape, shapes ? shapes.hidden : `[batch, tokens, hidden=${textHidden}]`),
        modality: "vision", bias: true,
      }, { input: [-1, -1, mergedWidth], output: dims?.hidden }),
    ];
    return withShapeDims(moduleSpec("projector", "Multi-modal Projector", "projector", {
      class: "PatchMergerMLP",
      mm_hidden_size: mmHidden,
      merge_kernel_size: mergeSize,
      merged_width: mergedWidth,
      // 出处见文件头注释（MoonViT 的 PatchMergerMLP）。这里只留类名，不写家族名 ——
      // §8.1：家族名不得出现在非注释代码里。
      implementation: ["PatchMergerMLP"],
      dataflow_edges: [["pre_norm", "fc1"], ["fc1", "activation"], ["activation", "fc2"]],
      ...flow,
    }, children), dims?.visionOutput, dims?.hidden);
  }

  return withShapeDims(moduleSpec("projector", "Multi-modal Projector", "projector", { class: "Projector", ...flow }, [
    operatorSpec("projector.linear", "vision-text projection", "linear", { ...flow, modality: "vision" }, { input: dims?.visionOutput, output: dims?.hidden }),
  ]), dims?.visionOutput, dims?.hidden);
}
