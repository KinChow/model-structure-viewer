// countsAtomsConsistency.test.js —— counts ↔ 原子逐位一致护栏（M11.5 子项 3）。
//
// 背景：addCounts / softmaxCounts / hashRouteCounts 已在 counts.js 内直接委托
// atoms.js 的 add / softmax / gather（scratch 证明全字段 Object.is 相等后才委托）。
// 本测试把「证明过一次」固化为永久法则：谁改了其中一边而没改另一边，这里当场失败，
// 而不是让 313 个既有用例里的某个手算期望悄悄失真。
//
// 注意边界：swigluCounts / causalConvCounts / linearCounts **不在**护栏内 ——
// 它们与朴素原子组合**已知不同构**（actIn 操作数口径不同，是有意的融合口径），
// 委托会改变运行时输出（golden hash 基线），故保留独立实现。
import assert from "node:assert/strict";
import test from "node:test";
import { add, softmax, gather } from "../atoms.js";
import { addCounts, softmaxCounts, hashRouteCounts } from "../counts.js";

test("counts ↔ atoms 逐位一致护栏：addCounts/softmaxCounts/hashRouteCounts ≡ add/softmax/gather", () => {
  // 小网格：tokens 1/128（+ 0 边界）、hidden 512/4096、bf16/fp32/fp8 字节宽。
  // 期望侧直接调原子（同一表达式即实现侧），护栏意义在「改一边必须改另一边」。
  for (const tokens of [0, 1, 128]) {
    for (const hidden of [512, 4096]) {
      for (const bytesPerElement of [1, 2, 4]) {
        assert.deepStrictEqual(
          addCounts({ tokens, hidden, bytesPerElement }),
          add({ elements: tokens * hidden, bytesPerElement }),
          `addCounts({tokens:${tokens}, hidden:${hidden}, b:${bytesPerElement}}) 与 add 原子漂移`,
        );
      }
    }
  }

  for (const elements of [0, 1, 512, 4096, 128 * 4096]) {
    for (const bytesPerElement of [1, 2, 4]) {
      assert.deepStrictEqual(
        softmaxCounts({ elements, bytesPerElement }),
        softmax({ elements, bytesPerElement }),
        `softmaxCounts({elements:${elements}, b:${bytesPerElement}}) 与 softmax 原子漂移`,
      );
    }
  }

  // dsv4 hash 路由：读 topk 个专家 id + 写 topk 个 = gather(rows=tokens, width=topk)。
  for (const tokens of [0, 1, 128]) {
    for (const topk of [1, 8]) {
      for (const bytesPerElement of [1, 2, 4]) {
        assert.deepStrictEqual(
          hashRouteCounts({ tokens, topk, bytesPerElement }),
          gather({ rows: tokens, width: topk, bytesPerElement }),
          `hashRouteCounts({tokens:${tokens}, topk:${topk}, b:${bytesPerElement}}) 与 gather 原子漂移`,
        );
      }
    }
  }
});
