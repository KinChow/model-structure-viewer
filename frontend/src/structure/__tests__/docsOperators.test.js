// 文档机器段一致性（W6）：`docs/details/operators_reference.md` 的生成段必须与
// 注册表 + 探针一致。这条把「表和代码各自漂移」变成 CI 可拦的失败——用户看到的
// 「漏洞百出」有一部分就是手写表与实现长期不同步造成的。
//
// 与 golden diff 测试同套路：生成物 diff 非空即失败，修法是跑
// `npm run docs:operators` 重生成。
import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

// 本文件在 frontend/src/structure/__tests__/ → 仓库根要上溯四级。
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");

test("operators_reference.md 机器段与注册表/探针一致（docs:check）", () => {
  let output = "";
  try {
    output = execFileSync(
      process.execPath,
      [path.join(repoRoot, "scripts/gen-operators-reference.mjs"), "--check"],
      { cwd: repoRoot, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
    );
  } catch (error) {
    assert.fail(`docs:check 失败——跑 \`npm run docs:operators\` 重生成机器段。\n${error.stderr || error.message}`);
  }
  assert.match(output, /与注册表\/探针一致/);
});
