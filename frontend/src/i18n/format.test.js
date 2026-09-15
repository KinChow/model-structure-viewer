import assert from "node:assert/strict";
import test from "node:test";
import { catalogKeys, formatIssue, t } from "./format.js";
import { badgeText, nodeBadges } from "../diagram/nodeBadges.js";
import { normalizeParallelPlan } from "../cost/parallelPlan.js";

const HAN = /\p{Script=Han}/u;

test("en and zh catalogs have the same keys", () => {
  assert.deepEqual(catalogKeys("en"), catalogKeys("zh"));
});

test("English catalog contains no Han", () => {
  for (const key of catalogKeys("en")) {
    assert.doesNotMatch(t("en", key), HAN, key);
  }
});

test("DSpark node badges do not render billing zero as a count", () => {
  const node = {
    name: "DSpark",
    type: "dspark",
    repeat: 0,
    attributes: { modules: 1, stages: 3 },
    children: Array.from({ length: 9 }, (_, index) => ({ id: String(index) })),
  };
  const zh = badgeText(node, "zh");
  const en = badgeText(node, "en");
  assert.equal(nodeBadges(node, "zh").some((badge) => badge.kind === "repeat"), false);
  assert.match(zh, /投机头/);
  assert.match(zh, /9 个子模块/);
  assert.doesNotMatch(zh, /×0/);
  assert.doesNotMatch(zh, /0 9/);
  assert.match(en, /draft/);
  assert.match(en, /9 children/);
  assert.doesNotMatch(en, HAN);
});

test("repeat greater than 1 still shows a multiplier", () => {
  const node = { repeat: 61, children: [{ id: "0" }] };
  assert.equal(nodeBadges(node, "zh").find((badge) => badge.kind === "repeat").text, "×61");
});

test("plan errors format in English without Han", () => {
  const { errors } = normalizeParallelPlan({ tp: 2, pp: 2, dp: 2, worldSize: 4 });
  const english = formatIssue("en", errors[0]);
  assert.match(english, /world_size/);
  assert.doesNotMatch(english, HAN);
  assert.match(formatIssue("zh", errors[0]), /应为/);
});

test("HTTP and chip load issues format without leaking Han in English", () => {
  const verify = formatIssue("en", { code: "http.verifyUnavailable" });
  assert.match(verify, /msv serve/);
  assert.doesNotMatch(verify, HAN);
  assert.match(formatIssue("zh", { code: "http.verifyUnavailable" }), /后端不可用/);
  assert.match(formatIssue("en", { code: "http.backendUnavailable", params: { status: 502 } }), /HTTP 502/);
  assert.doesNotMatch(formatIssue("en", { code: "chip.localHttp", params: { status: 500 } }), HAN);
  assert.match(formatIssue("zh", { code: "chip.localShape" }), /chips/);
});

test("PD link source and lens missing-input codes format in both languages", () => {
  assert.equal(t("en", "comm.bothInterNode"), "both sides inter_node");
  assert.doesNotMatch(t("en", "comm.minAvailable"), HAN);
  assert.doesNotMatch(t("en", "comm.missingLink"), HAN);
  assert.match(t("zh", "comm.missingLink"), /缺少/);
  assert.doesNotMatch(formatIssue("en", { code: "lens.missingInputs" }), HAN);
  assert.match(formatIssue("zh", { code: "lens.missingInputs" }), /缺少结构/);
});
