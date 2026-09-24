import assert from "node:assert/strict";
import test from "node:test";
import { FORMULAS } from "../structure/operators/formulas/index.js";
import { EXPLANATIONS_EN, formulaExplanation } from "./formulaExplanations.js";

test("every registered formula has English explanation copy without Han characters", () => {
  const han = /\p{Script=Han}/u;
  for (const [id, formula] of Object.entries(FORMULAS)) {
    const explanation = formulaExplanation(id, formula.title);
    assert.ok(explanation, `${id} explanation is empty`);
    assert.doesNotMatch(explanation, han, `${id} explanation contains Han characters`);
    assert.ok(EXPLANATIONS_EN[id], `${id} must have explicit English copy, not only the fallback`);
  }
  assert.deepEqual(Object.keys(EXPLANATIONS_EN).sort(), Object.keys(FORMULAS).sort());
});
