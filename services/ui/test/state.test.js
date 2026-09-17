import { test } from "node:test";
import assert from "node:assert/strict";

import {
  STATE_COPY,
  contributionBars,
  createCallModel,
  currentAlert,
  escapeHtml,
  gaugeArc,
  highlightTranscript,
  reduce,
  sparklinePath,
  stateCopy,
  windowAtTime,
} from "../public/lib/state.js";

test("copy is advisory, plain, and ABSTAIN never implies safety", () => {
  for (const [state, c] of Object.entries(STATE_COPY)) {
    assert.ok(c.label && c.help, state);
    assert.doesNotMatch(`${c.label} ${c.help}`.toLowerCase(), /\bfraud|fake|scammer\b/);
  }
  assert.match(STATE_COPY.ABSTAIN.help, /NOT a sign the caller is genuine/);
  assert.equal(stateCopy("UNKNOWN").label, STATE_COPY.ABSTAIN.label);
  assert.notEqual(STATE_COPY.ABSTAIN.tone, STATE_COPY.LOW.tone); // ABSTAIN is styled distinctly from LOW
});

test("event reducer builds the live call model", () => {
  let m = createCallModel();
  m = reduce(m, { type: "session_started", data: { session_id: "s1" } });
  m = reduce(m, { type: "window_score", data: { window_id: 0, p_spoof: 0.2, state: "LOW", contributions: { A: 0.7, B: 0.3 }, heads_abstained: ["C"] } });
  m = reduce(m, { type: "window_score", data: { window_id: 1, p_spoof: 0, state: "ABSTAIN", contributions: {}, heads_abstained: ["A", "B", "C"] } });
  m = reduce(m, { type: "session_risk", data: { state: "HIGH", risk_score: 81 } });
  m = reduce(m, { type: "policy_decision", data: { actions: ["agent_banner", "force_callback"], suppressed: true, evidence_bundle_id: "sha256:x" }, agent_prompt: "Call back.", band: "HIGH" });
  assert.equal(m.sessionId, "s1");
  assert.equal(m.timeline.length, 2);
  assert.deepEqual(m.contributions, { A: 0.7, B: 0.3 }); // an abstained window keeps the last real contributions
  const alert = currentAlert(m);
  assert.equal(alert.band, "HIGH");
  assert.deepEqual(alert.actions, ["Show caution banner", "Call back on registered number"]);
  assert.equal(alert.suppressed, true);
  assert.equal(currentAlert(createCallModel()), null);
});

test("gauge and sparkline geometry", () => {
  assert.equal(gaugeArc(0), "M 20 100 A 80 80 0 0 1 20.00 100.00");
  assert.equal(gaugeArc(100), "M 20 100 A 80 80 0 0 1 180.00 100.00");
  assert.equal(gaugeArc(50), "M 20 100 A 80 80 0 0 1 100.00 20.00");
  const d = sparklinePath([{ p_spoof: 0, state: "LOW" }, { p_spoof: 0, state: "ABSTAIN" }, { p_spoof: 1, state: "HIGH" }], 100, 50);
  assert.equal(d, "M 0.0 50.0 M 100.0 0.0"); // abstained window breaks the line
  assert.equal(sparklinePath([]), "");
});

test("contribution bars are named, sorted, and list idle heads", () => {
  const { bars, idle } = contributionBars({ B: 0.25, A: 0.75 }, ["A", "D"]);
  assert.deepEqual(bars.map((b) => [b.head, b.pct]), [["A", 75], ["B", 25]]);
  assert.equal(bars[0].name, "Synthetic-voice detector");
  assert.deepEqual(idle.map((i) => i.head), ["D"]);
});

test("transcript highlighting marks intent snippets, case-insensitively", () => {
  const parts = highlightTranscript("Please keep this confidential and act right now.", [
    { text: "right now", label: "manufactured_urgency" },
    { text: "Keep this confidential", label: "secrecy_demand" },
  ]);
  assert.deepEqual(parts.map((p) => p.label ?? null), [null, "secrecy_demand", null, "manufactured_urgency", null]);
  assert.equal(parts[1].copy, "Asked for secrecy");
  assert.deepEqual(highlightTranscript("hello", []), [{ text: "hello" }]);
});

test("helpers", () => {
  assert.equal(escapeHtml(`<img src=x onerror="alert(1)">`), "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;");
  assert.equal(windowAtTime(2.5, 10), 0);
  assert.equal(windowAtTime(5.2, 10), 2);
  assert.equal(windowAtTime(99, 10), 9);
  assert.equal(windowAtTime(1, 0), -1);
});
