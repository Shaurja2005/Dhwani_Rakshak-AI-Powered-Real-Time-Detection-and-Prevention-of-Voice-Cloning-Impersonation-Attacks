// Pure view-model logic for the VoiceGuard agent / analyst UI (B13).
// No DOM access here, so it runs unchanged in the browser and under `node --test`.
//
// Copy rules (invariant I1): advisory, plain language for a frontline agent,
// never "fraud"/"fake" verdicts, and ABSTAIN never implies the caller is safe.

export const STATE_COPY = {
  LOW: {
    label: "Low risk",
    tone: "ok",
    help: "No strong signs this caller is using a cloned voice or a scam script. Follow your normal checks.",
  },
  ELEVATED: {
    label: "Caution",
    tone: "warn",
    help: "Some signs this caller may not be who they claim to be. Verify them before acting on any request.",
  },
  HIGH: {
    label: "High risk — verify the caller",
    tone: "danger",
    help: "Strong signs this caller may not be who they claim to be. Do not act until they are verified another way.",
  },
  ABSTAIN: {
    label: "Not enough audio to check",
    tone: "neutral",
    help: "The audio is too poor or too short to check the voice. This is NOT a sign the caller is genuine — follow standard verification.",
  },
};

export const HEAD_NAMES = {
  A: "Synthetic-voice detector",
  B: "Audio artifacts & room echo",
  C: "Speech rhythm & breathing",
  D: "Voice vs. enrolled customer",
  E: "Live challenge",
  F: "AI watermark",
};

export const LABEL_COPY = {
  credential_request: "Asked for OTP / PIN",
  secrecy_demand: "Asked for secrecy",
  manufactured_urgency: "Pushing for speed",
  authority_pressure: "Invoking authority",
  callback_resistance: "Avoiding a call back",
  unusual_beneficiary: "New / unusual recipient",
  unusual_channel: "Moving to another app",
  payment_request: "Asking for a payment",
};

export const ACTION_COPY = {
  agent_banner: "Show caution banner",
  force_callback: "Call back on registered number",
  hold_transaction: "Hold the transaction",
  supervisor_escalate: "Escalate to supervisor",
  soc_ticket: "Security ticket raised",
  flag_recording: "Recording flagged",
  send_sms: "SMS to customer",
  send_email: "Email to customer",
  push_notification: "App notification",
  siem_webhook: "Security team notified",
  step_up_mfa: "Approve in banking app",
  dual_approval: "Needs a second approver",
};

export function stateCopy(state) {
  return STATE_COPY[state] ?? STATE_COPY.ABSTAIN;
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function createCallModel(sessionId = null) {
  return {
    sessionId,
    risk: null, // latest SessionRisk
    timeline: [], // [{window_id, p_spoof, state}]
    contributions: {}, // head -> share (latest scored window)
    abstainedHeads: [],
    decisions: [], // [{decision, agent_prompt, band}]
    context: null, // latest ContextSignals
    transcript: [], // [{start_ms, end_ms, text}] sent by this UI
    ended: false,
  };
}

/** Fold one server event into the call model (returns a new object). */
export function reduce(model, event) {
  const m = { ...model };
  const d = event?.data ?? {};
  switch (event?.type) {
    case "session_started":
      m.sessionId = d.session_id;
      break;
    case "window_score":
      m.timeline = [...m.timeline, { window_id: d.window_id, p_spoof: d.p_spoof, state: d.state }];
      if (d.state !== "ABSTAIN") m.contributions = d.contributions ?? {};
      m.abstainedHeads = d.heads_abstained ?? [];
      break;
    case "session_risk":
      m.risk = d;
      m.sessionId = m.sessionId ?? d.session_id;
      break;
    case "context_signals":
      m.context = d;
      break;
    case "policy_decision":
      m.decisions = [...m.decisions, { decision: d, agent_prompt: event.agent_prompt ?? "", band: event.band ?? null }];
      break;
    default:
      break;
  }
  return m;
}

/** The single alert the agent should see right now, or null. */
export function currentAlert(model) {
  const last = model.decisions.at(-1);
  if (!last) return null;
  const actions = last.decision.actions ?? [];
  if (!actions.length && last.band !== "ABSTAIN") return null;
  return {
    band: last.band ?? model.risk?.state ?? "ELEVATED",
    prompt: last.agent_prompt,
    actions: actions.map((a) => ACTION_COPY[a] ?? a),
    rawActions: actions,
    suppressed: Boolean(last.decision.suppressed),
    bundleId: last.decision.evidence_bundle_id,
  };
}

/** SVG path for a semicircular gauge arc from 0 to score (0..100). */
export function gaugeArc(score, radius = 80, cx = 100, cy = 100) {
  const s = Math.max(0, Math.min(100, Number(score) || 0));
  const angle = Math.PI * (1 - s / 100);
  const x = cx + radius * Math.cos(angle);
  const y = cy - radius * Math.sin(angle);
  const large = 0;
  return `M ${cx - radius} ${cy} A ${radius} ${radius} 0 ${large} 1 ${x.toFixed(2)} ${y.toFixed(2)}`;
}

/** Polyline path for the score timeline; abstained windows break the line. */
export function sparklinePath(points, width = 300, height = 60) {
  if (!points.length) return "";
  const n = Math.max(points.length - 1, 1);
  let d = "";
  let pen = false;
  points.forEach((p, i) => {
    if (p.state === "ABSTAIN") {
      pen = false;
      return;
    }
    const x = (i / n) * width;
    const y = height - Math.max(0, Math.min(1, p.p_spoof)) * height;
    d += `${pen ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)} `;
    pen = true;
  });
  return d.trim();
}

/** Head contribution bars, largest first, with readable names and abstained heads listed separately. */
export function contributionBars(contributions, abstained = []) {
  const bars = Object.entries(contributions ?? {})
    .map(([head, share]) => ({ head, name: HEAD_NAMES[head] ?? head, pct: Math.round(share * 100) }))
    .sort((a, b) => b.pct - a.pct);
  const idle = abstained.filter((h) => !(h in (contributions ?? {}))).map((h) => ({ head: h, name: HEAD_NAMES[h] ?? h }));
  return { bars, idle };
}

/**
 * Split a transcript line into plain and highlighted parts using intent snippets.
 * A snippet highlights the part of the line it quotes.
 */
export function highlightTranscript(text, snippets = []) {
  const parts = [];
  let rest = String(text ?? "");
  const hits = snippets
    .map((s) => ({ label: s.label, at: rest.toLowerCase().indexOf(String(s.text ?? "").toLowerCase()), text: s.text }))
    .filter((h) => h.text && h.at >= 0)
    .sort((a, b) => a.at - b.at);
  let cursor = 0;
  for (const h of hits) {
    if (h.at < cursor) continue;
    if (h.at > cursor) parts.push({ text: rest.slice(cursor, h.at) });
    parts.push({ text: rest.slice(h.at, h.at + h.text.length), label: h.label, copy: LABEL_COPY[h.label] ?? h.label });
    cursor = h.at + h.text.length;
  }
  if (cursor < rest.length) parts.push({ text: rest.slice(cursor) });
  return parts;
}

/** Window index (0-based) for a playback time in seconds, given 3 s windows with 1 s hop. */
export function windowAtTime(seconds, windowCount, hopS = 1, windowS = 3) {
  if (windowCount <= 0 || seconds < 0) return -1;
  const idx = Math.floor(Math.max(0, seconds - windowS) / hopS);
  return Math.min(idx, windowCount - 1);
}

export function formatPercent(p) {
  return `${Math.round((Number(p) || 0) * 100)}%`;
}
