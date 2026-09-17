// Small render helpers shared by the views. All dynamic text goes through escapeHtml.

import {
  STATE_COPY,
  contributionBars,
  currentAlert,
  escapeHtml,
  formatPercent,
  gaugeArc,
  highlightTranscript,
  LABEL_COPY,
  sparklinePath,
  stateCopy,
} from "./state.js";

export function stateBadge(state) {
  const c = stateCopy(state);
  return `<span class="badge tone-${c.tone}" role="status">${escapeHtml(c.label)}</span>`;
}

export function gauge(risk) {
  const state = risk?.state ?? "ABSTAIN";
  const c = stateCopy(state);
  const score = state === "ABSTAIN" ? 0 : risk?.risk_score ?? 0;
  const value = state === "ABSTAIN" ? "—" : String(score);
  return `
    <figure class="gauge tone-${c.tone}" aria-label="Risk ${escapeHtml(c.label)} ${escapeHtml(value)}">
      <svg viewBox="0 0 200 118" role="img">
        <path class="gauge-track" d="${gaugeArc(100)}" />
        ${state === "ABSTAIN" ? "" : `<path class="gauge-fill" d="${gaugeArc(score)}" />`}
        <text x="100" y="92" class="gauge-value">${escapeHtml(value)}</text>
        <text x="100" y="112" class="gauge-caption">risk score</text>
      </svg>
      <figcaption>${stateBadge(state)}<p class="help">${escapeHtml(c.help)}</p></figcaption>
    </figure>`;
}

export function timeline(points, { cursor = -1, width = 320, height = 64 } = {}) {
  if (!points.length) return `<p class="muted">Waiting for the first 3 seconds of speech…</p>`;
  const n = Math.max(points.length - 1, 1);
  const x = cursor >= 0 ? (cursor / n) * width : -1;
  const hi = height - 0.7 * height;
  const el = height - 0.4 * height;
  return `
    <svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Risk over the call">
      <line x1="0" x2="${width}" y1="${hi}" y2="${hi}" class="spark-high" />
      <line x1="0" x2="${width}" y1="${el}" y2="${el}" class="spark-elevated" />
      <path d="${sparklinePath(points, width, height)}" class="spark-line" />
      ${x >= 0 ? `<line x1="${x}" x2="${x}" y1="0" y2="${height}" class="spark-cursor" />` : ""}
    </svg>`;
}

export function headBars(contributions, abstained) {
  const { bars, idle } = contributionBars(contributions, abstained);
  if (!bars.length && !idle.length) return `<p class="muted">No detector has scored yet.</p>`;
  return `
    <ul class="bars">
      ${bars
        .map(
          (b) => `<li><span class="bar-name">${escapeHtml(b.name)}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${b.pct}%"></span></span>
            <span class="bar-pct">${b.pct}%</span></li>`,
        )
        .join("")}
    </ul>
    ${idle.length ? `<p class="muted small">Not contributing right now: ${idle.map((i) => escapeHtml(i.name)).join(", ")}</p>` : ""}`;
}

export function alertBanner(model) {
  const a = currentAlert(model);
  if (!a) return "";
  const tone = stateCopy(a.band).tone;
  return `
    <section class="alert tone-${tone}" role="alert" aria-live="assertive">
      <div>
        <h2>${escapeHtml(stateCopy(a.band).label)}</h2>
        <p>${escapeHtml(a.prompt || stateCopy(a.band).help)}</p>
        ${a.actions.length ? `<p class="small">Recommended: ${a.actions.map(escapeHtml).join(" · ")}</p>` : ""}
        ${a.suppressed ? `<p class="small shadow-note">Shadow mode: this alert is recorded for review and was not sent to anyone.</p>` : ""}
      </div>
      <div class="alert-actions">
        <button data-action="callback" class="primary">Verify by call-back</button>
        <button data-action="challenge">Trigger challenge phrase</button>
      </div>
    </section>`;
}

export function intentPanel(context, transcript) {
  const labels = context?.intent_labels ?? [];
  const snippets = context?.transcript_snippets ?? [];
  const chips = labels.length
    ? labels.map((l) => `<span class="chip">${escapeHtml(LABEL_COPY[l] ?? l)}</span>`).join("")
    : `<span class="muted">No pressure tactics noticed so far.</span>`;
  const lines = (transcript.length ? transcript : snippets).map((seg) => {
    const parts = highlightTranscript(seg.text, snippets);
    return `<p class="line"><span class="ts">${formatClock(seg.start_ms)}</span> ${parts
      .map((p) => (p.label ? `<mark title="${escapeHtml(p.copy)}">${escapeHtml(p.text)}</mark>` : escapeHtml(p.text)))
      .join("")}</p>`;
  });
  return `
    <div class="chips">${chips}</div>
    <div class="transcript" aria-live="polite">${lines.join("") || `<p class="muted">Transcript appears here (personal numbers are hidden).</p>`}</div>
    ${context ? `<p class="small muted">Language: ${escapeHtml(context.language_detected)}${context.code_switched ? " (mixed languages)" : ""} · intent risk ${formatPercent(context.intent_risk)}</p>` : ""}`;
}

export function feedbackButtons(bundleId) {
  if (!bundleId) return "";
  return `
    <div class="feedback" data-bundle="${escapeHtml(bundleId)}">
      <span class="small">Was this alert right?</span>
      <button data-feedback="true_positive">Yes, it was a real problem</button>
      <button data-feedback="false_positive">No, the caller was genuine</button>
    </div>`;
}

export function formatClock(ms) {
  const s = Math.floor((Number(ms) || 0) / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function legend() {
  return Object.entries(STATE_COPY)
    .map(([k]) => stateBadge(k))
    .join(" ");
}
