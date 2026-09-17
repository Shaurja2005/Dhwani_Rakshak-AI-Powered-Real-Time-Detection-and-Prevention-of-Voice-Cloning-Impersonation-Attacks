// Post-call forensics view (B13-T04, T06): timeline scrub, evidence bundle, model versions, PDF export.

import { api } from "../lib/api.js";
import { feedbackButtons, headBars, stateBadge, timeline } from "../lib/components.js";
import { ACTION_COPY, LABEL_COPY, escapeHtml, formatPercent } from "../lib/state.js";

export function mount(root, { toast, params }) {
  let windows = [];
  let bundles = [];
  let cursor = 0;

  root.innerHTML = `
    <section class="toolbar no-print">
      <label>Session ID <input id="sid" size="40" placeholder="paste a session id" value="${escapeHtml(params.get("session") ?? "")}"></label>
      <button id="load" class="primary">Load call</button>
      <button id="print">Export to PDF</button>
    </section>
    <div id="body"><p class="muted">Load a call to review its score timeline and evidence.</p></div>`;
  const $ = (id) => root.querySelector(`#${id}`);

  function render() {
    if (!windows.length && !bundles.length) return;
    const w = windows[cursor] ?? {};
    const bundle = bundles.at(-1);
    const pd = bundle?.policy_decision ?? {};
    const risk = bundle?.session_risk ?? {};
    const ctx = bundle?.context_signals;
    $("body").innerHTML = `
      <div class="grid">
        <section class="card wide">
          <h3>Score timeline</h3>
          ${timeline(windows.map((x) => ({ p_spoof: x.p_window ?? 0, state: x.window_state })), { cursor, width: 640, height: 90 })}
          <input type="range" id="scrub" min="0" max="${Math.max(windows.length - 1, 0)}" value="${cursor}" class="no-print" aria-label="Scrub the call">
          <p>Window ${cursor + 1} of ${windows.length} · ${stateBadge(w.window_state ?? "ABSTAIN")}
             · window score ${w.p_window == null ? "—" : formatPercent(w.p_window)}
             · call-level risk ${w.risk_score ?? "—"} (${stateBadge(w.state ?? "ABSTAIN")})</p>
          ${headBars(w.contributions ?? {}, w.heads_abstained ?? [])}
        </section>
        <section class="card">
          <h3>Decision</h3>
          ${bundle ? `<p>${stateBadge(risk.state)} risk score <strong>${escapeHtml(risk.risk_score)}</strong></p>
          <p>Profile <code>${escapeHtml(pd.threshold_profile)}</code>${pd.shadow_mode ? " · <span class='chip'>shadow mode</span>" : ""}</p>
          <p>Actions: ${(pd.actions ?? []).map((a) => `<span class="chip">${escapeHtml(ACTION_COPY[a] ?? a)}</span>`).join(" ") || "none"}</p>
          <p class="small">Evidence <code>${escapeHtml(bundle.bundle_id)}</code> ${bundle._verified ? "<span class='badge tone-ok'>integrity verified</span>" : "<span class='badge tone-danger'>integrity check failed</span>"}</p>` : `<p class="muted">No decision recorded.</p>`}
        </section>
        <section class="card">
          <h3>Models used</h3>
          <table class="kv">${Object.entries(bundle?.model_versions ?? {}).map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td><code>${escapeHtml(v)}</code></td></tr>`).join("") || "<tr><td class='muted'>—</td></tr>"}</table>
        </section>
        <section class="card wide">
          <h3>Why (full rationale)</h3>
          ${ctx ? `<p>${(ctx.intent_labels ?? []).map((l) => `<span class="chip">${escapeHtml(LABEL_COPY[l] ?? l)}</span>`).join(" ")}</p>` : ""}
          <pre class="rationale">${escapeHtml(pd.rationale ?? "")}</pre>
        </section>
      </div>
      <div class="no-print">${feedbackButtons(bundle?.bundle_id)}</div>`;
    $("scrub")?.addEventListener("input", (e) => {
      cursor = Number(e.target.value);
      render();
    });
  }

  async function load() {
    const sid = $("sid").value.trim();
    if (!sid) return;
    try {
      const [tl, ev] = await Promise.all([
        api("GET", `/v1/sessions/${encodeURIComponent(sid)}/timeline`).catch(() => ({ windows: [] })),
        api("GET", `/v1/sessions/${encodeURIComponent(sid)}/evidence`),
      ]);
      windows = tl.windows;
      bundles = ev.bundles;
      for (const b of bundles) {
        const check = await api("GET", `/v1/evidence/${encodeURIComponent(b.bundle_id)}`);
        b._verified = check.verified;
      }
      cursor = Math.max(windows.length - 1, 0);
      if (!windows.length && !bundles.length) $("body").innerHTML = `<p class="muted">Nothing recorded for that session.</p>`;
      render();
    } catch (e) {
      toast(`Could not load the call: ${e.message}`, "error");
    }
  }

  root.addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    if (btn.id === "load") return load();
    if (btn.id === "print") return window.print();
    if (btn.dataset.feedback) {
      const bundle = btn.closest(".feedback").dataset.bundle;
      try {
        await api("POST", `/v1/evidence/${encodeURIComponent(bundle)}/feedback`, { label: btn.dataset.feedback, analyst: "analyst-ui" });
        toast("Feedback recorded.");
      } catch (err) {
        toast(`Feedback failed: ${err.message}`, "error");
      }
    }
  });
  if (params.get("session")) load();
}
