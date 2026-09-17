// Admin console (B13-T05): threshold profile, shadow mode, voiceprint enrollment.

import { api, decodeTo16k, getSettings, uuid, wavBase64 } from "../lib/api.js";
import { ACTION_COPY, escapeHtml } from "../lib/state.js";

export function mount(root, { toast }) {
  const { tenant } = getSettings();
  let profile = null;

  root.innerHTML = `
    <div class="grid">
      <section class="card">
        <h3>Alerting mode</h3>
        <div id="shadow"></div>
      </section>
      <section class="card wide">
        <h3>Threshold profile</h3>
        <div id="bands"></div>
        <details><summary>Edit profile (JSON)</summary>
          <textarea id="profilejson" rows="18" spellcheck="false"></textarea>
          <button id="saveprofile" class="primary">Save profile</button>
        </details>
      </section>
      <section class="card wide">
        <h3>Enrolled voiceprints</h3>
        <p class="small muted">Voiceprints are biometric data. Enrol only with the customer's explicit, recorded consent.</p>
        <div id="speakers"></div>
        <form id="enroll" class="stack">
          <label>Customer / speaker ID <input name="speaker" required></label>
          <label>Consent record reference <input name="consent" required placeholder="e.g. CONSENT-2026-0917-0042"></label>
          <label>Recording (at least 8 seconds of clear speech) <input type="file" name="audio" accept="audio/*" required></label>
          <label class="small"><input type="checkbox" name="replace"> replace existing voiceprint</label>
          <button class="primary">Enrol voiceprint</button>
        </form>
      </section>
    </div>`;
  const $ = (id) => root.querySelector(`#${id}`);

  function renderProfile() {
    if (!profile) return;
    $("shadow").innerHTML = `
      <p>${profile.shadow_mode
        ? "<span class='badge tone-neutral'>Shadow mode</span> Calls are scored and recorded, but <strong>no alerts are sent</strong>. Use this while calibrating."
        : "<span class='badge tone-ok'>Live</span> Alerts are sent to agents and connected systems."}</p>
      <button id="toggleshadow" class="${profile.shadow_mode ? "primary" : ""}">${profile.shadow_mode ? "Go live (send alerts)" : "Switch to shadow mode"}</button>`;
    const tiers = { low: "Low-sensitivity requests (e.g. balance)", medium: "Medium", high: "High-value / privileged requests" };
    $("bands").innerHTML = `
      <p>Profile <code>${escapeHtml(profile.name)}</code></p>
      <table class="bands"><tr><th>Request type</th><th>Caution from</th><th>High risk from</th><th>Actions at high risk</th></tr>
      ${Object.entries(profile.bands).map(([tier, b]) => `<tr><td>${escapeHtml(tiers[tier] ?? tier)}</td>
        <td>${Math.round(b.elevated * 100)}</td><td>${Math.round(b.high * 100)}</td>
        <td>${(profile.actions?.[tier]?.HIGH ?? []).map((a) => `<span class="chip">${escapeHtml(ACTION_COPY[a] ?? a)}</span>`).join(" ")}</td></tr>`).join("")}
      </table>`;
    $("profilejson").value = JSON.stringify(profile, null, 2);
  }

  async function loadProfile() {
    try {
      profile = await api("GET", `/v1/tenants/${encodeURIComponent(tenant)}/profile`);
      renderProfile();
    } catch (e) {
      $("bands").innerHTML = `<p class="muted">Profile unavailable (${escapeHtml(e.message)}). This needs an admin key.</p>`;
    }
  }

  async function loadSpeakers() {
    try {
      const { speakers } = await api("GET", `/v1/tenants/${encodeURIComponent(tenant)}/speakers`);
      $("speakers").innerHTML = speakers.length
        ? `<ul class="list">${speakers.map((s) => `<li><code>${escapeHtml(s)}</code> <button data-delete="${escapeHtml(s)}">Delete (erasure)</button></li>`).join("")}</ul>`
        : `<p class="muted">No voiceprints enrolled.</p>`;
    } catch (e) {
      $("speakers").innerHTML = `<p class="muted">Enrollment unavailable (${escapeHtml(e.message)}).</p>`;
    }
  }

  root.addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    try {
      if (btn.id === "toggleshadow") {
        profile = await api("POST", `/v1/tenants/${encodeURIComponent(tenant)}/shadow`, { shadow_mode: !profile.shadow_mode });
        renderProfile();
        toast(profile.shadow_mode ? "Shadow mode on — alerts are recorded, not sent." : "Live — alerts will be sent.");
      } else if (btn.id === "saveprofile") {
        profile = await api("PUT", `/v1/tenants/${encodeURIComponent(tenant)}/profile`, JSON.parse($("profilejson").value));
        renderProfile();
        toast("Profile saved.");
      } else if (btn.dataset.delete) {
        if (!window.confirm(`Permanently delete the voiceprint for ${btn.dataset.delete}?`)) return;
        await api("DELETE", `/v1/tenants/${encodeURIComponent(tenant)}/speakers/${encodeURIComponent(btn.dataset.delete)}`);
        toast("Voiceprint deleted.");
        loadSpeakers();
      }
    } catch (err) {
      toast(err.message, "error");
    }
  });

  $("enroll").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    try {
      const samples = await decodeTo16k(f.get("audio"));
      const res = await api("POST", `/v1/tenants/${encodeURIComponent(tenant)}/speakers/${encodeURIComponent(f.get("speaker"))}/enrollments`, {
        audio_base64: wavBase64(samples), encoding: "wav", consent_ref: f.get("consent"), session_ref: `ui-${uuid()}`,
        replace: f.get("replace") === "on",
      });
      toast(`Enrolled (${res.sessions} recording${res.sessions === 1 ? "" : "s"}).`);
      e.target.reset();
      loadSpeakers();
    } catch (err) {
      const detail = (() => {
        try { return JSON.parse(err.message).reasons.join(" "); } catch { return err.message; }
      })();
      toast(`Not enrolled: ${detail}`, "error");
    }
  });

  loadProfile();
  loadSpeakers();
}
