// Demo mode (B13-T07): genuine vs cloned call side by side with synchronized scoring.

import { api, callMetadata, decodeTo16k, getSettings, wavBase64 } from "../lib/api.js";
import { gauge, timeline } from "../lib/components.js";
import { escapeHtml, stateCopy, windowAtTime } from "../lib/state.js";

export function mount(root, { toast }) {
  const sides = { genuine: null, cloned: null };

  root.innerHTML = `
    <p class="lead">Play a genuine recording and a cloned one of the same voice together, and watch how the scores differ.
      Use only voices of people who have consented to being cloned (see docs/ETHICS.md).</p>
    <div class="grid two">
      ${["genuine", "cloned"].map((s) => `
        <section class="card" id="side-${s}">
          <h3>${s === "genuine" ? "Genuine voice" : "Cloned voice"}</h3>
          <input type="file" accept="audio/*" data-side="${s}">
          <div class="result"><p class="muted">Choose an audio file.</p></div>
        </section>`).join("")}
    </div>
    <p><button id="play" class="primary" disabled>Play both</button> <button id="stop">Stop</button></p>`;

  function renderSide(name, t = -1) {
    const side = sides[name];
    const el = root.querySelector(`#side-${name} .result`);
    if (!side) return;
    const idx = t < 0 ? side.points.length - 1 : windowAtTime(t, side.points.length);
    const risk = idx >= 0 ? { state: side.points[idx].state, risk_score: Math.round(side.points[idx].p_spoof * 100) } : null;
    el.innerHTML = `
      ${gauge(t < 0 ? side.risk : risk)}
      ${timeline(side.points, { cursor: idx, width: 320, height: 64 })}
      <p class="small">Final: <strong>${escapeHtml(stateCopy(side.risk?.state).label)}</strong></p>`;
    el.appendChild(side.audio);
  }

  async function analyse(name, file) {
    const el = root.querySelector(`#side-${name} .result`);
    el.innerHTML = `<p class="muted">Analysing…</p>`;
    try {
      const samples = await decodeTo16k(file);
      const { tenant } = getSettings();
      const res = await api("POST", "/v1/analyze/file", {
        call_metadata: callMetadata(tenant, { channel: "file" }),
        audio_base64: wavBase64(samples),
      });
      const audio = new Audio(URL.createObjectURL(file));
      audio.controls = true;
      audio.addEventListener("timeupdate", () => renderSide(name, audio.currentTime));
      sides[name] = { risk: res.session_risk, points: res.windows.map((w) => ({ p_spoof: w.p_spoof, state: w.state })), audio };
      renderSide(name);
      root.querySelector("#play").disabled = !(sides.genuine && sides.cloned);
    } catch (e) {
      el.innerHTML = `<p class="muted">Could not analyse: ${escapeHtml(e.message)}</p>`;
      toast(e.message, "error");
    }
  }

  root.addEventListener("change", (e) => {
    const input = e.target.closest("input[type=file]");
    if (input?.files[0]) analyse(input.dataset.side, input.files[0]);
  });
  root.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    const both = [sides.genuine?.audio, sides.cloned?.audio].filter(Boolean);
    if (btn.id === "play") {
      both.forEach((a) => { a.currentTime = 0; });
      both.forEach((a) => a.play());
    } else if (btn.id === "stop") {
      both.forEach((a) => a.pause());
    }
  });
}
