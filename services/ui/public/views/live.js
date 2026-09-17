// Live call view (B13-T01, T02, T03, T06).

import { api, callMetadata, decodeTo16k, floatTo16BitPCM, getSettings, wsUrl } from "../lib/api.js";
import { alertBanner, feedbackButtons, gauge, headBars, intentPanel, timeline } from "../lib/components.js";
import { createCallModel, currentAlert, escapeHtml, reduce } from "../lib/state.js";

export function mount(root, { toast }) {
  let model = createCallModel();
  let socket = null;
  let challengeText = "";

  root.innerHTML = `
    <section class="toolbar">
      <label>Active calls <select id="sessions"><option value="">— choose a call —</option></select></label>
      <button id="refresh">Refresh</button>
      <span class="sep"></span>
      <label class="file">Simulate a call from an audio file <input type="file" id="simfile" accept="audio/*"></label>
      <label class="small"><input type="checkbox" id="simscript"> add a scripted scam transcript</label>
    </section>
    <div id="alert"></div>
    <div class="grid">
      <section class="card" id="gaugecard"></section>
      <section class="card"><h3>Risk over the call</h3><div id="timeline"></div></section>
      <section class="card"><h3>What is driving the score</h3><div id="heads"></div></section>
      <section class="card wide"><h3>Conversation</h3><div id="intent"></div></section>
    </div>
    <div id="challenge"></div>
    <div id="feedback"></div>`;

  const $ = (id) => root.querySelector(`#${id}`);

  function render() {
    $("alert").innerHTML = alertBanner(model);
    $("gaugecard").innerHTML = gauge(model.risk);
    $("timeline").innerHTML = timeline(model.timeline);
    $("heads").innerHTML = headBars(model.contributions, model.abstainedHeads);
    $("intent").innerHTML = intentPanel(model.context, model.transcript);
    $("challenge").innerHTML = challengeText
      ? `<section class="card note"><h3>Read this to the caller</h3><p class="challenge">${escapeHtml(challengeText)}</p>
         <p class="small muted">Their answer is checked automatically when the transcript arrives.</p></section>`
      : "";
    const alert = currentAlert(model);
    $("feedback").innerHTML = feedbackButtons(alert?.bundleId);
  }

  function onEvent(ev) {
    if (ev.type === "error") return toast(ev.detail, "error");
    model = reduce(model, ev);
    render();
  }

  async function refresh() {
    try {
      const { sessions } = await api("GET", "/v1/sessions");
      const sel = $("sessions");
      sel.innerHTML = `<option value="">— choose a call —</option>` + sessions
        .map((s) => `<option value="${escapeHtml(s.session_id)}">${escapeHtml(s.caller_number ?? s.channel)} · ${escapeHtml(s.state)} · started ${escapeHtml(new Date(s.started_at).toLocaleTimeString())}</option>`)
        .join("");
    } catch (e) {
      toast(`Could not list calls: ${e.message}`, "error");
    }
  }

  function watch(sessionId) {
    socket?.close();
    model = createCallModel(sessionId);
    challengeText = "";
    render();
    if (!sessionId) return;
    socket = new WebSocket(wsUrl(`/v1/sessions/${encodeURIComponent(sessionId)}/watch`));
    socket.onmessage = (m) => onEvent(JSON.parse(m.data));
    socket.onclose = () => toast("Call ended or stream closed.");
  }

  async function simulate(file) {
    socket?.close();
    const samples = await decodeTo16k(file);
    const { tenant } = getSettings();
    const meta = callMetadata(tenant, { caller_number: "+91 98xxx (simulated)" });
    model = createCallModel(meta.session_id);
    challengeText = "";
    render();
    socket = new WebSocket(wsUrl("/v1/stream"));
    socket.onmessage = (m) => onEvent(JSON.parse(m.data));
    socket.onopen = async () => {
      socket.send(JSON.stringify({ call_metadata: meta, encoding: "pcm_s16le", sample_rate: 16000 }));
      const chunk = 1600; // 100 ms, sent in real time
      for (let i = 0; i < samples.length && socket.readyState === 1; i += chunk) {
        socket.send(floatTo16BitPCM(samples.subarray(i, i + chunk)));
        if (i === chunk * 60 && $("simscript").checked) {
          const segs = [
            { start_ms: 1000, end_ms: 5000, text: "This is the CFO, I am in a board meeting so don't call me back." },
            { start_ms: 5000, end_ms: 9000, text: "Transfer 18 lakh to the new vendor account right now and keep it confidential." },
          ];
          model = { ...model, transcript: segs };
          socket.send(JSON.stringify({ type: "transcript", segments: segs }));
        }
        await new Promise((r) => setTimeout(r, 100));
      }
      if (socket.readyState === 1) socket.send(JSON.stringify({ type: "close" }));
    };
  }

  root.addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    if (btn.id === "refresh") return refresh();
    if (btn.dataset.action === "callback") {
      toast("Noted: end the call politely and call back on the number registered to the account.");
      return;
    }
    if (btn.dataset.action === "challenge") {
      if (!model.sessionId) return;
      try {
        const res = await api("POST", `/v1/sessions/${encodeURIComponent(model.sessionId)}/challenges`,
                              { kind: "digits_codeswitch", languages: ["en", "hi"] });
        challengeText = res.prompt_text;
        render();
      } catch (err) {
        toast(`Could not start a challenge: ${err.message}`, "error");
      }
      return;
    }
    if (btn.dataset.feedback) {
      const bundle = btn.closest(".feedback").dataset.bundle;
      try {
        await api("POST", `/v1/evidence/${encodeURIComponent(bundle)}/feedback`, { label: btn.dataset.feedback, analyst: "agent-ui" });
        toast("Thanks — your feedback helps tune the system.");
      } catch (err) {
        toast(`Feedback failed: ${err.message}`, "error");
      }
    }
  });
  $("sessions").addEventListener("change", (e) => watch(e.target.value));
  $("simfile").addEventListener("change", (e) => e.target.files[0] && simulate(e.target.files[0]));

  render();
  refresh();
  return () => socket?.close();
}
