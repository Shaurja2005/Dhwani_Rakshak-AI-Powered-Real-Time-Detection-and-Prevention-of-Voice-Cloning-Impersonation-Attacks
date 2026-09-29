# 3-minute video: script and shot list (B18-T04)

Record from a real rehearsal on the demo laptop. Use screen capture plus a voice-over. **Every
number on screen carries its REPORT.md row id** in small type in a corner. The cloned voice
belongs to a consenting volunteer, with the consent form shown for 2 seconds.

| # | Time | Picture | Voice-over |
|---|---|---|---|
| 1 | 0:00–0:15 | News headline montage (voice-clone bank fraud), then the title | "Thirty seconds of someone's voice is now enough to clone it. Banks verify customers by voice every day." |
| 2 | 0:15–0:35 | Architecture diagram (docs/ARCHITECTURE.md) building up layer by layer | "VoiceGuard listens to the call on the bank's own servers and runs five independent defences." |
| 3 | 0:35–1:20 | Demo 1: consent form → recording → clone → agent screen with gauge, banner and prompt | "A volunteer, a consented clone, an urgent transfer request. The agent sees one instruction: call back on the registered number." |
| 4 | 1:20–1:55 | Demo 2: human reads the fraud script. Acoustic LOW, intent HIGH. | "A real human reading a scam script: the voice is genuine, but the conversation isn't. Layered defence." |
| 5 | 1:55–2:20 | Headline chart: per-language EER before and after Indic fine-tuning (row ids visible) | "Detectors trained on English collapse on Indian languages and phone codecs. Here is the before and after, measured, not quoted from a paper." |
| 6 | 2:20–2:40 | Grafana ops dashboard with replayed traffic; DPIA page; `pytest -m compliance` passing | "Built to be deployed: on-prem, no audio stored, consent enforced, audited." |
| 7 | 2:40–3:00 | The two framing statements as text, then the logo | "Detection is advisory, never the only gate. And the strongest defences are partly not acoustic at all." |

Do not show:
- a real customer's data;
- any number that is not in REPORT.md;
- untrained-model output presented as a result. If the heads are untrained, the video waits.
