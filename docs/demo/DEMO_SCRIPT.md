# Demo script: live, 5 minutes (B18-T01, T02, T05)

Two demos, one message: **the voice can be faked, so VoiceGuard never trusts the voice alone.**
Rehearse with `make demo-rehearse` until every line of the table reads `ok`. Record the offline
pack after the final rehearsal with `make demo-pack`.

## Before the day (checklist)

- [ ] Trained heads are installed and promoted (`python scripts/model_rollout.py status`). With
      untrained heads the acoustic layer **abstains**, and the demo must not pretend otherwise.
- [ ] **Consent.** Every voice that will be cloned is in `ml/data/consent_register.yaml` and
      `docs/ETHICS.md`, with a signed form stored outside git (invariant I9). A judge who
      volunteers on the day signs the same one-page form before anything is recorded. No form,
      no clone. Never clone an executive, official or public figure, even as a joke.
- [ ] `demo_assets/` (git-ignored) holds `genuine_customer.wav`, `human_fraud_script.wav`
      (a consenting team member) and a pre-made `cloned_volunteer.wav` as backup for the live clone.
- [ ] Laptop runs everything locally: `python scripts/dev_gateway.py` with
      `VG_GATEWAY_HEADS=A,B,C,F`. Wi-Fi **off** during rehearsal, to prove nothing needs it.
- [ ] `make demo-pack` is recorded, and `make demo-offline` plays it (fallback tier 3).
- [ ] Browser open at `http://localhost:8765/ui/` → **Demo** tab. Font size up. Notifications off.

## Demo 1: "clone the judge" (2 min)

| Time | Say | Do |
|---|---|---|
| 0:00 | "One volunteer, 30 seconds of your voice, with your written consent. That is all a fraudster needs." | Volunteer signs the consent form. Record 30 s with the laptop mic. |
| 0:40 | "While it clones: this is the agent's screen during a normal call." | Play `genuine_customer.wav` in the **Demo** tab (left). The gauge stays low. |
| 1:10 | "Now the clone asks for an urgent transfer and says it can't take a call back." | Clone job runs locally (`ml/data/clone_job.py`, consent-checked). Play the clone in the right panel with the transfer script. |
| 1:30 | "Synthetic-voice heads fire, *and* the conversation shows urgency, secrecy and callback resistance. The agent sees one plain instruction: call back on the registered number." | Point at the banner and agent prompt, then the head bars. |
| 1:50 | "It didn't block the customer. It added friction, which is exactly what a bank's risk team wants." | — |

**If the live clone fails** (GPU, time): "Here is one we made this morning from the same kind
of recording." Use `demo_assets/cloned_volunteer.wav`. Never hide the switch.

## Demo 2: a real human reads a fraud script (1.5 min)

| Say | Do |
|---|---|
| "Voice detectors can be fooled, and a real fraudster may use their own voice. So here is a real human, reading a real scam script." | Play `human_fraud_script.wav` with its transcript (`make demo-rehearse --only human_reads_fraud_script`, or the UI). |
| "Acoustic score: low. The voice *is* genuine. The intent layer caught 'OTP', 'right now', 'don't tell anyone'. Decision: caution, verify the caller." | Point at *acoustic LOW, intent HIGH*. |
| "Five layers: synthetic-voice detection, channel forensics, speaker verification, liveness challenge and conversation intent. When one is fooled, the others still hold." | — |

## The two things to say out loud (B18-T06)

1. **Detection is advisory, never authoritative.** Genuine and synthetic speech overlap under
   real phone conditions, so every detector has an error floor. VoiceGuard raises friction and
   triggers out-of-band verification. It is never the sole gate on a transaction.
2. **The strongest defences are partly non-acoustic.** Speaker verification, transaction
   context, transcript intent and an enforced call-back all degrade gracefully when the audio
   detector is fooled.

## Failure modes: rehearse each once

| Failure | Recovery line | Action |
|---|---|---|
| Wi-Fi dies | "Everything runs on this laptop, like it would inside a bank." | Nothing: the demo never needs the network. |
| Gateway won't start / model fails to load | "Let me show you the recording from this morning's run." | `make demo-offline` (tier 3: recorded scores, no models). |
| Clone quality is poor | "Even a bad clone should be caught, but let's use a good one." | Backup `cloned_volunteer.wav`. |
| Detector says ABSTAIN | "It is telling you it can't judge this audio, too short or too noisy. That is not a pass, and the agent is told so." | Continue. Abstain is a feature. |
| Detector misses the clone | "No detector is perfect. That's why the intent layer and the call-back exist." | Go straight to the intent panel. This is Demo 2's message. |
| Mic picks up room noise | — | Use the pre-recorded genuine call. |

## Numbers you may quote

Only rows in `docs/benchmarks/REPORT.md` (cite the row id on the slide). In particular:

- **Headline chart:** `make headline-chart BEFORE=... AFTER=...`, the per-language EER before and
  after Indic fine-tuning.
- **Serving capacity:** the `lt-...` serving rows, e.g. `lt-cpu_i7-13620H_proxyL6_int8.8`. That is
  8 real-time calls per laptop CPU at p95 < 700 ms, measured with a proxy model of the same shape
  as the student.

Never quote a paper's number as ours.
