"""Enrollment service core (B7-T02, B7-T05).

Turns enrolment audio into voiceprint embeddings and stores them in the vault.
Raw audio exists only in memory for the duration of the request (I5).

Policy (per tenant, configurable):
* quality gate — minimum voiced seconds, minimum SNR, maximum clipping
* multi-session — up to ``max_sessions`` sessions are averaged into the voiceprint
* consistency — a new session must match the existing voiceprint
  (cosine >= ``consistency_min_cosine``) or it is rejected as a possibly
  different speaker; ``replace=True`` starts over (re-enrolment)
* expiry — voiceprints expire after ``expiry_days`` and must be re-enrolled
* explicit consent reference is mandatory

Channel compensation (B7-T05): each session is also embedded after passing
through telephony channels (G.711 narrowband, AMR-NB when available), so a
headset enrolment can be compared against an 8 kHz call on matched terms.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from ml.data.channel.base import ChannelRecord, resample
from ml.data.channel.codecs import G711, FFmpegCodec, codec_available
from packages.vg_audio.quality import compute_clipping_ratio, estimate_snr_db
from packages.vg_models.heads.head_d_speaker.embedders import Embedder
from packages.vg_models.heads.head_d_speaker.scoring import cosine
from packages.vg_models.heads.head_d_speaker.vault import (
    EnrollmentSession,
    Voiceprint,
    VoiceprintVault,
)

SR = 16000


@dataclass
class EnrollmentPolicy:
    min_voiced_seconds: float = 8.0
    min_snr_db: float = 15.0
    max_clipping_ratio: float = 0.01
    max_sessions: int = 5
    consistency_min_cosine: float = 0.5
    expiry_days: int = 365
    channel_compensation: bool = True


@dataclass
class EnrollmentResult:
    accepted: bool
    speaker_id: str
    reasons: list[str] = field(default_factory=list)
    voiced_seconds: float = 0.0
    snr_db: float = 0.0
    sessions: int = 0
    conditions: list[str] = field(default_factory=list)
    consistency_cosine: float | None = None


def voiced_seconds(pcm: np.ndarray, frame: int = 320) -> float:
    n = len(pcm) // frame
    if n == 0:
        return 0.0
    e = 10 * np.log10((pcm[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    active = e > max(e.max() - 35.0, -60.0)
    return float(active.sum() * frame / SR)


def channel_variants(pcm: np.ndarray, enabled: bool) -> dict[str, np.ndarray]:
    """Enrolment audio under each channel condition the head may score against."""
    out = {"wideband": pcm}
    if not enabled:
        return out
    rng = np.random.default_rng(0)
    y, sr = G711("u").apply(pcm, SR, rng, ChannelRecord())
    narrow = [resample(y, sr, SR)]
    if codec_available("amr_nb"):
        try:
            z, zsr = FFmpegCodec("amr_nb").apply(pcm, SR, rng, ChannelRecord())
            narrow.append(resample(z, zsr, SR))
        except Exception:  # noqa: BLE001, S110 - AMR is optional compensation
            pass
    out["narrowband"] = narrow  # type: ignore[assignment]
    return out


class EnrollmentService:
    def __init__(
        self, vault: VoiceprintVault, embedder: Embedder, policy: EnrollmentPolicy | None = None
    ) -> None:
        self.vault = vault
        self.embedder = embedder
        self.policy = policy or EnrollmentPolicy()

    def enroll(
        self,
        tenant_id: str,
        speaker_id: str,
        pcm: np.ndarray,
        sample_rate: int,
        consent_ref: str,
        session_ref: str,
        replace: bool = False,
    ) -> EnrollmentResult:
        p = self.policy
        res = EnrollmentResult(accepted=False, speaker_id=speaker_id)
        if not consent_ref:
            res.reasons.append("explicit consent reference is required for voice enrolment")
            return res

        x = np.nan_to_num(np.asarray(pcm, dtype=np.float32))
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sample_rate != SR:
            x = resample(x, sample_rate, SR)
        res.voiced_seconds = voiced_seconds(x)
        res.snr_db = float(estimate_snr_db(x))
        clipping = compute_clipping_ratio(x)
        if res.voiced_seconds < p.min_voiced_seconds:
            res.reasons.append(
                f"only {res.voiced_seconds:.1f} s of speech; need {p.min_voiced_seconds:.0f} s"
            )
        if res.snr_db < p.min_snr_db:
            res.reasons.append(f"too noisy (SNR {res.snr_db:.0f} dB < {p.min_snr_db:.0f} dB)")
        if clipping > p.max_clipping_ratio:
            res.reasons.append(f"audio is clipped ({clipping:.1%} of samples)")
        if res.reasons:
            return res

        existing = None if replace else self.vault.get(tenant_id, speaker_id)
        if existing is not None and existing.expired():
            existing = None  # expired voiceprints are re-enrolled from scratch
        if existing is not None and existing.embedder != self.embedder.name:
            res.reasons.append(
                f"existing voiceprint uses {existing.embedder}; re-enrol with replace=true"
            )
            return res
        if existing is not None and len(existing.sessions) >= p.max_sessions:
            res.reasons.append(
                f"maximum of {p.max_sessions} sessions reached; re-enrol with replace=true"
            )
            return res

        variants = channel_variants(x, p.channel_compensation)
        new_embs: dict[str, list[list[float]]] = {}
        for cond, audio in variants.items():
            clips = audio if isinstance(audio, list) else [audio]
            new_embs[cond] = [self.embedder.embed(c).tolist() for c in clips]

        if existing is not None:
            centroid = existing.centroid("wideband")
            res.consistency_cosine = (
                cosine(centroid, np.array(new_embs["wideband"][0]))
                if centroid is not None
                else None
            )
            if (
                res.consistency_cosine is not None
                and res.consistency_cosine < p.consistency_min_cosine
            ):
                res.reasons.append(
                    "this recording does not match the existing voiceprint "
                    f"(cosine {res.consistency_cosine:.2f}); possible different speaker — "
                    "re-enrol with replace=true after verifying identity"
                )
                return res

        now = dt.datetime.now(dt.UTC)
        vp = existing or Voiceprint(
            tenant_id=tenant_id,
            speaker_id=speaker_id,
            embedder=self.embedder.name,
            consent_ref=consent_ref,
        )
        vp.consent_ref = consent_ref
        for cond, embs in new_embs.items():
            vp.embeddings.setdefault(cond, []).extend(embs)
        vp.sessions.append(
            EnrollmentSession(
                session_ref=session_ref,
                recorded_at=now.isoformat(),
                voiced_seconds=round(res.voiced_seconds, 2),
                snr_db=round(res.snr_db, 1),
                channel="wideband" if sample_rate > 8000 else "narrowband",
            )
        )
        vp.expires_at = (now + dt.timedelta(days=p.expiry_days)).isoformat()
        self.vault.put(vp)
        res.accepted = True
        res.sessions = len(vp.sessions)
        res.conditions = sorted(vp.embeddings)
        return res
