"""B7 — Head D speaker verification, voiceprint vault and enrollment service tests."""

from __future__ import annotations

import base64
import datetime as dt
import io
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from scipy.signal import lfilter

from packages.vg_core.models import AbstainReason, AnalysisWindow, SessionContext
from packages.vg_core.sample_store import put_samples
from packages.vg_models.heads.head_d_speaker.embedders import MFCCStatsEmbedder, l2norm
from packages.vg_models.heads.head_d_speaker.head import HeadD, channel_condition
from packages.vg_models.heads.head_d_speaker.scoring import (
    as_norm,
    cosine,
    eer_with_ci,
    fit_calibration,
    gap_with_ci,
)
from packages.vg_models.heads.head_d_speaker.vault import VaultError, Voiceprint, VoiceprintVault
from services.enrollment.api import create_app
from services.enrollment.core import EnrollmentPolicy, EnrollmentService, channel_variants

SR = 16000
TENANT = "bank-a"
KEYS = {"bank-a": os.urandom(32), "bank-b": os.urandom(32)}


def keys(tenant: str) -> bytes:
    return KEYS[tenant]


# ---------------------------------------------------------------- synthetic voices
_VOWELS = [
    (730, 1090, 2440),
    (270, 2290, 3010),
    (300, 870, 2240),
    (530, 1840, 2480),
    (570, 840, 2410),
]


def voice(
    f0: float,
    formant_scale: float,
    rng: np.random.Generator,
    seconds: float = 3.0,
    tilt: float = 0.97,
) -> np.ndarray:
    n = int(seconds * SR)
    out = np.zeros(n)
    pos = 0
    while pos < n:
        seg = int(rng.uniform(0.15, 0.3) * SR)
        vowel = _VOWELS[rng.integers(len(_VOWELS))]
        f = f0 * (1 + 0.05 * rng.standard_normal())
        src = np.zeros(seg)
        src[np.arange(0, seg, SR / f).astype(int)] = 1
        y = lfilter([1], [1, -tilt], src)
        for formant, bw in zip([v * formant_scale for v in vowel], (80, 100, 120), strict=True):
            r, th = np.exp(-np.pi * bw / SR), 2 * np.pi * formant / SR
            y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
        out[pos : pos + seg] = (y * np.hanning(seg))[: n - pos]
        pos += seg + int(rng.uniform(0.15, 0.35) * SR)  # pauses, like real speech
    out = 0.5 * out / np.abs(out).max()
    return (out + 0.0005 * rng.standard_normal(n)).astype(np.float32)


class FakeEmbedder:
    """Deterministic test double: speaker identity is encoded in the first sample."""

    name = "fake"
    dim = 16
    is_baseline = False

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        spk = int(round(float(pcm[0]) * 1000))
        base = np.random.default_rng(1000 + spk).standard_normal(self.dim)
        jitter = np.random.default_rng(
            int(abs(float(pcm[1:100].sum())) * 1e6) % 2**32
        ).standard_normal(self.dim)
        return l2norm(base + 0.1 * jitter)


def tagged(
    speaker: int, rng: np.random.Generator, seconds: float = 10.0, snr_noise: float = 0.001
) -> np.ndarray:
    x = voice(120 + 10 * speaker, 1.0, rng, seconds)
    x = x + snr_noise * rng.standard_normal(len(x)).astype(np.float32)
    x[0] = speaker / 1000
    return x


def vault() -> VoiceprintVault:
    return VoiceprintVault(":memory:", key_provider=keys)


# ---------------------------------------------------------------- vault (T03)
def test_vault_roundtrip_encrypts_and_binds_tenant_speaker(tmp_path: Path) -> None:
    db = tmp_path / "v.db"
    v = VoiceprintVault(db, key_provider=keys)
    vp = Voiceprint(
        TENANT, "ceo", "fake", "consent-001", embeddings={"wideband": [[0.123456, 0.654321]]}
    )
    v.put(vp)
    got = v.get(TENANT, "ceo")
    assert got is not None and got.embeddings == vp.embeddings and got.consent_ref == "consent-001"
    raw = db.read_bytes()
    assert b"0.123456" not in raw and b"consent-001" not in raw  # only ciphertext at rest
    con = sqlite3.connect(db)  # copy ciphertext onto another speaker id: AEAD must reject it
    con.execute(
        "INSERT INTO voiceprints "
        "SELECT tenant_id, 'cfo', nonce, ciphertext, key_version FROM voiceprints"
    )
    con.commit()
    with pytest.raises(VaultError):
        v.get(TENANT, "cfo")


def test_vault_requires_consent_and_key() -> None:
    v = vault()
    with pytest.raises(VaultError, match="consent"):
        v.put(Voiceprint(TENANT, "x", "fake", ""))
    with pytest.raises(VaultError, match="no vault key"):
        VoiceprintVault(":memory:").put(Voiceprint("no-such-tenant", "x", "fake", "c"))


def test_vault_erasure_listing_rotation_and_cohort() -> None:
    v = vault()
    for s in ("a", "b"):
        v.put(Voiceprint(TENANT, s, "fake", "c", embeddings={"wideband": [[1.0, 0.0]]}))
    assert v.list_speakers(TENANT) == ["a", "b"]
    old, new = KEYS[TENANT], os.urandom(32)
    assert v.rotate_tenant_key(TENANT, old, new) == 2
    KEYS[TENANT] = new
    try:
        assert v.get(TENANT, "a") is not None
    finally:
        KEYS[TENANT] = old
    assert v.delete_speaker(TENANT, "a") and not v.delete_speaker(TENANT, "a")
    v.set_cohort("fake", np.eye(3, dtype=np.float32))
    assert np.array_equal(v.get_cohort("fake"), np.eye(3))


# ---------------------------------------------------------------- scoring (T04)
def test_as_norm_falls_back_to_cosine_and_separates_with_cohort() -> None:
    rng = np.random.default_rng(0)
    target = l2norm(rng.standard_normal(32))
    same = l2norm(target + 0.3 * rng.standard_normal(32))
    other = l2norm(rng.standard_normal(32))
    assert as_norm(target, same, None) == (cosine(target, same), cosine(target, same))
    cohort = rng.standard_normal((100, 32))
    assert as_norm(target, same, cohort)[1] > as_norm(target, other, cohort)[1] + 2


def test_calibration_eer_and_gap_with_confidence_intervals() -> None:
    rng = np.random.default_rng(1)
    tgt, non = rng.normal(3, 1, 500), rng.normal(0, 1, 500)
    cal = fit_calibration(tgt, non)
    assert cal.p_mismatch(3.0) < 0.2 < 0.8 < cal.p_mismatch(0.0)
    e = eer_with_ci(tgt, non)
    assert e["ci95_low"] <= e["eer"] <= e["ci95_high"] and 0.03 < e["eer"] < 0.1
    g = gap_with_ci(tgt, non)
    assert g["ci95_low"] > 2.5 and g["ci95_low"] <= g["gap"] <= g["ci95_high"]


# ---------------------------------------- baseline embedder + channel compensation (T01/T05)
def test_baseline_embedder_is_deterministic_normalised_and_marked_baseline() -> None:
    rng = np.random.default_rng(2)
    emb = MFCCStatsEmbedder()
    x = voice(120, 1.0, rng)
    a, b = emb.embed(x), emb.embed(x.copy())
    assert emb.is_baseline and a.shape == (emb.dim,) and np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-4
    # Accuracy is deliberately NOT asserted: this classical baseline is content-dependent and
    # unreliable for verification. Real embedders are chosen with benchmark.py (B7-T01).


class SpyEmbedder(FakeEmbedder):
    """Records the bandwidth of every clip it embeds."""

    def __init__(self) -> None:
        self.bandwidths: list[float] = []

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(pcm[1:])) ** 2
        freqs = np.fft.rfftfreq(len(pcm) - 1, 1 / SR)
        cum = np.cumsum(spec) / spec.sum()
        self.bandwidths.append(float(freqs[np.searchsorted(cum, 0.999)]))
        return super().embed(pcm)


def test_channel_compensation_embeds_band_limited_enrolment_audio() -> None:
    spy = SpyEmbedder()
    svc = EnrollmentService(vault(), spy, EnrollmentPolicy(min_voiced_seconds=3.0))
    rng = np.random.default_rng(3)
    x = tagged(1, rng)
    speech = np.convolve(np.abs(x), np.ones(400) / 400, mode="same") > 0.01
    tone = (
        0.05 * np.sin(2 * np.pi * 6000 * np.arange(len(x)) / SR) * speech
    )  # wideband energy, speech only
    x[1:] += tone[1:].astype(np.float32)
    res = svc.enroll(TENANT, "ceo", x, SR, "c", "s1")
    assert res.accepted and "narrowband" in res.conditions
    wide_bw, *narrow_bw = spy.bandwidths
    assert wide_bw > 5500 and narrow_bw and all(bw < 4200 for bw in narrow_bw)
    assert channel_condition(8000) == "narrowband" and channel_condition(16000) == "wideband"
    variants = channel_variants(x, enabled=False)
    assert list(variants) == ["wideband"]


# ---------------------------------------------------------------- enrollment service (T02)
def service(policy: EnrollmentPolicy | None = None) -> EnrollmentService:
    return EnrollmentService(
        vault(), FakeEmbedder(), policy or EnrollmentPolicy(min_voiced_seconds=3.0)
    )


def test_enrolment_quality_gates_and_consent() -> None:
    svc, rng = service(), np.random.default_rng(4)
    assert "consent" in svc.enroll(TENANT, "ceo", tagged(1, rng), SR, "", "s1").reasons[0]
    short = svc.enroll(TENANT, "ceo", tagged(1, rng, seconds=2.0), SR, "c", "s1")
    assert not short.accepted and "speech" in short.reasons[0]
    noisy = svc.enroll(TENANT, "ceo", tagged(1, rng, snr_noise=0.2), SR, "c", "s1")
    assert not noisy.accepted and any("noisy" in r for r in noisy.reasons)
    clipped = np.clip(tagged(1, rng) * 20, -1, 1)
    clipped[0] = 0.001
    assert any("clipped" in r for r in svc.enroll(TENANT, "ceo", clipped, SR, "c", "s1").reasons)
    assert svc.vault.get(TENANT, "ceo") is None


def test_multi_session_consistency_replace_and_max_sessions() -> None:
    svc = service(EnrollmentPolicy(min_voiced_seconds=3.0, max_sessions=2))
    rng = np.random.default_rng(5)
    first = svc.enroll(TENANT, "ceo", tagged(1, rng), SR, "c", "s1")
    assert first.accepted and first.sessions == 1 and first.conditions == ["narrowband", "wideband"]
    assert svc.enroll(TENANT, "ceo", tagged(1, rng), SR, "c", "s2").sessions == 2
    assert "maximum" in svc.enroll(TENANT, "ceo", tagged(1, rng), SR, "c", "s3").reasons[0]
    svc.policy.max_sessions = 5
    impostor = svc.enroll(TENANT, "ceo", tagged(7, rng), SR, "c", "s4")
    assert not impostor.accepted and "does not match" in impostor.reasons[0]
    replaced = svc.enroll(TENANT, "ceo", tagged(7, rng), SR, "c2", "s5", replace=True)
    assert replaced.accepted and replaced.sessions == 1


def test_enrollment_api_end_to_end() -> None:
    svc = service()
    client = TestClient(create_app(svc))
    rng = np.random.default_rng(6)
    buf = io.BytesIO()
    sf.write(buf, tagged(1, rng), SR, format="WAV")
    body = {
        "audio_base64": base64.b64encode(buf.getvalue()).decode(),
        "consent_ref": "c-1",
        "session_ref": "s-1",
    }
    url = f"/v1/tenants/{TENANT}/speakers/ceo"
    r = client.post(f"{url}/enrollments", json=body)
    assert r.status_code == 200 and r.json()["accepted"]
    pcm = (tagged(1, rng) * 32767).astype("<i2").tobytes()
    r = client.post(
        f"{url}/enrollments",
        json={
            **body,
            "audio_base64": base64.b64encode(pcm).decode(),
            "encoding": "pcm_s16le",
            "sample_rate": SR,
            "session_ref": "s-2",
        },
    )
    assert r.status_code == 200 and r.json()["sessions"] == 2
    assert client.post(f"{url}/enrollments", json={**body, "audio_base64": "!!"}).status_code == 400
    short = io.BytesIO()
    sf.write(short, tagged(1, rng, seconds=1.0), SR, format="WAV")
    assert (
        client.post(
            f"{url}/enrollments",
            json={**body, "audio_base64": base64.b64encode(short.getvalue()).decode()},
        ).status_code
        == 422
    )
    status = client.get(url).json()
    assert len(status["sessions"]) == 2 and "embeddings" not in status
    assert client.get(f"/v1/tenants/{TENANT}/speakers").json() == {"speakers": ["ceo"]}
    assert client.delete(url).status_code == 204 and client.get(url).status_code == 404


# ---------------------------------------------------------------- head (T04/T06)
SID = "01900b1a-0000-7000-8000-00000000d7d1"


def _window(pcm: np.ndarray, wid: int, sr: int = 16000) -> AnalysisWindow:
    ref = f"shm://{SID}/d{wid}"
    put_samples(ref, pcm)
    return AnalysisWindow(
        session_id=SID,
        window_id=wid,
        start_ms=0,
        end_ms=3000,
        samples_ref=ref,
        voiced_ms=2500,
        snr_db=20.0,
        clipping_ratio=0.0,
        quality_ok=True,
        original_sample_rate=sr,
    )


def _ctx(claim: str | None) -> SessionContext:
    return SessionContext(session_id=SID, tenant_id=TENANT, claimed_identity_id=claim)


@pytest.fixture()
def enrolled() -> EnrollmentService:
    svc = service()
    rng = np.random.default_rng(7)
    for i in range(3):
        assert svc.enroll(TENANT, "ceo", tagged(1, rng), SR, "c", f"s{i}").accepted
    return svc


def test_no_enrolment_paths_abstain_cleanly(enrolled: EnrollmentService) -> None:
    head = HeadD(enrolled.vault, FakeEmbedder())
    rng = np.random.default_rng(8)
    w = _window(tagged(1, rng, 3.0), 0)
    assert head.score(w, _ctx(None)).abstain_reason == AbstainReason.NO_ENROLLMENT
    assert head.score(w, _ctx("cfo")).abstain_reason == AbstainReason.NO_ENROLLMENT
    vp = enrolled.vault.get(TENANT, "ceo")
    assert vp is not None
    vp.expires_at = (dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).isoformat()
    enrolled.vault.put(vp)
    s = head.score(w, _ctx("ceo"))
    assert (
        s.abstain_reason == AbstainReason.NO_ENROLLMENT
        and s.evidence["note"] == "voiceprint expired"
    )


def test_genuine_scores_lower_mismatch_than_impostor(enrolled: EnrollmentService) -> None:
    enrolled.vault.set_cohort(
        "fake",
        np.stack(
            [
                FakeEmbedder().embed(tagged(100 + i, np.random.default_rng(i), 1.0))
                for i in range(40)
            ]
        ),
    )
    head = HeadD(enrolled.vault, FakeEmbedder())
    head.warmup()
    rng = np.random.default_rng(9)
    genuine = head.score(_window(tagged(1, rng, 3.0), 1), _ctx("ceo"))
    impostor = head.score(_window(tagged(9, rng, 3.0), 2), _ctx("ceo"))
    assert not genuine.abstain and not impostor.abstain
    assert (genuine.p_spoof or 1) < (impostor.p_spoof or 0)
    assert (
        genuine.evidence["cosine"] > impostor.evidence["cosine"]
        and genuine.evidence["as_norm"] is not None
    )
    narrow = head.score(_window(tagged(1, rng, 3.0), 3, sr=8000), _ctx("ceo"))
    assert narrow.evidence["enrolment_condition"] == "narrowband"


def test_baseline_embedder_is_refused_unless_allowed() -> None:
    svc = EnrollmentService(vault(), MFCCStatsEmbedder(), EnrollmentPolicy(min_voiced_seconds=3.0))
    rng = np.random.default_rng(10)
    assert svc.enroll(TENANT, "ceo", voice(120, 1.0, rng, 10.0), SR, "c", "s1").accepted
    w = _window(voice(120, 1.0, rng), 4)
    assert (
        HeadD(svc.vault, MFCCStatsEmbedder()).score(w, _ctx("ceo")).abstain_reason
        == AbstainReason.UNTRAINED
    )
    allowed = HeadD(svc.vault, MFCCStatsEmbedder(), allow_baseline=True).score(w, _ctx("ceo"))
    assert not allowed.abstain and allowed.evidence["baseline_embedder"] is True


def test_embedder_mismatch_and_errors_abstain(enrolled: EnrollmentService) -> None:
    rng = np.random.default_rng(11)
    w = _window(tagged(1, rng, 3.0), 5)
    s = HeadD(enrolled.vault, MFCCStatsEmbedder(), allow_baseline=True).score(w, _ctx("ceo"))
    assert s.abstain_reason == AbstainReason.NO_ENROLLMENT and "fake" in s.evidence["note"]

    class Boom(FakeEmbedder):
        def embed(self, pcm: np.ndarray) -> np.ndarray:
            raise RuntimeError("model crashed")

    assert HeadD(enrolled.vault, Boom()).score(w, _ctx("ceo")).abstain
