"""B16 compliance suite: retention jobs, consent per purpose, DSAR, DPIA present.

Definition of Done (IMPLEMENTATION_PLAN B16): no raw audio at rest on the default
path (test_no_raw_audio_at_rest.py), retention jobs verified, consent basis
enforced per processing purpose, and the DPIA is written.
"""

from __future__ import annotations

import base64
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.vg_core.models import ConsentBasis
from packages.vg_core.stub_head import StubHead
from packages.vg_models.heads.head_d_speaker.vault import Voiceprint, VoiceprintVault
from sdks.python.voiceguard import call_metadata
from services.api_gateway.app import create_app
from services.api_gateway.auth import KeyStore
from services.api_gateway.pipeline import Services, SessionPipeline
from services.api_gateway.webhooks import WebhookRegistry
from services.fusion.persistence import SQLiteTimelineStore
from services.policy.evidence import EvidenceStore
from services.policy.feedback import FeedbackStore
from services.privacy.audit import AuditLog
from services.privacy.policy import ConsentError, ConsentMatrix
from services.privacy.retention import DSAR, RetentionJob, Stores, Subject

ROOT = Path(__file__).resolve().parents[2]


def voiced(seconds: float = 5.0) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    return (0.3 * np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )


def run_calls(svc: Services, n: int, **meta_kw: object) -> list[str]:
    from packages.vg_core.models import CallMetadata

    out = []
    for _ in range(n):
        meta = CallMetadata.model_validate(call_metadata("bank-a", **meta_kw))
        p = SessionPipeline(meta, svc, with_context=False)
        p.push_pcm16k(voiced())
        p.close()
        out.append(meta.session_id)
    return out


@pytest.fixture()
def world(tmp_path: Path) -> dict[str, object]:
    os.environ["VG_VAULT_KEY_BANK_A"] = base64.b64encode(b"k" * 32).decode()
    ev = EvidenceStore(tmp_path / "ev.db")
    tl = SQLiteTimelineStore(tmp_path / "tl.db")
    svc = Services(
        evidence=ev,
        timeline=tl,
        heads_factory=lambda: [StubHead(h, abstain_fraction=0.0) for h in "ABC"],
    )
    vault = VoiceprintVault(tmp_path / "vp.db")
    stores = Stores(
        evidence=ev, timeline=tl, feedback=FeedbackStore(ev, tmp_path / "fb.db"), voiceprints=vault
    )
    return {"svc": svc, "stores": stores, "audit": AuditLog(tmp_path / "audit.db"), "tmp": tmp_path}


@pytest.mark.compliance
def test_retention_job_erases_expired_evidence_and_keeps_chain(world: dict) -> None:
    svc, stores, audit = world["svc"], world["stores"], world["audit"]
    sessions = run_calls(svc, 3, claimed_identity_id="cust-1")
    bids = [b["bundle_id"] for s in sessions for b in stores.evidence.for_session(s)]
    assert bids and stores.evidence.verify_chain()
    job = RetentionJob(stores, audit)
    # nothing is due today
    assert job.run(["bank-a"])[0].evidence_erased == 0
    later = dt.datetime.now(dt.UTC) + dt.timedelta(days=400)  # default evidence_days = 365
    dry = job.run(["bank-a"], now=later, dry_run=True)[0]
    assert dry.evidence_erased == len(bids) and stores.evidence.get(bids[0]) is not None
    rep = job.run(["bank-a"], now=later)[0]
    assert rep.evidence_erased == len(bids)
    assert all(
        stores.evidence.get(b) is None and stores.evidence.status(b) == "erased" for b in bids
    )
    assert stores.evidence.verify_chain()  # erasure is provable, history intact
    assert all(stores.timeline.session(s) == [] for s in sessions)
    assert [e["outcome"] for e in audit.events("bank-a", "retention_run")] == [
        "ok",
        "dry_run",
        "ok",
    ]
    assert audit.verify()


@pytest.mark.compliance
def test_legal_hold_blocks_retention(world: dict, tmp_path: Path) -> None:
    svc, stores, audit = world["svc"], world["stores"], world["audit"]
    held = run_calls(svc, 1)[0]
    cfg = tmp_path / "privacy"
    (cfg / "tenants").mkdir(parents=True)
    (cfg / "tenants" / "bank-a.yaml").write_text(f"legal_holds: [{held}]\n", encoding="utf-8")
    later = dt.datetime.now(dt.UTC) + dt.timedelta(days=400)
    rep = RetentionJob(stores, audit, config_dir=cfg).run(["bank-a"], now=later)[0]
    assert rep.held_sessions_skipped >= 1 and stores.evidence.for_session(held)


@pytest.mark.compliance
def test_consent_enforced_per_purpose(world: dict) -> None:
    m = ConsentMatrix.load()
    assert m.check("fraud_detection", "legitimate_use")[0]
    assert not m.check("fraud_detection", "none")[0]
    assert not m.check("speaker_enrollment", "legitimate_use")[
        0
    ]  # biometrics need explicit consent
    assert not m.check("speaker_enrollment", "explicit_consent")[0]  # ... and a verifiable record
    assert m.check("speaker_enrollment", "explicit_consent", consent_ref="C-1")[0]
    assert not m.check(
        "speaker_enrollment",
        "explicit_consent",
        consent_ref="C-1",
        bundled_with=["marketing_analytics"],
    )[0]
    assert not m.check("marketing_analytics", "explicit_consent")[0]  # disabled in this product
    assert not m.check("sell_voice_data", "explicit_consent")[0]  # purpose limitation
    with pytest.raises(ConsentError):
        run_calls(world["svc"], 1, consent_basis=ConsentBasis.NONE.value)
    keys = KeyStore()
    raw, _ = keys.create("bank-a", {"analyze", "stream"})
    c = TestClient(create_app(world["svc"], keys, WebhookRegistry()))
    meta = call_metadata("bank-a", consent_basis="none")
    r = c.post(
        "/v1/sessions", json={"call_metadata": meta}, headers={"Authorization": f"Bearer {raw}"}
    )
    assert r.status_code == 403 and "consent" in r.json()["detail"]
    with pytest.raises(Exception, match="consent"):
        VoiceprintVault(world["tmp"] / "x.db").put(
            Voiceprint("bank-a", "s1", "ecapa", consent_ref="")
        )


@pytest.mark.compliance
def test_dsar_export_and_erasure(world: dict) -> None:
    svc, stores, audit = world["svc"], world["stores"], world["audit"]
    mine = run_calls(svc, 2, claimed_identity_id="cust-42", caller_number="+919800000001")
    other = run_calls(svc, 1, claimed_identity_id="cust-7")
    stores.voiceprints.put(
        Voiceprint(
            "bank-a", "cust-42", "ecapa", consent_ref="C-42", embeddings={"clean": [[0.1] * 8]}
        )
    )
    bid = stores.evidence.for_session(mine[0])[0]["bundle_id"]
    stores.feedback.record(bid, "false_positive", "analyst-1")
    dsar = DSAR(stores, audit)
    subj = Subject(claimed_identity_id="cust-42")
    exp = dsar.export("bank-a", subj, "dpo")
    assert {b["session_id"] for b in exp["evidence_bundles"]} == set(mine)
    assert exp["voiceprint"]["consent_ref"] == "C-42" and "embeddings" not in exp["voiceprint"]
    assert len(exp["analyst_feedback"]) == 1 and set(exp["score_timeline"]) == set(mine)
    res = dsar.erase("bank-a", subj, "dpo")
    assert (
        res["evidence_erased"] >= 2 and res["voiceprint_deleted"] and res["feedback_deleted"] == 1
    )
    assert not dsar.export("bank-a", subj, "dpo")["evidence_bundles"]
    assert stores.evidence.for_session(other[0])  # other customers untouched
    assert stores.evidence.verify_chain() and audit.verify()
    ev = audit.events("bank-a", "dsar_erasure")[0]
    assert ev["subject_ref"].startswith("h:") and "cust-42" not in str(ev)  # audit holds no raw id


@pytest.mark.compliance
def test_dpia_and_governance_docs_exist() -> None:
    dpia = (ROOT / "docs/compliance/DPIA.md").read_text(encoding="utf-8")
    for section in (
        "Processing description",
        "Lawful basis",
        "Necessity and proportionality",
        "Risks to individuals",
        "Mitigations",
        "Residual risk",
        "Sign-off",
    ):
        assert section in dpia, section
    assert "TODO(B16-T08)" not in dpia
    assert (ROOT / "docs/compliance/DATA_RESIDENCY.md").exists()
    assert (ROOT / "docs/model_cards/TEMPLATE.md").exists()
