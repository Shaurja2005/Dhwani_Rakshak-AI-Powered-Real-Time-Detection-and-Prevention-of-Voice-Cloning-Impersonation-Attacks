"""B16 — egress guard, feature-only logging, keys + rotation, audit log, flagged-audio vault,
erasable evidence, model cards, privacy service API."""

from __future__ import annotations

import base64
import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import structlog
from fastapi.testclient import TestClient

from packages.vg_models.heads.head_d_speaker.vault import Voiceprint, VoiceprintVault
from services.privacy.audit import AuditLog, hash_identifier
from services.privacy.egress import EgressDenied, EgressGuard, audio_findings
from services.privacy.feature_log import REDACTED, redact_audio
from services.privacy.keys import TenantKeyError, TenantKeyManager
from services.privacy.policy import ConsentMatrix, TenantPrivacy
from services.privacy.raw_retention import FlaggedAudioVault

KEK = b"K" * 32


# ---------------------------------------------------------------- T03/T05 egress + residency
def test_egress_guard_default_deny_residency_and_audio() -> None:
    g = EgressGuard()
    t = TenantPrivacy("bank-a", egress_allowlist=["hooks.bank-a.in", "*.bank-a.co.in"])
    g.check("https://hooks.bank-a.in/vg", {"type": "policy_decision", "risk": 81}, t)
    g.check("https://ops.bank-a.co.in/x", {}, t)
    with pytest.raises(EgressDenied, match="allowlist"):
        g.check("https://api.some-cloud.com/detect", {}, t)
    with pytest.raises(EgressDenied, match="https"):
        g.check("http://hooks.bank-a.in/vg", {}, t)
    with pytest.raises(EgressDenied, match="residency"):
        g.check("https://hooks.bank-a.in/vg", {}, t, offshore=True)
    opted = TenantPrivacy("bank-b", central_processing=True, egress_allowlist=["x.in"])
    g.check("https://x.in/", {}, opted, offshore=True)
    for bad in (
        {"audio_base64": "AAAA"},
        {"data": {"pcm": [0.1] * 10}},
        {"x": b"\x00" * 10},
        {"emb": list(np.linspace(0, 1, 256))},
        {"blob": base64.b64encode(b"\x01" * 5000).decode()},
        {"arr": np.zeros(8)},
    ):
        assert audio_findings(bad), bad
        with pytest.raises(EgressDenied, match="audio"):
            g.check("https://hooks.bank-a.in/vg", bad, t)
    assert audio_findings({"layer_weights": [0.1] * 5, "p_spoof": 0.3, "audio": None}) == []


def test_tenant_policy_defaults_and_override(tmp_path: Path) -> None:
    d = TenantPrivacy.load("unknown-tenant")
    assert d.region == "IN" and not d.central_processing and d.retention.raw_audio_hours == 0
    (tmp_path / "tenants").mkdir()
    (tmp_path / "tenants" / "bank-z.yaml").write_text(
        "retention: {raw_audio_hours: 24}\nlegal_holds: [s1]\n", encoding="utf-8"
    )
    z = TenantPrivacy.load("bank-z", tmp_path)
    assert (
        z.retention.raw_audio_hours == 24
        and z.retention.evidence_days == 365
        and z.legal_holds == ["s1"]
    )


# ---------------------------------------------------------------- T02 feature-only logging
def test_logging_processor_strips_audio() -> None:
    ev = redact_audio(
        None,
        "info",
        {
            "event": "scored",
            "p_spoof": 0.4,
            "codec": "g711u",
            "pcm": np.ones(10),
            "chunk": b"\x00\x01",
            "nested": {"embedding": [0.1] * 200, "ok": 1},
        },
    )
    assert ev["p_spoof"] == 0.4 and ev["codec"] == "g711u"
    assert ev["pcm"] == REDACTED and ev["chunk"] == REDACTED
    assert ev["nested"]["embedding"] == REDACTED and ev["nested"]["ok"] == 1
    from services.privacy import feature_log

    feature_log.install()
    feature_log.install()  # idempotent
    assert structlog.get_config()["processors"].count(redact_audio) == 1


# ---------------------------------------------------------------- T07 keys
def test_key_manager_rotation_rewraps_vaults_and_shreds_old_key(tmp_path: Path) -> None:
    km = TenantKeyManager(tmp_path / "keys.json", kek_provider=lambda: KEK)
    vault = VoiceprintVault(tmp_path / "vp.db", key_provider=km.provider)
    vault.put(
        Voiceprint(
            "bank-a", "cust-1", "ecapa", consent_ref="C-1", embeddings={"clean": [[0.5] * 4]}
        )
    )
    old = km.key("bank-a")
    assert "K" * 8 not in (tmp_path / "keys.json").read_text()  # DEKs never stored in clear
    assert base64.b64encode(old).decode() not in (tmp_path / "keys.json").read_text()
    v = km.rotate("bank-a", [vault])
    assert v == 2 and km.versions("bank-a") == [2] and km.key("bank-a") != old
    assert vault.get("bank-a", "cust-1").consent_ref == "C-1"  # readable with the new key
    with pytest.raises(TenantKeyError, match="retired"):
        km.key("bank-a", 1)
    assert TenantKeyManager(tmp_path / "keys.json", lambda: KEK).key("bank-a") == km.key("bank-a")
    with pytest.raises(TenantKeyError):
        TenantKeyManager(tmp_path / "keys.json", lambda: b"X" * 32).key("bank-a")  # wrong KEK
    km.shred("bank-a")
    assert km.versions("bank-a") == []


# ---------------------------------------------------------------- T08 audit
def test_audit_log_is_append_only_and_validated(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.db")
    log.record(
        "dsar_export",
        "dpo",
        "bank-a",
        "data_subject_request",
        "legal_obligation",
        hash_identifier("cust-1"),
    )
    log.record("key_rotated", "cli", "bank-a", "key_management", details={"version": 2})
    assert log.verify() and len(log.events("bank-a")) == 2
    with pytest.raises(ValueError, match="invalid audit event"):
        log.record("export_everything", "x", "bank-a", "p")
    con = sqlite3.connect(tmp_path / "a.db")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM audit")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE audit SET body='{}'")
    assert hash_identifier("cust-1") == hash_identifier("cust-1") != hash_identifier("cust-2")


# ---------------------------------------------------------------- T02 flagged raw audio
def test_flagged_audio_vault_off_by_default_and_expires(tmp_path: Path) -> None:
    km = TenantKeyManager(tmp_path / "k.json", lambda: KEK)
    audit = AuditLog()
    fav = FlaggedAudioVault(km.provider, ConsentMatrix.load(), audit, tmp_path / "fa.db")
    pcm = np.sin(np.arange(16000) / 10).astype(np.float32)
    default = TenantPrivacy("bank-a")
    assert not fav.maybe_retain("s1", default, "legitimate_use", "HIGH", pcm)  # default: never
    on = TenantPrivacy("bank-a")
    on.retention.raw_audio_hours = 24
    assert not fav.maybe_retain("s2", on, "legitimate_use", "LOW", pcm)  # not flagged
    assert not fav.maybe_retain("s3", on, "none", "HIGH", pcm)  # no lawful basis
    assert fav.maybe_retain("s4", on, "legitimate_use", "HIGH", pcm)
    np.testing.assert_allclose(fav.get("s4", "bank-a"), pcm)
    assert pcm.astype("<f4").tobytes()[:64] not in (tmp_path / "fa.db").read_bytes()  # encrypted
    assert fav.get("s4", "bank-b") is None  # tenant-scoped
    km.rotate("bank-a", [fav])
    np.testing.assert_allclose(fav.get("s4", "bank-a"), pcm)
    assert fav.purge_expired(dt.datetime.now(dt.UTC) + dt.timedelta(hours=25)) == 1
    assert fav.get("s4", "bank-a") is None
    acts = [e["action"] for e in audit.events("bank-a")]
    assert acts == ["consent_refused", "raw_audio_retained", "raw_audio_purged"]


# ---------------------------------------------------------------- T06 erasable evidence
def test_evidence_erasure_is_provable_and_body_tampering_detected(tmp_path: Path) -> None:
    from packages.vg_core.models import CallMetadata
    from packages.vg_core.stub_head import StubHead
    from sdks.python.voiceguard import call_metadata
    from services.api_gateway.pipeline import Services, SessionPipeline
    from services.policy.evidence import EvidenceStore

    src = EvidenceStore()
    svc = Services(
        evidence=src, heads_factory=lambda: [StubHead(h, abstain_fraction=0.0) for h in "AB"]
    )
    t = np.arange(4 * 16000) / 16000
    for _ in range(3):
        pipe = SessionPipeline(
            CallMetadata.model_validate(call_metadata("bank-a")), svc, with_context=False
        )
        pipe.push_pcm16k((0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32))
        pipe.close()
    bundles = src.iter_bodies()
    assert len(bundles) >= 3
    store = EvidenceStore(tmp_path / "ev.db")
    for b in bundles[:3]:
        store.append(b)
    assert store.erase(bundles[1]["bundle_id"], "dsar") and not store.erase("sha256:nope", "x")
    assert store.status(bundles[1]["bundle_id"]) == "erased" and store.status("nope") is None
    assert store.get(bundles[1]["bundle_id"]) is None and store.verify_chain()
    con = sqlite3.connect(tmp_path / "ev.db")
    con.execute("DELETE FROM evidence_body WHERE bundle_id=?", (bundles[2]["bundle_id"],))
    con.commit()  # body removed without an erasure record = tampering
    assert not EvidenceStore(tmp_path / "ev.db").verify_chain()


# ---------------------------------------------------------------- T09 model cards
def test_model_card_generation(tmp_path: Path) -> None:
    from packages.vg_eval import protocols as P
    from packages.vg_eval.fairness import GateDecision
    from packages.vg_eval.report import RunRecord
    from packages.vg_models.registry import ModelRegistry
    from services.privacy import model_card

    rng = np.random.default_rng(0)
    import pandas as pd

    table = lambda: pd.DataFrame(  # noqa: E731
        [
            {
                "utt_id": f"u{i}",
                "label": "spoof" if i % 2 else "bona_fide",
                "score": rng.normal(-1 if i % 2 else 1, 1),
                "attack_family": ("f_seen" if i % 4 == 1 else "f_new") if i % 2 else None,
            }
            for i in range(200)
        ]
    )
    rec = RunRecord(
        "A@card-v1", "head_a", {}, {"eval_sets": ["x"], "train_datasets": ["asvspoof5"]}
    )
    rec.add_rows([P.summarize(table(), "overall", "all", 0)] + P.logo(table(), ["f_seen"], 0))
    rec.add_fairness([], GateDecision("not_configured", "Q9"))
    rec.save(tmp_path / "runs")
    art = tmp_path / "a.pt"
    art.write_bytes(b"weights")
    reg = ModelRegistry(tmp_path / "released.yaml")
    reg.register(
        "A@card-v1",
        "head_a",
        art,
        "research",
        eval_run_id=rec.run_id,
        fairness_gate="not_configured",
    )
    assert (
        model_card.main(
            [
                "--version",
                "A@card-v1",
                "--registry",
                str(tmp_path / "released.yaml"),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    card = (tmp_path / "A_card-v1.md").read_text(encoding="utf-8")
    assert f"`{rec.run_id}.logo.ALL_UNSEEN`" in card and "asvspoof5" in card
    assert "non-commercial" in card and "commercial deployment allowed: no" in card


# ---------------------------------------------------------------- privacy service API
def test_privacy_service_api(tmp_path: Path) -> None:
    from services.policy.evidence import EvidenceStore
    from services.privacy.main import create_app
    from services.privacy.retention import Stores

    audit = AuditLog()
    app = create_app(
        Stores(evidence=EvidenceStore()), audit, TenantKeyManager(tmp_path / "k.json", lambda: KEK)
    )
    c = TestClient(app)
    h = c.get("/healthz").json()
    assert h["audit_chain_ok"] and h["evidence_chain_ok"]
    assert c.get("/v1/consent-matrix").json()["speaker_enrollment"]["bundling_allowed"] is False
    assert c.get("/v1/tenants/bank-a/privacy").json()["region"] == "IN"
    assert c.post("/v1/tenants/bank-a/dsar/export", json={"actor": "dpo"}).status_code == 422
    r = c.post("/v1/tenants/bank-a/dsar/export", json={"actor": "dpo", "caller_number": "+91980"})
    assert r.status_code == 200 and r.json()["evidence_bundles"] == []
    assert c.post("/v1/retention/run", json=["bank-a"]).status_code == 200
    assert [e["action"] for e in c.get("/v1/tenants/bank-a/audit").json()] == [
        "dsar_export",
        "retention_run",
    ]


def _v1_db(path: Path, bundles: list[dict], tamper: bool = False) -> None:
    """Write evidence the way the B11 v1 store did (body inside the chain row)."""
    import hashlib

    from services.policy.evidence import canonical

    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE evidence (seq INTEGER PRIMARY KEY AUTOINCREMENT, bundle_id TEXT UNIQUE, "
        "session_id TEXT, created_at TEXT, body BLOB, prev_hash TEXT, row_hash TEXT)"
    )
    prev = "genesis"
    for b in bundles:
        body = canonical(b)
        row = hashlib.sha256(prev.encode() + body).hexdigest()
        con.execute(
            "INSERT INTO evidence (bundle_id, session_id, created_at, body, prev_hash, row_hash) VALUES (?,?,?,?,?,?)",
            (b["bundle_id"], b["session_id"], b["created_at"], body, prev, row),
        )
        prev = row
    if tamper:
        con.execute("UPDATE evidence SET prev_hash='x' WHERE seq=2")
    con.commit()
    con.close()


def test_evidence_store_migrates_v1_databases(tmp_path: Path) -> None:
    from packages.vg_core.models import CallMetadata
    from packages.vg_core.stub_head import StubHead
    from sdks.python.voiceguard import call_metadata
    from services.api_gateway.pipeline import Services, SessionPipeline
    from services.policy.evidence import EvidenceIntegrityError, EvidenceStore

    src = EvidenceStore()
    svc = Services(evidence=src, heads_factory=lambda: [StubHead("A", abstain_fraction=0.0)])
    t = np.arange(4 * 16000) / 16000
    for _ in range(3):
        pipe = SessionPipeline(
            CallMetadata.model_validate(call_metadata("bank-a")), svc, with_context=False
        )
        pipe.push_pcm16k((0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32))
        pipe.close()
    bundles = src.iter_bodies()
    _v1_db(tmp_path / "v1.db", bundles)
    store = EvidenceStore(tmp_path / "v1.db")  # migrates on open
    assert store.verify_chain() and len(store.iter_bodies("bank-a")) == len(bundles)
    assert store.get(bundles[0]["bundle_id"]) == bundles[0]
    con = sqlite3.connect(tmp_path / "v1.db")
    assert con.execute("SELECT rows, from_schema FROM evidence_migration").fetchone() == (
        len(bundles),
        "v1",
    )
    con.close()
    assert EvidenceStore(tmp_path / "v1.db").verify_chain()  # idempotent: no second migration
    _v1_db(tmp_path / "bad.db", bundles, tamper=True)
    with pytest.raises(EvidenceIntegrityError, match="refusing to migrate"):
        EvidenceStore(tmp_path / "bad.db")
