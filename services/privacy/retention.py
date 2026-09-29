"""Retention automation and data-subject requests (B16-T06).

``RetentionJob.run`` applies each tenant's retention policy across every store
that holds personal data about calls:

    evidence bodies (B11)       erase bodies older than ``evidence_days`` (chain kept)
    score timeline (B9)         delete windows older than ``timeline_days``
    analyst feedback (B11)      delete labels older than ``feedback_days``
    flagged raw audio (B16)     purge after ``raw_audio_hours``
    voiceprints (B7)            delete when expired or inactive > ``voiceprint_inactive_days``

Sessions on a tenant's ``legal_holds`` list are skipped. ``DSAR`` exports or
erases everything held about one data subject (claimed customer id, caller
number or enrolled speaker id). Both write to the audit log; both support
``dry_run``.

Timeline and feedback rows carry no tenant column, so their cut-off is applied
per session for sessions whose tenant is known (from evidence) and otherwise with
the most conservative (longest) window configured for any tenant.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from services.privacy.audit import AuditLog, hash_identifier
from services.privacy.policy import TenantPrivacy


@dataclass
class Stores:
    evidence: Any  # services.policy.evidence.EvidenceStore
    timeline: Any = None  # services.fusion.persistence.SQLiteTimelineStore
    feedback: Any = None  # services.policy.feedback.FeedbackStore
    voiceprints: Any = None  # packages.vg_models.heads.head_d_speaker.vault.VoiceprintVault
    flagged_audio: Any = None  # services.privacy.raw_retention.FlaggedAudioVault


@dataclass
class RetentionReport:
    tenant_id: str
    dry_run: bool
    evidence_erased: int = 0
    timeline_deleted: int = 0
    feedback_deleted: int = 0
    raw_audio_purged: int = 0
    voiceprints_deleted: int = 0
    held_sessions_skipped: int = 0
    erased_bundle_ids: list[str] = field(default_factory=list)


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.UTC).isoformat()


class RetentionJob:
    def __init__(
        self, stores: Stores, audit: AuditLog, config_dir: Path | str | None = None
    ) -> None:
        self.s = stores
        self.audit = audit
        self.config_dir = config_dir

    def run(
        self, tenant_ids: list[str], now: dt.datetime | None = None, dry_run: bool = False
    ) -> list[RetentionReport]:
        now = now or dt.datetime.now(dt.UTC)
        policies = {t: TenantPrivacy.load(t, self.config_dir) for t in tenant_ids}
        reports = []
        for tid, pol in policies.items():
            rep = RetentionReport(tid, dry_run)
            holds = set(pol.legal_holds)
            cutoff = _iso(now - dt.timedelta(days=pol.retention.evidence_days))
            expired = self.s.evidence.expired(cutoff, tid)
            erase_sessions = []
            for bid, sid in expired:
                if sid in holds:
                    rep.held_sessions_skipped += 1
                    continue
                rep.erased_bundle_ids.append(bid)
                erase_sessions.append(sid)
                if not dry_run:
                    self.s.evidence.erase(bid, "retention: evidence_days elapsed")
            rep.evidence_erased = len(rep.erased_bundle_ids)
            if self.s.timeline is not None and erase_sessions and not dry_run:
                rep.timeline_deleted += self.s.timeline.delete_sessions(sorted(set(erase_sessions)))
            if self.s.voiceprints is not None:
                rep.voiceprints_deleted = self._voiceprints(tid, pol, now, dry_run)
            reports.append(rep)
        # stores without a tenant column: the longest configured window wins (never too early)
        if policies:
            tl_days = max(p.retention.timeline_days for p in policies.values())
            fb_days = max(p.retention.feedback_days for p in policies.values())
            if not dry_run:
                if self.s.timeline is not None:
                    reports[0].timeline_deleted += self.s.timeline.delete_before(
                        _iso(now - dt.timedelta(days=tl_days))
                    )
                if self.s.feedback is not None:
                    reports[0].feedback_deleted += self.s.feedback.delete_before(
                        _iso(now - dt.timedelta(days=fb_days))
                    )
                if self.s.flagged_audio is not None:
                    reports[0].raw_audio_purged = self.s.flagged_audio.purge_expired(now)
        for r in reports:
            d = asdict(r)
            d.pop("erased_bundle_ids")
            self.audit.record(
                "retention_run",
                "retention_job",
                r.tenant_id,
                "retention",
                "legal_obligation",
                outcome="dry_run" if dry_run else "ok",
                details=d,
            )
        return reports

    def _voiceprints(self, tid: str, pol: TenantPrivacy, now: dt.datetime, dry_run: bool) -> int:
        n = 0
        limit = now - dt.timedelta(days=pol.retention.voiceprint_inactive_days)
        for spk in self.s.voiceprints.list_speakers(tid):
            vp = self.s.voiceprints.get(tid, spk)
            if vp is None:
                continue
            last = dt.datetime.fromisoformat(vp.updated_at or vp.created_at)
            if vp.expired(now) or last < limit:
                n += 1
                if not dry_run:
                    self.s.voiceprints.delete_speaker(tid, spk)
                    self.audit.record(
                        "voiceprint_deleted",
                        "retention_job",
                        tid,
                        "speaker_enrollment",
                        "explicit_consent",
                        spk,
                        details={"reason": "expired" if vp.expired(now) else "inactive"},
                    )
        return n


@dataclass
class Subject:
    """At least one identifier of the data subject."""

    claimed_identity_id: str | None = None
    caller_number: str | None = None
    speaker_id: str | None = None

    def ref(self) -> str:
        key = self.claimed_identity_id or self.caller_number or self.speaker_id or ""
        return hash_identifier(key)

    def matches(self, bundle: dict[str, Any]) -> bool:
        cm = bundle.get("call_metadata") or {}
        return bool(
            (self.claimed_identity_id and cm.get("claimed_identity_id") == self.claimed_identity_id)
            or (self.caller_number and cm.get("caller_number") == self.caller_number)
        )


class DSAR:
    def __init__(
        self, stores: Stores, audit: AuditLog, config_dir: Path | str | None = None
    ) -> None:
        self.s = stores
        self.audit = audit
        self.config_dir = config_dir

    def _bundles(self, tenant_id: str, subject: Subject) -> list[dict[str, Any]]:
        return [b for b in self.s.evidence.iter_bodies(tenant_id) if subject.matches(b)]

    def export(self, tenant_id: str, subject: Subject, actor: str) -> dict[str, Any]:
        bundles = self._bundles(tenant_id, subject)
        sessions = sorted({b["session_id"] for b in bundles})
        out: dict[str, Any] = {
            "tenant_id": tenant_id,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "evidence_bundles": bundles,
            "score_timeline": {},
            "analyst_feedback": [],
            "voiceprint": None,
            "raw_audio_retained": [],
            "note": "Raw call audio is not retained on the default path. Voiceprint templates are "
            "described, not exported: the biometric template is available only through the "
            "DPO over a secure channel.",
        }
        if self.s.timeline is not None:
            out["score_timeline"] = {
                s: [asdict(r) for r in self.s.timeline.session(s)] for s in sessions
            }
        if self.s.feedback is not None:
            out["analyst_feedback"] = [
                r for r in self.s.feedback.latest().values() if r["session_id"] in sessions
            ]
        spk = subject.speaker_id or subject.claimed_identity_id
        if self.s.voiceprints is not None and spk:
            vp = self.s.voiceprints.get(tenant_id, spk)
            if vp is not None:
                out["voiceprint"] = {
                    "speaker_id": vp.speaker_id,
                    "embedder": vp.embedder,
                    "consent_ref": vp.consent_ref,
                    "created_at": vp.created_at,
                    "updated_at": vp.updated_at,
                    "expires_at": vp.expires_at,
                    "enrolment_sessions": [asdict(x) for x in vp.sessions],
                    "templates": {c: len(v) for c, v in vp.embeddings.items()},
                }
        if self.s.flagged_audio is not None:
            out["raw_audio_retained"] = [
                s for s in sessions if self.s.flagged_audio.get(s, tenant_id) is not None
            ]
        self.audit.record(
            "dsar_export",
            actor,
            tenant_id,
            "data_subject_request",
            "legal_obligation",
            subject.ref(),
            details={"sessions": len(sessions), "bundles": len(bundles)},
        )
        return out

    def erase(
        self, tenant_id: str, subject: Subject, actor: str, dry_run: bool = False
    ) -> dict[str, Any]:
        pol = TenantPrivacy.load(tenant_id, self.config_dir)
        bundles = self._bundles(tenant_id, subject)
        held = [b["session_id"] for b in bundles if b["session_id"] in set(pol.legal_holds)]
        todo = [b for b in bundles if b["session_id"] not in set(pol.legal_holds)]
        sessions = sorted({b["session_id"] for b in todo})
        res: dict[str, Any] = {
            "evidence_erased": 0,
            "timeline_deleted": 0,
            "feedback_deleted": 0,
            "raw_audio_deleted": 0,
            "voiceprint_deleted": False,
            "held_sessions": sorted(set(held)),
            "dry_run": dry_run,
        }
        if not dry_run:
            for b in todo:
                res["evidence_erased"] += int(
                    self.s.evidence.erase(b["bundle_id"], "data subject erasure")
                )
            if self.s.timeline is not None:
                res["timeline_deleted"] = self.s.timeline.delete_sessions(sessions)
            if self.s.feedback is not None:
                res["feedback_deleted"] = self.s.feedback.delete_sessions(sessions)
            if self.s.flagged_audio is not None:
                res["raw_audio_deleted"] = self.s.flagged_audio.delete(sessions)
            spk = subject.speaker_id or subject.claimed_identity_id
            if self.s.voiceprints is not None and spk:
                res["voiceprint_deleted"] = self.s.voiceprints.delete_speaker(tenant_id, spk)
        else:
            res["evidence_erased"] = len(todo)
        self.audit.record(
            "dsar_erasure",
            actor,
            tenant_id,
            "data_subject_request",
            "legal_obligation",
            subject.ref(),
            outcome="dry_run" if dry_run else ("partial" if held else "ok"),
            details={k: v for k, v in res.items() if k != "held_sessions"} | {"held": len(held)},
        )
        return res
