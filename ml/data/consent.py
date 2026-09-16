"""ml.data.consent — consent register checks (B3-T11, invariant I9)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REGISTER_PATH = Path(__file__).with_name("consent_register.yaml")


class CorpusConsent(BaseModel):
    name: str
    approved_by: str
    date: dt.date
    basis: str


class SpeakerConsent(BaseModel):
    speaker_id: str
    name: str
    date: dt.date
    scope: str
    expires: dt.date | None = None
    withdrawal_contact: str
    form_ref: str


class ConsentRegister(BaseModel):
    corpora: list[CorpusConsent] = Field(default_factory=list)
    speakers: list[SpeakerConsent] = Field(default_factory=list)

    def allows(self, source_corpus: str, speaker_id: str, today: dt.date | None = None) -> bool:
        today = today or dt.date.today()
        if any(c.name == source_corpus for c in self.corpora):
            return True
        return any(
            s.speaker_id == speaker_id and (s.expires is None or s.expires >= today)
            for s in self.speakers
        )


def load_consent(path: Path | str = REGISTER_PATH) -> ConsentRegister:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ConsentRegister.model_validate({k: v or [] for k, v in raw.items()})
