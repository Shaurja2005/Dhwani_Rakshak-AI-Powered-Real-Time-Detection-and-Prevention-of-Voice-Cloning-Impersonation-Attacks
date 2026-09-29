"""Scorer adapters: turn any model into ``wav -> (score, p_spoof, window p_spoof)`` (B15-T01).

Score convention: higher = more bona fide (ASVspoof). NaN = abstained.

Model specs accepted by ``build_scorer`` (and ``make eval MODEL=...``):

    head_a:<checkpoint.pt>        Head A model, 3 s windows / 1 s hop, mean raw score
    heads:A,B,C                   deployed DetectionHead classes (checkpoints from env)
    pipeline[:A,B,C]              full gateway pipeline incl. fusion (B9) — what customers get
    baseline:flatness             trivial DSP baseline (spectral flatness), a sanity floor
    <model version>               looked up in the model registry (models/released.yaml, B17)
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

SR = 16000
WIN = 3 * SR
HOP = 1 * SR


@dataclass
class ScoreOut:
    score: float  # higher = more bona fide; NaN = abstained
    p_spoof: float = float("nan")
    window_p: list[float] = field(default_factory=list)


class Scorer(Protocol):
    name: str
    model_version: str

    def score(self, wav: np.ndarray) -> ScoreOut: ...


def windows(wav: np.ndarray) -> list[np.ndarray]:
    if len(wav) <= WIN:
        return [wav]
    return [wav[i : i + WIN] for i in range(0, len(wav) - WIN + 1, HOP)]


class FlatnessBaseline:
    """Mean spectral flatness (Wiener entropy). Not a detector — a floor every model must beat."""

    name = "baseline:flatness"
    model_version = "baseline-flatness-v1"

    def score(self, wav: np.ndarray) -> ScoreOut:
        frames = np.lib.stride_tricks.sliding_window_view(wav, 512)[::256]
        if len(frames) == 0:
            return ScoreOut(float("nan"))
        spec = np.abs(np.fft.rfft(frames * np.hanning(512), axis=1)) ** 2 + 1e-12
        flat = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
        return ScoreOut(float(np.mean(flat)))


class HeadAScorer:
    name = "head_a"

    def __init__(
        self,
        checkpoint: str | None = None,
        model: torch.nn.Module | None = None,
        cal: tuple[float, float] = (8.0, 0.0),
    ) -> None:
        from packages.vg_models.heads.head_a_ssl.model import load_checkpoint

        if model is None:
            if checkpoint is None:
                raise ValueError("head_a scorer needs a checkpoint")
            model, meta = load_checkpoint(checkpoint)
            c = meta.get("calibration")
            cal = (float(c["scale"]), float(c["bias"])) if c else cal
            self.meta = meta
        else:
            self.meta = {}
        from packages.vg_models.heads.head_a_ssl.model import truncate_to_used_layers

        self.model_version = str(getattr(model, "model_version", "A@unknown"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # identical outputs for the layers the head reads, much less compute (see model.py)
        self.model = truncate_to_used_layers(model).eval().to(self.device)
        self.cal = cal

    @torch.no_grad()
    def raw_windows(self, wav: np.ndarray) -> np.ndarray:
        ws = windows(wav)
        amp = torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda")
        with amp:
            if len({len(w) for w in ws}) == 1:
                x = torch.from_numpy(np.stack(ws).astype(np.float32)).to(self.device)
                return self.model(x)[1].float().cpu().numpy()
            return np.array(
                [
                    float(
                        self.model(torch.from_numpy(w[None].astype(np.float32)).to(self.device))[1]
                    )
                    for w in ws
                ]
            )

    def score(self, wav: np.ndarray) -> ScoreOut:
        raw = self.raw_windows(wav)
        a, b = self.cal
        p = [1 / (1 + math.exp(a * r + b)) for r in raw]
        return ScoreOut(float(np.mean(raw)), float(np.mean(p)), p)


class HeadScorer:
    """Any deployed DetectionHead, window by window, exactly as in production."""

    def __init__(self, head: object) -> None:
        from packages.vg_core.models import SessionContext

        self.head = head
        self.head.warmup()  # type: ignore[attr-defined]
        self.name = f"head:{self.head.head_id}"  # type: ignore[attr-defined]
        self.model_version = str(self.head.model_version)  # type: ignore[attr-defined]
        self._ctx = SessionContext(session_id="eval", tenant_id="eval")

    def score(self, wav: np.ndarray) -> ScoreOut:
        from packages.vg_audio.quality import assess_quality
        from packages.vg_audio.vad import EnergyVAD
        from packages.vg_core.models import AnalysisWindow
        from packages.vg_core.sample_store import drop_session, put_samples

        sid = f"eval-{uuid.uuid4().hex[:8]}"
        ps: list[float] = []
        vad = EnergyVAD()
        for i, w in enumerate(windows(wav)):
            ratio, _ = vad.process_window(w)
            q = assess_quality(pcm=w, voiced_ratio=ratio, window_duration_s=len(w) / SR)
            ref = put_samples(f"shm://{sid}/{i}", w.astype(np.float32))
            aw = AnalysisWindow(
                session_id=sid,
                window_id=i,
                start_ms=i * 1000,
                end_ms=i * 1000 + int(len(w) / 16),
                samples_ref=ref,
                voiced_ms=q.voiced_ms,
                snr_db=q.snr_db,
                clipping_ratio=q.clipping_ratio,
                quality_ok=q.quality_ok,
                quality_flags=q.quality_flags,
                original_sample_rate=SR,
            )
            hs = self.head.score(aw, self._ctx)  # type: ignore[attr-defined]
            ps.append(float("nan") if hs.abstain or hs.p_spoof is None else float(hs.p_spoof))
        drop_session(sid)
        good = [p for p in ps if math.isfinite(p)]
        if not good:
            return ScoreOut(float("nan"), float("nan"), ps)
        return ScoreOut(-float(np.mean(good)), float(np.mean(good)), ps)


class PipelineScorer:
    """Full gateway path (quality gate → heads → fusion). Score = -session mean p_spoof."""

    def __init__(self, heads: str = "A,B,C,F") -> None:
        from services.api_gateway import pipeline as pl

        self._pl = pl
        wanted = heads
        self._services = pl.Services(heads_factory=lambda: _heads_from(wanted), load=None)
        self.name = f"pipeline:{heads}"
        hs = _heads_from(wanted)
        for h in hs:
            h.warmup()
        self.model_version = "pipeline[" + ",".join(str(h.model_version) for h in hs) + "]"

    def score(self, wav: np.ndarray) -> ScoreOut:
        from packages.vg_core.models import CallMetadata

        meta = CallMetadata(
            session_id=f"eval-{uuid.uuid4().hex[:8]}",
            tenant_id="eval",
            direction="inbound",
            started_at=dt.datetime.now(dt.UTC),
            channel="file",
            codec_hint="pcm",
            source_sample_rate=SR,
            consent_basis="legitimate_use",
        )
        pipe = self._pl.SessionPipeline(meta, self._services, with_context=False)
        pipe.push_pcm16k(wav.astype(np.float32))
        pipe.close()
        window_p = [
            float(w.p_spoof) if not _all_abstained(w) else float("nan") for w in pipe.window_scores
        ]
        good = [p for p in window_p if math.isfinite(p)]
        if not good:
            return ScoreOut(float("nan"), float("nan"), window_p)
        return ScoreOut(-float(np.mean(good)), float(np.mean(good)), window_p)


def _all_abstained(w: object) -> bool:
    return w.state.value == "ABSTAIN"  # type: ignore[attr-defined]


def _heads_from(spec: str) -> list[object]:
    import os

    from services.api_gateway.pipeline import default_heads

    old = os.environ.get("VG_GATEWAY_HEADS")
    os.environ["VG_GATEWAY_HEADS"] = spec
    try:
        return list(default_heads())
    finally:
        if old is None:
            os.environ.pop("VG_GATEWAY_HEADS", None)
        else:
            os.environ["VG_GATEWAY_HEADS"] = old


def build_scorer(spec: str) -> Scorer:
    kind, _, arg = spec.partition(":")
    if kind == "baseline":
        if arg != "flatness":
            raise ValueError(f"unknown baseline {arg!r}")
        return FlatnessBaseline()
    if kind == "head_a":
        return HeadAScorer(checkpoint=arg)
    if kind == "heads":
        heads = _heads_from(arg)
        if len(heads) != 1:
            raise ValueError("heads:<X> scores one head at a time; use pipeline:A,B,C for fusion")
        return HeadScorer(heads[0])
    if kind == "pipeline":
        return PipelineScorer(arg or "A,B,C,F")
    from packages.vg_models.registry import ModelRegistry

    entry = ModelRegistry().get(spec)
    if entry.kind == "head_a":
        return HeadAScorer(checkpoint=entry.path)
    raise ValueError(f"cannot build a scorer for registry entry kind {entry.kind!r}")
