"""Head C analysis pipeline + temporal model (B6-T05).

``analyse`` turns one window into (a) language-normalisable global prosody
features and (b) a per-frame prosodic feature sequence.

``ProsodyNet``: a small bidirectional GRU over the frame sequence, mean+max
pooled, concatenated with the normalised global features, then an MLP -> spoof
logit. ~20k parameters; a few ms per window on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from packages.vg_models.heads.head_c_prosody.breath_rhythm import (
    breath_features,
    filled_pause_features,
    rhythm_features,
    speech_mask,
)
from packages.vg_models.heads.head_c_prosody.disfluency import Token, disfluency_features
from packages.vg_models.heads.head_c_prosody.normalize import LanguageNorm, min_pause_ms
from packages.vg_models.heads.head_c_prosody.pitch import contour_features, track

FRAME_FEATURES = ("f0_st_rel", "voiced", "energy_rel", "energy_delta", "f0_delta", "speech")


@dataclass
class ProsodyAnalysis:
    features: dict[str, float]
    frames: np.ndarray  # [T, len(FRAME_FEATURES)]
    voiced_s: float


def analyse(
    pcm: np.ndarray, language: str | None = None, tokens: list[Token] | None = None
) -> ProsodyAnalysis:
    x = np.nan_to_num(np.asarray(pcm, dtype=np.float64))
    p = track(x)
    speech = speech_mask(p)
    feats = contour_features(p)
    feats.update(breath_features(x, p, speech))
    feats.update(rhythm_features(p, speech, min_pause_ms(language)))
    feats.update(filled_pause_features(p))
    feats.update(disfluency_features(tokens, language))

    st = p.semitones
    med = float(np.median(st[p.voiced])) if p.voiced.any() else 0.0
    st_rel = np.where(p.voiced, (st - med) / 6.0, 0.0)
    e_rel = (p.frame_db - np.median(p.frame_db)) / 20.0
    seq = np.stack(
        [
            st_rel,
            p.voiced.astype(float),
            e_rel,
            np.diff(e_rel, prepend=e_rel[0]),
            np.diff(st_rel, prepend=st_rel[0]) * p.voiced,
            speech.astype(float),
        ],
        axis=1,
    ).astype(np.float32)
    return ProsodyAnalysis(feats, seq, float(p.voiced.sum() * 0.01))


class ProsodyNet(nn.Module):
    def __init__(self, n_global: int, n_frame: int = len(FRAME_FEATURES), hidden: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(n_frame, hidden, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden + n_global, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, frames: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(frames)  # B,T,2H
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        return self.mlp(torch.cat([pooled, globals_], dim=-1)).squeeze(-1)  # spoof logit


def global_vector(
    feats: dict[str, float], names: list[str], norm: LanguageNorm | None, language: str | None
) -> np.ndarray:
    if norm is not None:
        feats, _ = norm.apply(feats, language)
    vals = [feats.get(n, float("nan")) for n in names]
    # NaN (not measurable in this window) -> 0 = "average" after normalisation, clipped for safety.
    return np.clip(np.array([0.0 if math.isnan(v) else v for v in vals], dtype=np.float32), -8, 8)


def save(
    model: ProsodyNet, names: list[str], norm: LanguageNorm, path: Path | str, meta: dict[str, Any]
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "global_names": names, "norm": norm.to_json(), "meta": meta},
        str(path),
    )


def load(path: Path | str) -> tuple[ProsodyNet, list[str], LanguageNorm, dict[str, Any]]:
    blob = torch.load(
        str(path), map_location="cpu", weights_only=False
    )  # noqa: S614 - own artifacts
    names = list(blob["global_names"])
    model = ProsodyNet(len(names))
    model.load_state_dict(blob["state"])
    model.eval()
    return model, names, LanguageNorm.from_json(blob["norm"]), blob.get("meta", {})
