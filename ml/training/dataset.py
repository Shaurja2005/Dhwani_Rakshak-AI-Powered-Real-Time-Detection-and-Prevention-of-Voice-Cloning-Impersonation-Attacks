"""ml.training.dataset — manifest-backed audio dataset + family-balanced sampling (B4-T07).

Every row passes the license gate (I7) before any audio is read.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ml.data.license_gate import DataPolicy, enforce
from ml.data.manifest import ManifestRow

LoadFn = Callable[[ManifestRow], np.ndarray]


def soundfile_loader(data_root: Path | str) -> LoadFn:
    import soundfile as sf

    from ml.data.channel.base import resample

    root = Path(data_root)

    def load(row: ManifestRow) -> np.ndarray:
        x, sr = sf.read(str(root / row.path), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return resample(x, sr, 16000)

    return load


def crop_or_pad(x: np.ndarray, n: int, rng: np.random.Generator, train: bool) -> np.ndarray:
    if len(x) >= n:
        start = int(rng.integers(len(x) - n + 1)) if train else (len(x) - n) // 2
        return x[start : start + n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]  # repeat-pad, as in the ASVspoof baselines


class ManifestAudioDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Yields (wav[T], label 1=bona fide / 0=spoof, row index)."""

    def __init__(
        self,
        rows: Sequence[ManifestRow],
        policy: DataPolicy,
        load: LoadFn,
        seconds: float = 4.0,
        train: bool = True,
        augment: Callable[[np.ndarray, np.random.Generator, str], np.ndarray] | None = None,
        seed: int = 0,
        purpose: str = "train",
    ) -> None:
        self.rows = enforce(rows, policy, purpose=purpose)  # type: ignore[arg-type]
        self.load, self.n = load, int(seconds * 16000)
        self.train, self.augment, self.seed = train, augment, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[i]
        rng = np.random.default_rng((self.seed, self.epoch, i) if self.train else (self.seed, i))
        x = crop_or_pad(self.load(row), self.n, rng, self.train)
        if self.train and self.augment is not None:
            x = self.augment(x, rng, row.utt_id)
        return torch.from_numpy(np.ascontiguousarray(x)), int(row.label == "bona_fide"), i


def group_key(row: ManifestRow) -> str:
    return "bona_fide" if row.label == "bona_fide" else f"spoof:{row.generator_family}"


class FamilyBalancedSampler(Sampler[int]):
    """Half of each epoch is bona fide, the other half split *equally across attack families*,
    so the largest corpus / generator cannot silently dominate."""

    def __init__(
        self, rows: Sequence[ManifestRow], num_samples: int | None = None, seed: int = 0
    ) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            groups[group_key(r)].append(i)
        self.bona = groups.pop("bona_fide", [])
        self.families = dict(sorted(groups.items()))
        self.num_samples = num_samples or len(rows)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng((self.seed, self.epoch))
        n_bona = (
            self.num_samples // 2
            if self.bona and self.families
            else (self.num_samples if self.bona else 0)
        )
        n_spoof = self.num_samples - n_bona
        idx: list[int] = []
        if self.bona:
            idx += rng.choice(self.bona, size=n_bona, replace=len(self.bona) < n_bona).tolist()
        fams = list(self.families.values())
        for k, members in enumerate(fams):
            share = n_spoof // len(fams) + (1 if k < n_spoof % len(fams) else 0)
            idx += rng.choice(members, size=share, replace=len(members) < share).tolist()
        rng.shuffle(idx)
        return iter(idx)
