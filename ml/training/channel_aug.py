"""Channel augmentation for the feature cache (augmented views, B4-T12).

    python -m ml.training.channel_aug --config ml/training/configs/head_a_aug.yaml   # check

An *augmented view* is a second copy of a training clip that went through a random
call channel (codec at a random bitrate, noise, room, AGC, noise suppression, packet
loss) and optionally RawBoost, *before* the frozen XLS-R. The cache stores its
features next to the clean ones, so the back-end learns from both.

Invariant I3 (symmetric augmentation): the chain is drawn from an RNG seeded by
``(augment.seed, view, utt_id)`` only. The label never reaches this module, so
bona fide and spoof clips get identically distributed channels.

The ``check`` CLI augments a few real clips, prints the chains and the speed, and
fails loudly if a configured codec cannot be encoded on this machine.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ml.data.channel.codecs import FFMPEG_CODECS, codec_available
from ml.data.channel.noise import AdditiveNoise
from ml.data.channel.webrtc_chain import ChannelConfig, ChannelSimulator, seed_for
from ml.training.augment import Augmenter
from ml.training.feature_cache import SR, CropDataset

BUILTIN = ("clean", "g711u", "g711a")


def unavailable_codecs(aug: dict[str, Any]) -> list[str]:
    names = aug.get("channel", {}).get("codec_weights", {})
    return sorted(
        n for n in names if n not in BUILTIN and (n not in FFMPEG_CODECS or not codec_available(n))
    )


def build_augmenter(aug: dict[str, Any]) -> Augmenter:
    ch = dict(aug.get("channel", {}))
    snr = tuple(ch.pop("snr_range_db", (5.0, 20.0)))
    rates = {k: tuple(v) for k, v in (ch.pop("bitrates_kbps", {}) or {}).items()}
    if "opus" in rates:
        ch["opus_kbps"] = rates.pop("opus")
    if "loss_rates" in ch:
        ch["loss_rates"] = tuple(ch["loss_rates"])
    cfg = ChannelConfig(**ch, bitrates_kbps=rates)
    sim = ChannelSimulator(cfg, noise=AdditiveNoise(snr_range_db=snr))  # type: ignore[arg-type]
    return Augmenter(
        rawboost_algo=int(aug.get("rawboost_algo", 0)),
        p_rawboost=float(aug.get("p_rawboost", 0.0)),
        p_channel=float(aug.get("p_channel", 1.0)),
        sr=SR,
        channel=sim,
    )


def augment_clip(
    x: np.ndarray, utt_id: str, view: int, aug: dict[str, Any], augmenter: Augmenter
) -> tuple[np.ndarray, str]:
    """Deterministic in (seed, view, utt_id); returns the audio and a chain description."""
    seed = int(aug.get("seed", 0))
    key = f"v{view}:{utt_id}"
    y, parts = augmenter.run(x, np.random.default_rng(seed_for(key, seed)), key, seed)
    n = len(x)
    # codecs and resampling can shift the length by a few samples
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if not np.isfinite(y).all() or peak == 0.0:
        return x.astype(np.float32), "fallback:clean"  # never feed NaN/silence to XLS-R
    return np.clip(y, -1, 1).astype(np.float32), "+".join(parts) or "none"


class AugCropDataset(CropDataset):
    """CropDataset + one augmented view. Picklable; the augmenter is built per worker."""

    def __init__(
        self,
        paths: list[str],
        utt_ids: list[str],
        seconds: float,
        seed: int,
        view: int,
        aug: dict[str, Any],
    ) -> None:
        # a different crop per view (same utterance, different 4 s) adds diversity
        super().__init__(paths, seconds, train=True, seed=seed * 1009 + view)
        self.utt_ids, self.view, self.aug = utt_ids, view, aug
        self._augmenter: Augmenter | None = None

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:  # type: ignore[override]
        wav, _ = super().__getitem__(i)
        if self._augmenter is None:
            self._augmenter = build_augmenter(self.aug)
        y, chain = augment_clip(wav.numpy(), self.utt_ids[i], self.view, self.aug, self._augmenter)
        return torch.from_numpy(y), i, chain


def check(cfg: dict[str, Any], n: int = 40) -> int:
    from ml.training.feature_cache import load_rows, resolve

    aug = cfg["augment"]
    missing = unavailable_codecs(aug)
    if missing:
        print(
            f"FAIL: ffmpeg cannot encode {missing}. Install a full ffmpeg build "
            "(winget install Gyan.FFmpeg) or remove them from augment.channel.codec_weights."
        )
        return 1
    all_rows = load_rows(cfg, "train")
    rows = all_rows[:: max(1, len(all_rows) // n)][:n]
    root = resolve(cfg.get("data", {}).get("data_root", "data"))
    ds = AugCropDataset(
        [str(root / r.path) for r in rows],
        [r.utt_id for r in rows],
        float(cfg.get("train", {}).get("seconds", 4.0)),
        int(cfg.get("seed", 0)),
        1,
        aug,
    )
    t0 = time.time()
    kinds: Counter[str] = Counter()
    for i in range(len(ds)):
        wav, _, chain = ds[i]
        assert wav.shape[0] == ds.n and torch.isfinite(wav).all(), chain
        codec = next((p for p in chain.split("+") if p.startswith("codec:")), "codec:none")
        kinds[codec.split("@")[0]] += 1
        if i < 8:
            print(f"  {rows[i].utt_id}: {chain}")
    rate = len(ds) / (time.time() - t0)
    workers = int(cfg["cache"].get("workers", 6))
    print(f"codecs drawn: {dict(sorted(kinds.items()))}")
    print(
        f"augmentation speed: {rate:.1f} clips/s on one CPU process "
        f"(~{rate * workers:.0f} clips/s with {workers} workers)"
    )
    print("RESULT augmentation ready")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("-n", type=int, default=40)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    return check(cfg, args.n)


if __name__ == "__main__":
    sys.exit(main())
