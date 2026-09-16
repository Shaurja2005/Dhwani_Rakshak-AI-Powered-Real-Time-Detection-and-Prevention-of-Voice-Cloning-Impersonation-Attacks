"""Speaker embedding models (B7-T01).

Candidates from the plan, all loaded lazily so nothing heavy is imported until used:

* ``ecapa``   — SpeechBrain ``speechbrain/spkrec-ecapa-voxceleb`` (192-d)
* ``wespeaker`` — WeSpeaker ResNet34 ONNX export (256-d), via onnxruntime
* ``titanet`` — NVIDIA NeMo ``titanet_large`` (192-d)

plus ``mfcc_stats``: a training-free classical baseline (MFCC and delta-MFCC
mean / standard deviation). It is **not** a usable verifier: its statistics
depend more on what is said than on who says it over a few seconds, and it is
very channel-sensitive. It exists only so the pipeline and tests run without
downloads; the head refuses to use it for live decisions unless explicitly
allowed (see head.py).

Pick the production embedder with ``benchmark.py`` on Indian-accented audio.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

SR = 16000


class Embedder(Protocol):
    name: str
    dim: int
    is_baseline: bool

    def embed(self, pcm: np.ndarray) -> np.ndarray: ...


def l2norm(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------- baseline
_N_MELS, _N_MFCC, _NFFT, _HOP = 40, 20, 512, 160


def _mel_fb_n(n_mels: int) -> np.ndarray:
    def hz2mel(f: float) -> float:
        return 2595 * np.log10(1 + f / 700)

    mels = np.linspace(hz2mel(20.0), hz2mel(7600.0), n_mels + 2)
    hz = 700 * (10 ** (mels / 2595) - 1)
    bins = np.floor((_NFFT + 1) * hz / SR).astype(int)
    fb = np.zeros((n_mels, _NFFT // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, lo:c] = (np.arange(lo, c) - lo) / max(c - lo, 1)
        fb[m - 1, c:hi] = (hi - np.arange(c, hi)) / max(hi - c, 1)
    return fb


_FB = _mel_fb_n(_N_MELS)
_DCT = np.cos(
    np.pi * np.arange(_N_MFCC)[:, None] * (2 * np.arange(_N_MELS)[None, :] + 1) / (2 * _N_MELS)
)


class MFCCStatsEmbedder:
    name = "mfcc_stats"
    is_baseline = True

    def __init__(self) -> None:
        self.dim = 4 * _N_MFCC

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(pcm, dtype=np.float64))
        if len(x) < _NFFT:
            x = np.pad(x, (0, _NFFT - len(x)))
        n = 1 + (len(x) - _NFFT) // _HOP
        fr = x[np.arange(_NFFT)[None, :] + _HOP * np.arange(n)[:, None]] * np.hamming(_NFFT)
        power = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        energy = 10 * np.log10(power.sum(axis=1) + 1e-12)
        keep = energy > energy.max() - 30  # speech frames only
        if keep.sum() < 10:
            keep[:] = True
        mfcc = np.log(power[keep] @ _FB.T + 1e-10) @ _DCT.T  # [T, 20]
        delta = np.diff(mfcc, axis=0, prepend=mfcc[:1])
        feats = np.concatenate([mfcc.mean(0), mfcc.std(0), delta.mean(0), delta.std(0)])
        # Standardise each block so no block dominates the cosine.
        blocks = feats.reshape(4, _N_MFCC)
        blocks = (blocks - blocks.mean(axis=1, keepdims=True)) / (
            blocks.std(axis=1, keepdims=True) + 1e-9
        )
        return l2norm(blocks.reshape(-1))


# ---------------------------------------------------------------- neural (lazy)
class SpeechBrainECAPA:
    name = "ecapa"
    dim = 192
    is_baseline = False

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "models/spkrec-ecapa",
    ) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier  # lazy, optional dependency

        self._torch = torch
        self._model = EncoderClassifier.from_hparams(
            source=source, savedir=savedir, run_opts={"device": "cpu"}
        )

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            emb = self._model.encode_batch(self._torch.tensor(pcm, dtype=self._torch.float32)[None])
        return l2norm(emb.squeeze().cpu().numpy())


class WeSpeakerONNX:
    name = "wespeaker"
    is_baseline = False

    def __init__(self, model_path: str = "models/wespeaker_resnet34.onnx") -> None:
        import onnxruntime as ort  # lazy, optional dependency

        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.dim = int(self._sess.get_outputs()[0].shape[-1])

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        # WeSpeaker ONNX exports take 80-d log-mel fbank [1, T, 80].
        x = np.asarray(pcm, dtype=np.float64)
        n = 1 + max(len(x) - 400, 0) // _HOP
        fr = np.pad(x, (0, 400))[
            np.arange(400)[None, :] + _HOP * np.arange(n)[:, None]
        ] * np.hamming(400)
        power = np.abs(np.fft.rfft(fr, n=_NFFT, axis=1)) ** 2
        fb80 = _mel_fb_n(80)
        feats = np.log(power @ fb80.T + 1e-10)
        feats -= feats.mean(axis=0)
        name = self._sess.get_inputs()[0].name
        out = self._sess.run(None, {name: feats[None].astype(np.float32)})[0]
        return l2norm(out.reshape(-1))


class NeMoTitaNet:
    name = "titanet"
    dim = 192
    is_baseline = False

    def __init__(self, model_name: str = "titanet_large") -> None:
        import torch
        from nemo.collections.asr.models import EncDecSpeakerLabelModel  # lazy, optional dependency

        self._torch = torch
        self._model = EncDecSpeakerLabelModel.from_pretrained(model_name).eval()

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        t = self._torch
        with t.inference_mode():
            sig = t.tensor(pcm, dtype=t.float32)[None]
            _, emb = self._model.forward(
                input_signal=sig, input_signal_length=t.tensor([sig.shape[1]])
            )
        return l2norm(emb.squeeze().cpu().numpy())


def build_embedder(name: str) -> Embedder:
    if name == "mfcc_stats":
        return MFCCStatsEmbedder()
    if name == "ecapa":
        return SpeechBrainECAPA()
    if name == "wespeaker":
        return WeSpeakerONNX()
    if name == "titanet":
        return NeMoTitaNet()
    raise ValueError(f"unknown embedder {name!r}")
