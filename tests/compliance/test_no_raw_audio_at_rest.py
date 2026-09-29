"""B16 compliance suite: invariant I5 (no raw audio at rest) and ephemeral processing.

The default path is exercised end to end through the public gateway API (file
analysis + live REST streaming) with every store file-backed in a temp dir.
Afterwards every file written anywhere during the run — the data dir and the
system temp dir — is scanned for the input audio in each representation it
could take on disk: the uploaded WAV bytes, raw PCM16, and float32 PCM at 16 kHz.
"""

from __future__ import annotations

import base64
import builtins
import io
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from packages.vg_core.sample_store import SampleStore, drop_session, get_samples, put_samples
from packages.vg_core.stub_head import StubHead
from sdks.python.voiceguard import call_metadata
from services.api_gateway.app import create_app
from services.api_gateway.auth import KeyStore
from services.api_gateway.pipeline import Services
from services.api_gateway.webhooks import WebhookRegistry
from services.fusion.persistence import SQLiteTimelineStore
from services.policy.evidence import EvidenceStore

SR = 16000


def distinctive_audio(seconds: float = 6.0) -> np.ndarray:
    """Voiced speech-like signal with a random fine structure (so byte patterns are unique)."""
    rng = np.random.default_rng(1234)
    t = np.arange(int(seconds * SR)) / SR
    x = 0.3 * np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return (x + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def needles(pcm: np.ndarray, wav_bytes: bytes, n: int = 40) -> list[bytes]:
    """Byte fragments that would reveal the audio if they appear in any stored file."""
    rng = np.random.default_rng(0)
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
    f32 = pcm.astype("<f4").tobytes()
    out = []
    for blob in (pcm16, f32, wav_bytes[44:]):
        for off in rng.integers(0, len(blob) - 64, n):
            out.append(blob[int(off) : int(off) + 32])
    return out


class WriteSpy:
    """Records every path opened for writing through Python's open()."""

    def __init__(self) -> None:
        self.paths: set[Path] = set()
        self._orig = builtins.open

    def __enter__(self) -> WriteSpy:
        spy = self

        def patched(file: Any, mode: str = "r", *a: Any, **kw: Any) -> Any:  # noqa: ANN401
            if isinstance(file, str | os.PathLike) and any(c in mode for c in "wax+"):
                spy.paths.add(Path(file).resolve())
            return spy._orig(file, mode, *a, **kw)

        builtins.open = patched  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        builtins.open = self._orig  # type: ignore[assignment]


def files_under(*roots: Path) -> set[Path]:
    out = set()
    for r in roots:
        if r.exists():
            for p in r.rglob("*"):
                if p.is_file():
                    out.add(p.resolve())
    return out


@pytest.mark.compliance
def test_no_raw_pcm_reaches_disk_on_default_path(tmp_path: Path) -> None:
    data = tmp_path / "var"
    data.mkdir()
    tmp_root = Path(tempfile.gettempdir())
    before_tmp = files_under(tmp_root) if tmp_root.exists() else set()

    pcm = distinctive_audio()
    buf = io.BytesIO()
    sf.write(buf, pcm, SR, format="WAV", subtype="PCM_16")
    wav = buf.getvalue()

    keys = KeyStore()
    raw_key, _ = keys.create("bank-a", {"stream", "analyze", "evidence:read"})
    svc = Services(
        evidence=EvidenceStore(data / "evidence.db"),
        timeline=SQLiteTimelineStore(data / "timeline.db"),
        heads_factory=lambda: [StubHead(h, abstain_fraction=0.0) for h in "ABC"],
    )
    client = TestClient(create_app(svc, keys, WebhookRegistry()))
    hdr = {"Authorization": f"Bearer {raw_key}"}

    with WriteSpy() as spy:
        body = {
            "call_metadata": call_metadata("bank-a"),
            "audio_base64": base64.b64encode(wav).decode(),
        }
        assert client.post("/v1/analyze/file", json=body, headers=hdr).status_code == 200
        sid = client.post(
            "/v1/sessions", json={"call_metadata": call_metadata("bank-a")}, headers=hdr
        ).json()["session_id"]
        pcm16 = (pcm * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm16), 8000):
            r = client.post(
                f"/v1/sessions/{sid}/audio",
                headers=hdr,
                json={"payload_base64": base64.b64encode(pcm16[i : i + 8000]).decode()},
            )
            assert r.status_code == 200
        assert client.post(f"/v1/sessions/{sid}/close", headers=hdr).status_code == 200

    # stores really were written (the test would be vacuous otherwise)
    assert (data / "evidence.db").stat().st_size > 0 and (data / "timeline.db").stat().st_size > 0
    new_tmp = (files_under(tmp_root) - before_tmp) if tmp_root.exists() else set()
    candidates = files_under(data) | new_tmp | {p for p in spy.paths if p.exists()}
    probes = needles(pcm, wav)
    leaks = []
    for f in candidates:
        try:
            blob = f.read_bytes()
        except OSError:
            continue
        if blob[:4] in (b"RIFF", b"OggS", b"fLaC") or any(n in blob for n in probes):
            leaks.append(str(f))
    assert not leaks, f"raw audio found at rest: {leaks}"
    assert get_samples(f"shm://{sid}/0") is None  # the session's windows left memory too


@pytest.mark.compliance
def test_sample_store_overwrites_buffers_when_they_leave() -> None:
    sid = f"c-{uuid.uuid4().hex[:6]}"
    src = np.ones(1600, dtype=np.float32)
    put_samples(f"shm://{sid}/0", src)
    view = get_samples(f"shm://{sid}/0")
    assert view is not None and not view.flags.writeable and view.sum() == 1600
    assert src.sum() == 1600  # the caller's own array is never touched
    drop_session(sid)
    assert view.sum() == 0.0  # overwritten, not just dereferenced

    store = SampleStore(max_entries=1, ttl_s=-1.0)
    store.put("shm://s/0", src)
    first = store._data["shm://s/0"][1]  # noqa: SLF001 - white-box check of eviction wipe
    store.put("shm://s/1", src)  # evicts the first
    assert first.sum() == 0.0 and len(store) == 1
    assert store.get("shm://s/1") is None  # past its TTL: expired, wiped on access
