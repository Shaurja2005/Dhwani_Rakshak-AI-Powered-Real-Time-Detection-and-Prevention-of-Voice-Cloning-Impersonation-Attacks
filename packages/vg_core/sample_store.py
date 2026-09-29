"""vg_core.sample_store — volatile store that resolves ``AnalysisWindow.samples_ref``.

The conditioner ``put``s a window's float32 PCM and hands heads a ``shm://`` ref;
heads ``get`` it. Memory only, bounded, with TTL — nothing ever touches disk
(invariant I5). A multi-process deployment can swap this for POSIX shared
memory behind the same three functions.

B16-T01 ephemeral processing: the store keeps its *own copy* of every window and
overwrites it with zeros when it leaves the store (session drop, TTL expiry,
eviction, clear), so raw PCM does not linger in freed memory either.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np

_MAX_ENTRIES = 4096
_TTL_S = 30.0


class SampleStore:
    def __init__(self, max_entries: int = _MAX_ENTRIES, ttl_s: float = _TTL_S) -> None:
        self._data: OrderedDict[str, tuple[float, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries
        self._ttl = ttl_s

    def put(self, ref: str, pcm: np.ndarray) -> str:
        arr = np.array(pcm, dtype=np.float32, copy=True)  # owned, so it can be wiped
        arr.setflags(write=False)
        with self._lock:
            old = self._data.pop(ref, None)
            if old is not None:
                _wipe(old[1])
            self._data[ref] = (time.monotonic(), arr)
            while len(self._data) > self._max:
                _wipe(self._data.popitem(last=False)[1][1])
        return ref

    def get(self, ref: str) -> np.ndarray | None:
        with self._lock:
            item = self._data.get(ref)
            if item is None:
                return None
            if time.monotonic() - item[0] > self._ttl:
                _wipe(self._data.pop(ref)[1])
                return None
            return item[1]

    def drop_session(self, session_id: str) -> int:
        with self._lock:
            keys = [k for k in self._data if f"/{session_id}/" in k]
            for k in keys:
                _wipe(self._data.pop(k)[1])
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            for _, arr in self._data.values():
                _wipe(arr)
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def _wipe(arr: np.ndarray) -> None:
    """Overwrite a buffer the store owns. Consumers only ever got read-only views of it."""
    arr.setflags(write=True)
    arr.fill(0.0)
    arr.setflags(write=False)


_store = SampleStore()


def put_samples(ref: str, pcm: np.ndarray) -> str:
    return _store.put(ref, pcm)


def get_samples(ref: str) -> np.ndarray | None:
    return _store.get(ref)


def drop_session(session_id: str) -> int:
    return _store.drop_session(session_id)
