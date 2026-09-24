"""Cross-session micro-batching for Head A (B14-T04/T06).

Windows from many concurrent calls are collected for at most ``max_wait_ms`` (or
until ``max_batch``) and run as one batch — the in-process equivalent of Triton's
dynamic batcher, used for the CPU tier and in tests.

Backpressure (invariant I10): the queue is bounded. ``submit`` raises
``OverloadedError`` immediately when it is full, and requests whose deadline has
passed are dropped before compute — audio is never queued behind live traffic.
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from services.inference.backends import InferenceBackend


class OverloadedError(RuntimeError):
    pass


@dataclass
class _Item:
    pcm: np.ndarray
    deadline: float
    future: cf.Future = field(default_factory=cf.Future)


@dataclass
class BatchStats:
    batches: int = 0
    items: int = 0
    rejected: int = 0
    expired: int = 0
    ema_batch_ms: float = 0.0

    @property
    def mean_batch(self) -> float:
        return self.items / self.batches if self.batches else 0.0


class BatchingInferenceServer:
    def __init__(
        self,
        backend: InferenceBackend,
        max_batch: int = 16,
        max_wait_ms: float = 10.0,
        max_queue: int = 64,
        default_timeout_ms: float = 1000.0,
    ) -> None:
        self.backend = backend
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.max_queue = max_queue
        self.default_timeout = default_timeout_ms / 1000
        self.stats = BatchStats()
        self._q: deque[_Item] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(target=self._loop, name="vg-batcher", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ client side
    def submit(self, pcm: np.ndarray, timeout_ms: float | None = None) -> cf.Future:
        timeout = self.default_timeout if timeout_ms is None else timeout_ms / 1000
        item = _Item(np.asarray(pcm, dtype=np.float32), time.monotonic() + timeout)
        with self._cv:
            if self._closed:
                raise OverloadedError("inference server closed")
            if len(self._q) >= self.max_queue:
                self.stats.rejected += 1
                raise OverloadedError(f"inference queue full ({self.max_queue})")
            self._q.append(item)
            self._cv.notify()
        return item.future

    def infer(self, pcm: np.ndarray, timeout_ms: float | None = None) -> float:
        timeout = self.default_timeout if timeout_ms is None else timeout_ms / 1000
        return float(self.submit(pcm, timeout_ms).result(timeout=timeout))

    @property
    def queue_depth(self) -> int:
        return len(self._q)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout=5)
        for it in self._q:
            it.future.set_exception(OverloadedError("inference server closed"))
        self._q.clear()

    # ------------------------------------------------------------------ worker
    def _take_batch(self) -> list[_Item]:
        with self._cv:
            while not self._q and not self._closed:
                self._cv.wait()
            if self._closed:
                return []
            batch_end = time.monotonic() + self.max_wait
            while len(self._q) < self.max_batch and not self._closed:
                left = batch_end - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(left)
            n = min(self.max_batch, len(self._q))
            return [self._q.popleft() for _ in range(n)]

    def _loop(self) -> None:
        while True:
            items = self._take_batch()
            if not items:
                if self._closed:
                    return
                continue
            now = time.monotonic()
            live = []
            for it in items:
                if it.deadline < now:  # stale: its caller has already abstained
                    self.stats.expired += 1
                    it.future.set_exception(TimeoutError("deadline passed before inference"))
                elif it.future.set_running_or_notify_cancel():
                    live.append(it)
            by_len: dict[int, list[_Item]] = {}
            for it in live:
                by_len.setdefault(len(it.pcm), []).append(it)
            for group in by_len.values():
                t0 = time.perf_counter()
                try:
                    out = self.backend.infer(np.stack([it.pcm for it in group]))
                except Exception as exc:  # noqa: BLE001 - propagate to callers, keep serving
                    for it in group:
                        it.future.set_exception(exc)
                    continue
                ms = (time.perf_counter() - t0) * 1000
                self.stats.ema_batch_ms = (
                    ms if self.stats.batches == 0 else 0.8 * self.stats.ema_batch_ms + 0.2 * ms
                )
                self.stats.batches += 1
                self.stats.items += len(group)
                for it, v in zip(group, out, strict=True):
                    it.future.set_result(float(v))
