"""Runtime helpers for long GPU jobs on a Windows laptop (base-model training).

* ``prepare_process()`` — opt this process out of Windows EcoQoS power throttling and
  ask Windows not to sleep *while this process runs* (per-process requests, no system
  setting is changed; both are released when the process exits).
* ``setup_cuda()`` — pick the device and enable the fast paths (cuDNN autotune, TF32).
* ``gpu_report()`` — one line of what the GPU is doing, for progress logs.

Measured in B14: a background Windows process was moved to efficiency cores and ran ~5x
slower after ~1 s of load. These helpers make training runs immune to that.
"""

from __future__ import annotations

import sys

import torch

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def disable_power_throttling() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class _State(ctypes.Structure):
        _fields_ = [("v", wintypes.ULONG), ("control", wintypes.ULONG), ("state", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    st = _State(1, 0x1 | 0x4, 0)  # EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION -> never throttle
    handle = k32.GetCurrentProcess()
    return bool(k32.SetProcessInformation(handle, 4, ctypes.byref(st), ctypes.sizeof(st)))


def keep_awake() -> bool:
    """Ask Windows not to sleep while this process is running (released on exit)."""
    if sys.platform != "win32":
        return False
    import ctypes

    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def prepare_process() -> dict[str, bool]:
    return {"power_throttling_disabled": disable_power_throttling(), "keep_awake": keep_awake()}


def setup_cuda(prefer: str | None = None) -> str:
    device = prefer or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True  # fixed 4 s crops -> autotuned kernels pay off
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def gpu_report() -> str:
    if not torch.cuda.is_available():
        return "gpu: none (CPU run)"
    used = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    util = ""
    try:
        util = f", util {torch.cuda.utilization(0)}%"
    except Exception:  # noqa: BLE001, S110 - pynvml not installed: skip utilisation
        pass
    return f"gpu: {torch.cuda.get_device_name(0)}, peak mem {used:.1f}/{total:.1f} GB{util}"
