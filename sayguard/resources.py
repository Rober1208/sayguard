"""Memory guards.  On this Windows box the binding limit is often the commit charge
(page-file backed), not free RAM, so check both plus free VRAM before loading a
model, and refuse with a reason instead of crashing the whole service."""
from __future__ import annotations

import ctypes
import os
import sys
import threading


class ResourceError(RuntimeError):
    pass


def gpu_free_gb() -> float | None:
    """VRAM (GB) a model loaded next in this process could use, or None if unknown.

    The driver's free figure (mem_get_info) counts memory PyTorch's caching allocator
    keeps reserved but no longer uses as taken, even though our next load reuses it.
    After a failed VLM load that cache (up to ~4 GB) made every retry refuse forever,
    so add reserved - allocated back.  torch is never imported just to ask: that alone
    costs host commit and, via mem_get_info, a CUDA context.  Before torch is loaded
    (pre-import checks in scripts) NVML answers instead; nothing of ours is cached then.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return _nvml_free_gb()
    try:
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        return (free + max(0, cached)) / 1e9
    except Exception:
        return None


def _nvml_free_gb() -> float | None:
    """Device-level free VRAM via NVML (ctypes only: no torch, no CUDA context)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    first = vis.split(",")[0].strip() if vis is not None else "0"
    if first in ("", "-1"):
        return None                       # GPUs hidden from this process
    try:
        import pynvml                      # nvidia-ml-py, pulled in by ultralytics; optional here
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        try:
            # NVML numbers GPUs in PCI order and CUDA fastest-first; they agree on a one-GPU box.
            h = (pynvml.nvmlDeviceGetHandleByIndex(int(first)) if first.isdigit()
                 else pynvml.nvmlDeviceGetHandleByUUID(first))
            return pynvml.nvmlDeviceGetMemoryInfo(h).free / 1e9
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def ram_available_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1e9


def commit_available_gb() -> float | None:
    if sys.platform != "win32":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPageFile / 1e9


def snapshot() -> dict:
    return {"gpu_free_gb": gpu_free_gb(), "ram_available_gb": round(ram_available_gb(), 1),
            "commit_available_gb": commit_available_gb()}


def require(what: str, gpu_gb: float = 0.0, ram_gb: float = 0.0, margin_gb: float = 2.0) -> None:
    """Raise ResourceError if loading `what` would leave less than margin_gb."""
    g = gpu_free_gb()
    if gpu_gb and g is not None and g < gpu_gb + 0.5:
        raise ResourceError(f"{what}: needs ~{gpu_gb:.1f} GB VRAM, only {g:.1f} GB free")
    if ram_gb:
        ra = ram_available_gb()
        if ra < ram_gb + margin_gb:
            raise ResourceError(f"{what}: needs ~{ram_gb:.1f} GB RAM, only {ra:.1f} GB available")
        c = commit_available_gb()
        if c is not None and c < ram_gb + margin_gb:
            raise ResourceError(f"{what}: needs ~{ram_gb:.1f} GB commit, only {c:.1f} GB left")


# transformers resolves classes lazily on first attribute access, and two threads doing that at
# once (the VLM worker and an ASR request) raised "cannot import name 'AutoProcessor'" in testing.
# Every heavy model import goes through this lock, and preload_model_libs() runs it once at startup.
IMPORT_LOCK = threading.RLock()


def preload_model_libs() -> str | None:
    """Import the transformers classes the VLM and ASR use, on the calling (main) thread."""
    try:
        with IMPORT_LOCK:
            from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,  # noqa: F401
                                      WhisperForConditionalGeneration, WhisperProcessor)
        return None
    except Exception as e:  # missing extra: the features that need it report it when used
        return f"{type(e).__name__}: {e}"
