"""VLM second opinion on no-helmet alerts.

A detector that sees a bare head for a few frames is often wrong (a hood, dark hair
under a cap, a helmet held in the hand).  Before an alert counts as confirmed, a
small VLM (Qwen2-VL-2B-Instruct, also in the Hailo GenAI Model Zoo) looks at the
person crop and answers one yes/no question.  It runs in its own thread with a
bounded queue so a slow or missing VLM can never stall the camera loop, and it is
loaded lazily (after a memory check) because many sessions never raise an alert.
"""
from __future__ import annotations

import gc
import logging
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .resources import IMPORT_LOCK, ResourceError, require
from .types import Event

if TYPE_CHECKING:
    from .events import EventStore

log = logging.getLogger("sayguard.verify")

QUEUE_MAX = 32
RETRY_S = 120.0
MIN_PIXELS = 224 * 224          # crops are small; upscale to >= 64 visual tokens
MAX_PIXELS = 448 * 448          # cap at 256 visual tokens so a check stays fast
MAX_NEW_TOKENS = 4
PROMPT = ("Look at the person in this image. Is the person wearing a hard hat (safety helmet) "
          "ON THEIR HEAD? A helmet held in the hand, hanging, or lying nearby does not count. "
          "Answer only yes or no.")

_YES = {"yes", "yeah", "yep", "y", "true", "wearing"}
_NO = {"no", "nope", "n", "not", "false", "none"}


def parse_yes_no(text: str) -> bool | None:
    """'Yes.' / ' no, ...' / '是' -> True/False; anything else -> None."""
    t = text.strip().lower()
    m = re.search(r"[a-z]+", t)
    if m and m.start() <= 2:
        if m.group(0) in _YES:
            return True
        if m.group(0) in _NO:
            return False
    if t.startswith(("是", "有")):
        return True
    if t.startswith(("否", "不", "沒", "没")):
        return False
    return None


def bgr_to_pil(crop_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(crop_bgr[:, :, ::-1]))


class Verifier:
    def __init__(self, model_dir: str, store: EventStore, enabled: bool = True):
        self.model_dir = model_dir
        self.store = store
        self.enabled = enabled
        self._q: queue.Queue[tuple[Event, np.ndarray] | None] = queue.Queue(maxsize=QUEUE_MAX)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = "idle" if enabled else "disabled"
        self._reason = "" if enabled else "disabled in config"
        self._fail_ts = 0.0
        self._model = None
        self._processor = None
        self._device: str | None = None
        self._checked = self._confirmed = self._false_alarms = 0
        self._total_ms = 0.0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="vlm-verifier", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def submit(self, ev: Event, crop_bgr: np.ndarray) -> None:
        if not self.enabled:
            self._mark(ev.id, None, "VLM disabled")
            return
        try:
            self._q.put_nowait((ev, crop_bgr))
        except queue.Full:
            self._mark(ev.id, None, "VLM queue full")

    def status(self) -> dict:
        with self._lock:
            return {"state": self._state, "reason": self._reason, "queue": self._q.qsize(),
                    "checked": self._checked, "confirmed": self._confirmed,
                    "false_alarms": self._false_alarms,
                    "avg_ms": round(self._total_ms / self._checked, 1) if self._checked else 0.0,
                    "device": self._device}

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            ev, crop = job
            try:
                self._handle(ev, crop)
            except Exception as e:  # one bad crop or CUDA hiccup must not kill the worker
                log.exception("VLM check failed for %s", ev.id)
                self._mark(ev.id, None, f"VLM error: {type(e).__name__}")

    def _handle(self, ev: Event, crop: np.ndarray) -> None:
        if ev.rule_type != "no_helmet":
            self._mark(ev.id, None, "not checked")
            return
        if crop is None or crop.ndim != 3 or crop.shape[0] < 8 or crop.shape[1] < 8:
            self._mark(ev.id, None, "VLM: no usable crop")
            return
        if not self._ensure_model():
            self._mark(ev.id, None, f"VLM skipped: {self._reason}")
            return
        t0 = time.perf_counter()
        raw = self._ask(bgr_to_pil(crop))
        ms = (time.perf_counter() - t0) * 1000
        wearing = parse_yes_no(raw)
        with self._lock:
            self._checked += 1
            self._total_ms += ms
            if wearing is True:
                self._false_alarms += 1
            elif wearing is False:
                self._confirmed += 1
        if wearing is True:
            self._mark(ev.id, False, "VLM: wearing a hard hat (false alarm)")
        elif wearing is False:
            self._mark(ev.id, True, "VLM: no hard hat (confirmed)")
        else:
            self._mark(ev.id, None, f"VLM: unclear ({raw.strip()[:40]})")

    def _mark(self, event_id: str, verified: bool | None, note: str) -> None:
        try:
            self.store.set_verified(event_id, verified, note)
        except Exception:
            log.exception("could not store VLM verdict for %s", event_id)

    def _set_state(self, state: str, reason: str = "") -> None:
        with self._lock:
            self._state, self._reason = state, reason

    # ----------------------------------------------------------------- model
    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._state == "unavailable" and time.time() - self._fail_ts < RETRY_S:
            return False
        self._set_state("loading")
        reason: str | None = None
        try:
            self._load()
        except Exception as e:  # missing weights, no memory, driver trouble: alerts stay unchecked
            reason = str(e) if isinstance(e, ResourceError) else f"{type(e).__name__}: {e}"
        if reason is not None:
            # Clean up only once the handler has exited: inside it e.__traceback__ still pins the
            # frames of _load/from_pretrained and with them the half-loaded model, so gc and
            # empty_cache freed nothing and GBs of VRAM stayed reserved for every later retry.
            self._fail_ts = time.time()
            self._model = self._processor = None
            self._set_state("unavailable", reason[:200])
            log.warning("VLM unavailable: %s", reason)
            self._release_partial()
            return False
        self._set_state("ready")
        log.info("VLM ready on %s", self._device)
        return True

    def _load(self) -> None:
        if not (Path(self.model_dir) / "config.json").is_file():
            raise FileNotFoundError(f"no model at {self.model_dir}")
        import torch  # already loaded by the detector in the server; needed to pick the budget

        cuda = torch.cuda.is_available()
        if cuda:
            # transformers 5 streams the safetensors shards one tensor at a time onto the GPU
            # (largest: embed_tokens, ~0.47 GB), so host memory peaks near 1-1.5 GB.  Asking
            # for 6 GB (+2 margin) here refused a model that fits in VRAM on a tight box.
            require("VLM", gpu_gb=5.0, ram_gb=1.5)
        else:
            require("VLM (CPU fp32)", ram_gb=10.0)
        with IMPORT_LOCK:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_dir, dtype=torch.float16 if cuda else torch.float32,
            device_map="cuda" if cuda else None, local_files_only=True)
        model.eval()
        gen = model.generation_config       # checkpoint ships do_sample=True; force plain greedy
        gen.do_sample, gen.temperature, gen.top_p, gen.top_k = False, None, None, None
        self._processor, self._model = processor, model
        with self._lock:
            self._device = "cuda" if cuda else "cpu"

    def _release_partial(self) -> None:
        """Drop whatever a failed load left behind (never imports torch just for this)."""
        torch = sys.modules.get("torch")
        try:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ask(self, img: Image.Image) -> str:
        import torch

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=[prompt], images=[img], return_tensors="pt",
                                 size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS})
        inputs = inputs.to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self._processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
