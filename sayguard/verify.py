"""Second opinion on no-helmet alerts.

A detector that sees a bare head for a few frames is often wrong (a hood, dark hair
under a cap, a helmet held in the hand).  Before an alert counts as confirmed, a second
model looks at the head-and-shoulders crop and answers one question: is this person
wearing a hard hat?  That is a yes/no decision, so the default backend is not a
generative model at all:

  clip_probe  CLIP ViT-B/32 image encoder (frozen) + a 2x512 linear head trained on site
              crops (scripts/train_probe.py).  ~14 ms per alert on the PC GPU, one crop at a
              time (docs/results/probe_train.json); on the UGen300 the
              encoder is the Model Zoo clip_vit_b_32_image_encoder HEF and the head is one
              host-side matrix multiply.  Same weights file as YOLO-World's text encoder.
  vlm         Qwen2-VL-2B-Instruct asked the same question in words (Hailo GenAI zoo).
              Kept as an option; on the same crops it was less accurate and, one crop at a
              time, ~12x slower (160 ms; docs/results/yesno_check.json).

Checks run in their own thread with a bounded queue, so a slow or missing model can never
stall the camera loop, and the model is loaded lazily (after a memory check) because many
sessions never raise an alert.
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
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PIL import Image

from .resources import IMPORT_LOCK, ResourceError, require
from .types import Event

if TYPE_CHECKING:
    from .events import EventStore

log = logging.getLogger("sayguard.verify")

QUEUE_MAX = 32
RETRY_S = 120.0

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


class Backend(Protocol):
    label: str          # shown in alert notes and on the dashboard
    device: str | None

    def load(self) -> None: ...
    def judge(self, crop_bgr: np.ndarray) -> tuple[bool | None, str]:
        """(wearing a hard hat?, short detail); None = could not tell."""
    def unload(self) -> None: ...


# --------------------------------------------------------------------------- CLIP probe
class LinearProbe:
    """2-way linear head on unit-norm CLIP features, as trained by scripts/train_probe.py.
    Class 1 = wearing a hard hat, class 0 = bare head."""

    def __init__(self, W: np.ndarray, b: np.ndarray, scale: float = 10.0):
        self.W = np.asarray(W, np.float32).reshape(2, -1)
        self.b = np.asarray(b, np.float32).reshape(2)
        self.scale = float(scale)

    @classmethod
    def load(cls, path: str | Path) -> LinearProbe:
        with np.load(path) as z:
            return cls(z["W"], z["b"], float(z["scale"]))

    def save(self, path: str | Path, **meta) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b, scale=np.float32(self.scale), **meta)

    def prob_wearing(self, feats: np.ndarray) -> np.ndarray:
        f = np.asarray(feats, np.float32).reshape(len(feats), -1)
        f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        z = f * self.scale @ self.W.T + self.b
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e[:, 1] / e.sum(1)


class ClipProbeBackend:
    label = "CLIP probe"

    def __init__(self, clip_path: str, probe_path: str):
        self.clip_path, self.probe_path = clip_path, probe_path
        self.device: str | None = None
        self._model = self._preprocess = self._probe = None

    def load(self) -> None:
        if not Path(self.probe_path).is_file():
            raise FileNotFoundError(f"no probe at {self.probe_path} (run scripts/train_probe.py)")
        probe = LinearProbe.load(self.probe_path)   # cheap: a bad file fails before the 354 MB encoder loads
        self.load_encoder()
        self._probe = probe

    def load_encoder(self) -> None:
        """The image encoder alone (scripts/train_probe.py trains the head on its features)."""
        if not Path(self.clip_path).is_file():
            raise FileNotFoundError(f"no CLIP ViT-B/32 weights at {self.clip_path}")
        import torch
        cuda = torch.cuda.is_available()
        require("CLIP probe", gpu_gb=0.6 if cuda else 0.0, ram_gb=1.0)
        with IMPORT_LOCK:
            import clip
        # An absolute file path makes clip.load read it directly: no download, no hash pass.
        model, preprocess = clip.load(str(Path(self.clip_path).resolve()), device="cuda" if cuda else "cpu")
        model.eval()
        del model.transformer, model.token_embedding   # text tower: YOLO-World keeps its own copy
        self._model, self._preprocess = model, preprocess
        self.device = "cuda" if cuda else "cpu"

    def embed(self, images: list[Image.Image], batch: int = 128) -> np.ndarray:
        """Raw (not normalized) 512-d image features; what the Hailo HEF outputs."""
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), batch):
                x = torch.stack([self._preprocess(im) for im in images[i:i + batch]]).to(self.device)
                out.append(self._model.encode_image(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def judge(self, crop_bgr: np.ndarray) -> tuple[bool | None, str]:
        p = float(self._probe.prob_wearing(self.embed([bgr_to_pil(crop_bgr)]))[0])
        return p >= 0.5, f"p(hard hat)={p:.2f}"

    def unload(self) -> None:
        self._model = self._preprocess = self._probe = None
        self.device = None


# --------------------------------------------------------------------------------- VLM
MIN_PIXELS = 224 * 224          # crops are small; upscale to >= 64 visual tokens
MAX_PIXELS = 448 * 448          # cap at 256 visual tokens so a check stays fast
MAX_NEW_TOKENS = 4
PROMPT = ("Look at the person in this image. Is the person wearing a hard hat (safety helmet) "
          "ON THEIR HEAD? A helmet held in the hand, hanging, or lying nearby does not count. "
          "Answer only yes or no.")


class VlmBackend:
    label = "VLM"

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device: str | None = None
        self._model = self._processor = None

    def load(self) -> None:
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
        self.device = "cuda" if cuda else "cpu"

    def judge(self, crop_bgr: np.ndarray) -> tuple[bool | None, str]:
        raw = self._ask(bgr_to_pil(crop_bgr))
        return parse_yes_no(raw), raw.strip()[:40]

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

    def unload(self) -> None:
        self._model = self._processor = None
        self.device = None


def make_backend(vcfg: dict, resolve) -> Backend:
    """`verifier:` config block -> backend.  Older configs had a `vlm:` block with `model:`."""
    kind = vcfg.get("backend", "vlm" if "model" in vcfg and "probe" not in vcfg else "clip_probe")
    if kind == "vlm":
        return VlmBackend(str(resolve(vcfg.get("vlm_model") or vcfg.get("model", ""))))
    if kind == "clip_probe":
        return ClipProbeBackend(str(resolve(vcfg.get("clip", "@work/models/clip/ViT-B-32.pt"))),
                                str(resolve(vcfg.get("probe", "@work/models/helmet_probe.npz"))))
    raise ValueError(f"unknown verifier backend {kind!r} (clip_probe | vlm)")


# ------------------------------------------------------------------------------ worker
class Verifier:
    def __init__(self, backend: Backend, store: EventStore, enabled: bool = True):
        self.backend = backend
        self.store = store
        self.enabled = enabled
        self._q: queue.Queue[tuple[Event, np.ndarray] | None] = queue.Queue(maxsize=QUEUE_MAX)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = "idle" if enabled else "disabled"
        self._reason = "" if enabled else "disabled in config"
        self._fail_ts = 0.0
        self._ready = False
        self._checked = self._confirmed = self._false_alarms = 0
        self._total_ms = 0.0

    @property
    def label(self) -> str:
        return self.backend.label

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="verifier", daemon=True)
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
            self._mark(ev.id, None, f"{self.label} disabled")
            return
        try:
            self._q.put_nowait((ev, crop_bgr))
        except queue.Full:
            self._mark(ev.id, None, f"{self.label} queue full")

    def status(self) -> dict:
        with self._lock:
            return {"state": self._state, "reason": self._reason, "queue": self._q.qsize(),
                    "backend": self.label,
                    "checked": self._checked, "confirmed": self._confirmed,
                    "false_alarms": self._false_alarms,
                    "avg_ms": round(self._total_ms / self._checked, 1) if self._checked else 0.0,
                    "device": self.backend.device}

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
                log.exception("%s check failed for %s", self.label, ev.id)
                self._mark(ev.id, None, f"{self.label} error: {type(e).__name__}")

    def _handle(self, ev: Event, crop: np.ndarray) -> None:
        L = self.label
        if ev.rule_type != "no_helmet":
            self._mark(ev.id, None, "not checked")
            return
        if crop is None or crop.ndim != 3 or crop.shape[0] < 8 or crop.shape[1] < 8:
            self._mark(ev.id, None, f"{L}: no usable crop")
            return
        if not self._ensure_model():
            self._mark(ev.id, None, f"{L} skipped: {self._reason}")
            return
        t0 = time.perf_counter()
        wearing, detail = self.backend.judge(crop)
        ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            self._checked += 1
            self._total_ms += ms
            if wearing is True:
                self._false_alarms += 1
            elif wearing is False:
                self._confirmed += 1
        if wearing is True:
            self._mark(ev.id, False, f"{L}: wearing a hard hat (false alarm, {detail})")
        elif wearing is False:
            self._mark(ev.id, True, f"{L}: no hard hat (confirmed, {detail})")
        else:
            self._mark(ev.id, None, f"{L}: unclear ({detail})")

    def _mark(self, event_id: str, verified: bool | None, note: str) -> None:
        try:
            self.store.set_verified(event_id, verified, note)
        except Exception:
            log.exception("could not store verdict for %s", event_id)

    def _set_state(self, state: str, reason: str = "") -> None:
        with self._lock:
            self._state, self._reason = state, reason

    # ----------------------------------------------------------------- model
    def _ensure_model(self) -> bool:
        if self._ready:
            return True
        if self._state == "unavailable" and time.time() - self._fail_ts < RETRY_S:
            return False
        self._set_state("loading")
        reason: str | None = None
        try:
            self.backend.load()
        except Exception as e:  # missing weights, no memory, driver trouble: alerts stay unchecked
            reason = str(e) if isinstance(e, ResourceError) else f"{type(e).__name__}: {e}"
        if reason is not None:
            # Clean up only once the handler has exited: inside it e.__traceback__ still pins the
            # frames of load/from_pretrained and with them the half-loaded model, so gc and
            # empty_cache freed nothing and GBs of VRAM stayed reserved for every later retry.
            self._fail_ts = time.time()
            self.backend.unload()
            self._set_state("unavailable", reason[:200])
            log.warning("%s unavailable: %s", self.label, reason)
            self._release_partial()
            return False
        self._ready = True
        self._set_state("ready")
        log.info("%s ready on %s", self.label, self.backend.device)
        return True

    def _release_partial(self) -> None:
        """Drop whatever a failed load left behind (never imports torch just for this)."""
        torch = sys.modules.get("torch")
        try:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
