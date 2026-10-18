"""Open-vocabulary detection (YOLO-World v2-S) for objects that only exist because
someone said a rule about them ("forklift", "ladder", ...).  No retraining: the
class list is just text.  Hailo-10H runs yolo_world_v2s at 46.6 FPS (Hailo Model
Zoo), so it is scheduled on a few cameras every Nth frame, not on all frames.

Changing the class list means encoding the names with CLIP's text tower: seconds,
plus loading the CLIP checkpoint the first time.  So the pipeline calls set_classes()
from a background thread.  The slow encoding runs WITHOUT the inference lock; only
the swap of embeddings/names (and the predictor rebuild that follows it) holds the
lock, and infer() never waits for it: while a swap holds the model it returns no
detections, so the camera loop keeps its pace.  ready_for() tells the caller whether
the loaded classes are the ones it wants yet.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..types import Detection

CLIP_FILE = "ViT-B-32.pt"   # OpenAI CLIP ViT-B/32: the text encoder YOLO-World v2 was trained with
TEXT_BATCH = 80             # same chunking as ultralytics WorldModel.get_text_pe


def find_clip_weights(weights: str, clip_weights: str | None = None) -> Path:
    """Absolute path of a local CLIP ViT-B/32 checkpoint.  Never downloads.

    Ultralytics itself loads CLIP via clip.load('ViT-B/32', download_root=WEIGHTS_DIR/'clip'),
    and WEIGHTS_DIR comes from its settings.json, where weights_dir is the *relative*
    'weights' on this machine: whatever the server's cwd is.  A miss there means a 354 MB
    download (an offline site PC just fails), and a hit still SHA-256s the whole file in
    one read.  Handing clip.load an absolute file path skips both, so the model always
    comes from our own models folder (D:/ugen-work/models/clip/ViT-B-32.pt, next to the
    YOLO-World weights).  Search order: explicit path, next to the YOLO-World weights,
    work_dir()/models/clip, then clip's own cache ~/.cache/clip.
    """
    if clip_weights:
        cands = [Path(clip_weights)]
    else:
        from ..config import work_dir

        w = Path(weights)
        cands = ([w.parent / "clip" / CLIP_FILE] if w.is_absolute() else []) + [
            work_dir() / "models" / "clip" / CLIP_FILE, Path.home() / ".cache" / "clip" / CLIP_FILE]
    for p in cands:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"CLIP ViT-B/32 text encoder not found; place {CLIP_FILE} at {cands[0]}")


class OpenVocabDetector:
    def __init__(self, weights: str, conf: float = 0.2, imgsz: int = 640, *, clip_weights: str | None = None):
        if not Path(weights).is_file():   # ultralytics would try to download a missing file
            raise FileNotFoundError(f"YOLO-World weights not found: {weights}")
        self._clip_path = find_clip_weights(weights, clip_weights)

        import torch
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(weights)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.conf, self.imgsz = conf, imgsz
        self.classes: list[str] = []        # what infer() labels with; [] = not ready
        self._loaded: list[str] | None = None   # names the model's text embeddings encode
        self._clip = None                   # CLIP text encoder, kept once loaded (~0.6 GB host RAM)
        self._lock = threading.Lock()        # guards self.model: predict vs. class swap
        self._update_lock = threading.Lock() # one set_classes (CLIP encoding) at a time

    # -------------------------------------------------------------- classes
    def ready_for(self, names: set[str]) -> bool:
        """True when the loaded classes are exactly `names` (blank names ignored)."""
        return self.classes == sorted(n for n in names if n)

    def set_classes(self, names: set[str]) -> None:
        """Load a new class list.  SLOW (CLIP); call it off the camera thread."""
        new = sorted(n for n in names if n)
        with self._update_lock:
            if new == self.classes:
                return
            if not new or new == self._loaded:      # nothing to encode
                with self._lock:
                    self.classes = new
                return
            feats = self._text_embeddings(new)      # infer() keeps running meanwhile
            with self._lock:
                # Not ready until the swap and warm-up both succeed: a failure part-way must
                # not leave old names mapped onto new embeddings.
                self.classes, self._loaded = [], None
                self._swap(new, feats)
                self._predict([np.zeros((self.imgsz, self.imgsz, 3), np.uint8)])
                self.classes, self._loaded = new, list(new)

    def _text_encoder(self):
        if self._clip is None:
            from ultralytics.nn.text_model import CLIP

            # Same device get_text_pe would use: the master WorldModel's (CPU, since the
            # predictor runs on its own deep copy).  An absolute path makes clip.load read
            # this file instead of resolving 'ViT-B/32' under WEIGHTS_DIR (see find_clip_weights).
            device = next(self.model.model.model.parameters()).device
            self._clip = CLIP(str(self._clip_path), device)
        return self._clip

    def _text_embeddings(self, names: list[str]):
        """What WorldModel.get_text_pe(names) returns, using a CLIP we keep ourselves.

        WorldModel's own cache (set_classes(..., cache_clip_model=True)) stores CLIP as a
        submodule of the detector, so every predictor rebuild would deep-copy the 150M-param
        CLIP and move it to the GPU too; keeping it here costs one copy on the host only."""
        import torch

        clip = self._text_encoder()
        tokens = clip.tokenize(names)
        feats = [clip.encode_text(t).detach() for t in tokens.split(TEXT_BATCH)]
        feats = feats[0] if len(feats) == 1 else torch.cat(feats, dim=0)
        return feats.reshape(-1, len(names), feats.shape[-1])

    def _swap(self, names: list[str], feats) -> None:
        """YOLOWorld.set_classes minus the CLIP work (ultralytics 8.4.182: no background
        class to strip): embeddings + head size, names, and drop the predictor, whose deep
        copy of the model still holds the old embeddings."""
        wm = self.model.model
        wm.txt_feats = feats
        wm.model[-1].nc = len(names)
        wm.names = list(names)
        self.model.predictor = None

    # ------------------------------------------------------------ inference
    def _predict(self, frames: list[np.ndarray]):
        return self.model.predict(frames, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                  quantize=16 if self.device == 0 else 32, verbose=False)

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        if not self._lock.acquire(blocking=False):  # a class swap holds the model: skip, never stall
            return [[] for _ in frames]
        try:
            names = list(self.classes)
            if not names:
                return [[] for _ in frames]
            results = self._predict(frames)
        finally:
            self._lock.release()
        out = []
        for r in results:
            b = r.boxes
            xyxy, cls, conf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
            out.append([Detection(tuple(map(float, xyxy[i])), names[cls[i]], float(conf[i]), source="ov")
                        for i in range(len(cls)) if cls[i] < len(names)])
        return out
