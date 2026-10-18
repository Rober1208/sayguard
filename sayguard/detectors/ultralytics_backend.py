"""PC backend: the fine-tuned YOLOv8s run with Ultralytics on CUDA (or CPU)."""
from __future__ import annotations

import numpy as np

from ..types import Detection


class UltralyticsDetector:
    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.35, half: bool = True):
        import torch
        from ultralytics import YOLO

        self.name = f"YOLOv8s-helmet (Ultralytics, {'cuda' if torch.cuda.is_available() else 'cpu'})"
        self.model = YOLO(weights)
        self.imgsz, self.conf = imgsz, conf
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = half and self.device == 0
        self.labels = self.model.names
        # warm-up so the first real batch is not slow
        self.model.predict([np.zeros((imgsz, imgsz, 3), np.uint8)], imgsz=imgsz, device=self.device,
                           quantize=16 if self.half else 32, verbose=False)

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        # helmet/head are two states of ONE head: class-aware NMS kept both boxes on an
        # ambiguous head, which became two tracks (double head count, and a "no hard hat"
        # alert for a worker whose helmet was also seen).  Agnostic NMS keeps the stronger.
        results = self.model.predict(frames, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                     quantize=16 if self.half else 32, agnostic_nms=True, verbose=False)
        out = []
        for r in results:
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            conf = b.conf.cpu().numpy()
            out.append([Detection(tuple(map(float, xyxy[i])), self.labels[cls[i]], float(conf[i]))
                        for i in range(len(cls))])
        return out
