"""Detector backends behind one interface, so the pipeline does not care whether
inference runs on a PC GPU (Stage I prototype) or on the UGen300 (Stage II)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..types import Detection


class Detector(Protocol):
    name: str

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """One list of detections per input frame (BGR uint8, any size)."""
        ...


def make_detector(cfg: dict) -> Detector:
    from ..config import resolve
    d = cfg["detector"]
    if d["backend"] == "ultralytics":
        from .ultralytics_backend import UltralyticsDetector
        return UltralyticsDetector(str(resolve(d["weights"])), imgsz=d["imgsz"], conf=d["conf"], half=d.get("half", True),
                                   classes=d.get("classes"), name=d.get("name"))
    if d["backend"] == "hailo":
        from .hailo_backend import HailoDetector
        return HailoDetector(str(resolve(d["hef"])), conf=d["conf"], labels=d.get("labels"))
    raise ValueError(f"unknown detector backend {d['backend']}")
