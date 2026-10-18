"""Small shared data types."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HEAD_HEIGHT_M = 0.23   # typical adult head height incl. hard hat; gives a local metres/pixel scale


@dataclass
class Detection:
    """One box from a detector. label is 'helmet'/'head' for the PPE model or any
    open-vocabulary label (e.g. 'forklift') for YOLO-World."""
    xyxy: tuple[float, float, float, float]
    label: str
    conf: float
    track_id: int | None = None
    source: str = "ppe"            # 'ppe' | 'ov'

    @property
    def w(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def h(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2)


@dataclass
class Person:
    """A tracked worker, represented by the head box (helmet or bare head)."""
    track_id: int
    head: tuple[float, float, float, float]
    helmet_now: bool               # label of the current frame's detection
    no_helmet_ratio: float         # share of recent observations without a helmet
    observations: int

    @property
    def head_h(self) -> float:
        return max(1.0, self.head[3] - self.head[1])

    @property
    def metres_per_px(self) -> float:
        return HEAD_HEIGHT_M / self.head_h

    def body_box(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Rough full-body box below the head (standing person ~7.5 heads tall)."""
        x1, y1, x2, y2 = self.head
        hw, hh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        bx1, bx2 = cx - 1.6 * hw, cx + 1.6 * hw
        by1, by2 = y1 - 0.15 * hh, y1 + 7.5 * hh
        return (int(max(0, bx1)), int(max(0, by1)), int(min(frame_w - 1, bx2)), int(min(frame_h - 1, by2)))

    def ground_point(self) -> tuple[float, float]:
        """Approximate feet position (for distance to floor-level objects)."""
        x1, y1, x2, y2 = self.head
        return ((x1 + x2) / 2, y1 + 7.0 * (y2 - y1))


@dataclass
class FramePacket:
    cam_id: str
    index: int
    ts: float
    frame: np.ndarray


@dataclass
class Event:
    id: str
    ts: float
    cam_id: str
    cam_name: str
    zone: str
    rule_id: str
    rule_name: str
    rule_type: str
    severity: str
    message: str
    track_id: int | None = None
    box: tuple[int, int, int, int] | None = None
    snapshot: str | None = None
    verified: bool | None = None   # None = pending/not checked, True = confirmed, False = false alarm
    verify_note: str = ""
    extra: dict = field(default_factory=dict)
