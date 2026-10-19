"""Face anonymisation for everything that leaves the inference loop (dashboard
tiles and saved snapshots).  It reuses the head boxes the PPE model already
produces, so it costs no extra network.  A bare-head box frames the whole head,
so the face is its lower part.  A hard-hat box frames the hat only (median
height/width 0.73 on the demo clips), so the face is BELOW the box: about one
hat-width down to the chin, and wider than the hat when the head is turned."""
from __future__ import annotations

import cv2
import numpy as np


# hard hat: from the brim down to the chin (FACE_BELOW_HAT hat-widths below the box), HAT_SIDE wider
# on each side; erring large only pixelates some neck or collar, erring small shows a face
FACE_BELOW_HAT = 1.0
HAT_SIDE = 0.18


def face_region(head: tuple, helmet: bool, W: int, H: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = head
    w, h = x2 - x1, y2 - y1
    if helmet:
        box = (x1 - HAT_SIDE * w, y1 + 0.38 * h, x2 + HAT_SIDE * w, y2 + FACE_BELOW_HAT * w)
    else:
        box = (x1 - 0.06 * w, y1 + 0.12 * h, x2 + 0.06 * w, y2 + 0.12 * h)
    return (int(max(0, box[0])), int(max(0, box[1])), int(min(W, box[2])), int(min(H, box[3])))


def pixelate(img: np.ndarray, box: tuple[int, int, int, int], blocks: int = 6) -> None:
    x1, y1, x2, y2 = box
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


# a detector misses a head now and then for a few frames (0.1-0.5 s on the demo clips); a face must
# not show in those frames, so each camera keeps masking where a face was for HOLD_S after its last
# detection, the box grown HOLD_GROW on each side because the head may have moved meanwhile
HOLD_S = 1.0
HOLD_GROW = 0.15
_SAME_IOU = 0.3


def _iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _grow(b: tuple, f: float) -> tuple:
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - f * w, b[1] - f * h, b[2] + f * w, b[3] + f * h)


class FaceHold:
    """One camera's recent face boxes: update() returns this frame's boxes plus the ones seen in the
    last HOLD_S that no current box covers, so a detector miss of a few frames shows no face."""

    def __init__(self, hold_s: float = HOLD_S, grow: float = HOLD_GROW):
        self.hold_s, self.grow = hold_s, grow
        self._seen: list[tuple[float, tuple, bool]] = []     # (last seen, box, wearing_helmet)

    def update(self, ts: float, boxes: list[tuple[tuple, bool]]) -> list[tuple[tuple, bool]]:
        current = [(tuple(float(v) for v in b), bool(h)) for b, h in boxes]
        if self._seen and ts < max(t for t, _, _ in self._seen):
            self._seen = []                                   # the clock went back (a restarted source)
        held = [(t, b, h) for t, b, h in self._seen
                if ts - t <= self.hold_s and not any(_iou(b, c) >= _SAME_IOU for c, _ in current)]
        self._seen = [(ts, b, h) for b, h in current] + held
        return current + [(_grow(b, self.grow), h) for _, b, h in held]


def anonymise(img: np.ndarray, heads: list[tuple[tuple, bool]]) -> np.ndarray:
    """heads: [(xyxy, wearing_helmet)] in img coordinates. Modifies img in place."""
    H, W = img.shape[:2]
    for box, helmet in heads:
        pixelate(img, face_region(box, helmet, W, H))
    return img
