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


def anonymise(img: np.ndarray, heads: list[tuple[tuple, bool]]) -> np.ndarray:
    """heads: [(xyxy, wearing_helmet)] in img coordinates. Modifies img in place."""
    H, W = img.shape[:2]
    for box, helmet in heads:
        pixelate(img, face_region(box, helmet, W, H))
    return img
