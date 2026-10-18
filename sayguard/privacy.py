"""Face anonymisation for everything that leaves the inference loop (dashboard
tiles and saved snapshots).  It reuses the head boxes the PPE model already
produces, so it costs no extra network: the face is the lower part of the head
box (below the brim when a hard hat is worn)."""
from __future__ import annotations

import cv2
import numpy as np


def face_region(head: tuple, helmet: bool, W: int, H: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = head
    w, h = x2 - x1, y2 - y1
    top = y1 + (0.38 if helmet else 0.12) * h
    return (int(max(0, x1 - 0.06 * w)), int(max(0, top)), int(min(W, x2 + 0.06 * w)), int(min(H, y2 + 0.12 * h)))


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
