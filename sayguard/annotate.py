"""Draw the dashboard overlay on a camera frame (ASCII only: OpenCV cannot render
CJK, so Chinese text is shown in the web UI instead)."""
from __future__ import annotations

import cv2
import numpy as np

from .privacy import anonymise
from .types import Detection

GREEN = (90, 200, 90)
RED = (40, 40, 235)
ORANGE = (0, 160, 255)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)

SHORT = {
    "no_helmet": "NO HARD HAT",
    "person_present": "RESTRICTED AREA",
    "person_count_over": "TOO MANY PEOPLE",
    "object_present": "OBJECT",
    "object_near_person": "TOO CLOSE",
}


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple, scale: float = 0.5) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def draw(frame: np.ndarray, cam_title: str, heads: list[Detection], objects: list[Detection],
         active: dict[str, list[tuple[str, float, str]]], active_types: dict[str, str],
         links: list[tuple[tuple, tuple, float]], fps: float, blur: bool = True,
         out_width: int = 640, blur_boxes: list[tuple[tuple, bool]] | None = None) -> np.ndarray:
    """blur_boxes: [(head xyxy, wearing_helmet)] to pixelate.  The pipeline passes the
    raw per-frame detections here, because the tracked `heads` omit boxes the tracker
    has not confirmed or matched, and those faces must be hidden too."""
    img = frame.copy()
    if blur:
        anonymise(img, blur_boxes if blur_boxes is not None else [(d.xyxy, d.label == "helmet") for d in heads])
    H, W = img.shape[:2]
    th = max(1, round(W / 640))
    for o in objects:
        x1, y1, x2, y2 = map(int, o.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, 2 * th)
        _label(img, f"{o.label} {o.conf:.2f}", x1, y1 - 2, ORANGE, 0.5 * th)
    for obj_box, body, dist in links:
        ox, oy = int((obj_box[0] + obj_box[2]) / 2), int((obj_box[1] + obj_box[3]) / 2)
        px, py = int((body[0] + body[2]) / 2), int((body[1] + body[3]) / 2)
        cv2.line(img, (ox, oy), (px, py), RED, 2 * th)
        _label(img, f"{dist:.1f} m", (ox + px) // 2, (oy + py) // 2, RED, 0.55 * th)
    for d in heads:
        x1, y1, x2, y2 = map(int, d.xyxy)
        key = f"t{d.track_id}"
        if key in active:
            name, dur, _ = max(active[key], key=lambda a: a[1])
            text = f"{SHORT.get(active_types.get(name, ''), name.upper()[:18])} {dur:.1f}s"
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 3 * th)
            _label(img, text, x1, y1 - 2, RED, 0.55 * th)
        else:
            color = GREEN if d.label == "helmet" else RED
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 * th)
            _label(img, "hard hat" if d.label == "helmet" else "no hat?", x1, y1 - 2, color, 0.45 * th)
    if "cam" in active:
        name, dur, _ = max(active["cam"], key=lambda a: a[1])
        text = f"{SHORT.get(active_types.get(name, ''), 'ALERT')}: {name[:40]} ({dur:.0f}s)".encode("ascii", "replace").decode()
        cv2.rectangle(img, (0, H - 34 * th), (W, H), RED, -1)
        cv2.putText(img, text, (8, H - 10 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * th, WHITE, 2, cv2.LINE_AA)
    out = cv2.resize(img, (out_width, int(H * out_width / W)), interpolation=cv2.INTER_AREA) if W != out_width else img
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), DARK, -1)
    cv2.putText(out, cam_title.encode("ascii", "replace").decode(), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    t = f"{fps:4.1f} fps"
    cv2.putText(out, t, (out.shape[1] - 78, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
    return out
