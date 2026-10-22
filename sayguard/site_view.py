"""Site-mode drawing: camera tiles (a worker in an active alert is red in every camera that sees
them), and the live floor plan with workers, vehicles, zones and active alerts.  Worker ids are
not drawn: fused across cameras they are reliable at any one moment, but over minutes one person
still goes through several (IDF1 about 0.5 on Warehouse_010), so a number on screen would mislead.
ASCII only, like sayguard.annotate (OpenCV cannot render CJK)."""
from __future__ import annotations

import cv2
import numpy as np

from .annotate import DARK, ORANGE, RED, SHORT, WHITE, _label
from .privacy import anonymise
from .site import Site, read_image

ZONE_FILL = (70, 160, 70)
WORKER = (90, 200, 90)


def head_of(box: tuple) -> tuple[float, float, float, float]:
    """Head region of a person box, centred at the top, for face pixelation.  For a standing person
    the head is the top ~16 % of the box; a box cut off by the frame edge or a shelf, or a crouching
    worker, is squat, and there the head is sized from the box width (about 0.6 of it) instead, so
    the mouth and chin are covered too.  Erring large only hides some neck or shoulder."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    hw = min(w, max(0.42 * w + 0.06 * h, 0.55 * w))
    hh = min(h, max(0.16 * h, 0.6 * w))
    cx = (x1 + x2) / 2
    return cx - hw / 2, y1, cx + hw / 2, y1 + hh


def draw_site_tile(frame: np.ndarray, title: str, people: list[tuple[tuple, int | None, float]],
                   vehicles: list[tuple[tuple, str]], active: dict[str, list[tuple[str, float, str]]],
                   active_types: dict[str, str], fps: float, blur: bool = True, out_width: int = 640,
                   blur_boxes: list[tuple] | None = None) -> np.ndarray:
    """people: [(box, gid, conf)]; vehicles: [(box, label)]; active: subject key -> [(rule, s, severity)].
    blur_boxes: every person box the detector returned (the tracker leaves some out, and those
    faces must be hidden too); defaults to the tracked people."""
    img = frame.copy()
    if blur:
        boxes = blur_boxes if blur_boxes is not None else [b for b, _, _ in people]
        anonymise(img, [(head_of(b), False) for b in boxes])
    H, W = img.shape[:2]
    th = max(1, round(W / 640))
    for box, label in vehicles:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, 2 * th)
        _label(img, label.replace("_", " "), x1, y1 - 2, ORANGE, 0.5 * th)
    for box, gid, _ in people:
        x1, y1, x2, y2 = map(int, box)
        key = f"g{gid}"
        if gid is not None and key in active:
            name, dur, _ = max(active[key], key=lambda a: a[1])
            text = f"{SHORT.get(active_types.get(name, ''), name.upper()[:16])} {dur:.1f}s"
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 3 * th)
            _label(img, text.encode("ascii", "replace").decode(), x1, y1 - 2, RED, 0.55 * th)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), WORKER, 2 * th)
    out = cv2.resize(img, (out_width, int(H * out_width / W)), interpolation=cv2.INTER_AREA) if W != out_width else img
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), DARK, -1)
    cv2.putText(out, title.encode("ascii", "replace").decode(), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(out, f"{fps:4.1f} fps", (out.shape[1] - 78, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1,
                cv2.LINE_AA)
    return out


class PlanRenderer:
    """Draws the live floor plan.  The static layer (plan image, zones, cameras) is built once."""

    def __init__(self, site: Site, out_width: int = 960):
        self.site = site
        base = read_image(site.plan_path) if site.plan_path else None
        if base is None:
            base = np.full((720, 1280, 3), 30, np.uint8)
        # crop to the area the cameras and zones cover, so the warehouse fills the tile
        pts = [site.to_plan(*c.position) for c in site.cameras.values() if c.position]
        for z in site.zones:
            pts += [site.to_plan(x, y) for x, y in z.polygon]
        if pts:
            p = np.array(pts)
            pad = 40
            x0, y0 = max(0, int(p[:, 0].min()) - pad), max(0, int(p[:, 1].min()) - pad)
            x1, y1 = min(base.shape[1], int(p[:, 0].max()) + pad), min(base.shape[0], int(p[:, 1].max()) + pad)
        else:
            x0, y0, x1, y1 = 0, 0, base.shape[1], base.shape[0]
        self.offset = (x0, y0)
        self.scale = out_width / max(1, x1 - x0)
        crop = base[y0:y1, x0:x1]
        static = cv2.resize(crop, (out_width, int(crop.shape[0] * self.scale)), interpolation=cv2.INTER_AREA)
        overlay = static.copy()
        for z in site.zones:
            poly = np.array([self.px(x, y) for x, y in z.polygon], np.int32)
            cv2.fillPoly(overlay, [poly], ZONE_FILL)
        static = cv2.addWeighted(overlay, 0.25, static, 0.75, 0)
        for z in site.zones:
            poly = np.array([self.px(x, y) for x, y in z.polygon], np.int32)
            cv2.polylines(static, [poly], True, (110, 220, 110), 2, cv2.LINE_AA)
            cx, cy = poly.mean(axis=0).astype(int)
            _label(static, z.name.encode("ascii", "replace").decode()[:24], int(cx) - 30, int(cy), (40, 110, 40), 0.45)
        for cam in site.cameras.values():
            if cam.position:
                x, y = self.px(*cam.position)
                cv2.drawMarker(static, (x, y), (200, 200, 200), cv2.MARKER_TRIANGLE_UP, 12, 2)
                cv2.putText(static, cam.id[-2:], (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
                            cv2.LINE_AA)
        self.static = static

    def px(self, x: float, y: float) -> tuple[int, int]:
        u, v = self.site.to_plan(x, y)
        return int((u - self.offset[0]) * self.scale), int((v - self.offset[1]) * self.scale)

    def render(self, workers: list[tuple[int, tuple[float, float], int]], vehicles: list[tuple[str, tuple[float, float]]],
               links: list[tuple[int, tuple[float, float], float]], active_gids: set[int], title: str) -> np.ndarray:
        """workers: [(gid, xy, n_cameras)]; vehicles: [(label, xy)]; links: [(gid, vehicle xy, metres)]."""
        img = self.static.copy()
        pos = {gid: xy for gid, xy, _ in workers}
        for gid, vxy, d in links:
            if gid in pos:
                a, b = self.px(*pos[gid]), self.px(*vxy)
                cv2.line(img, a, b, RED, 2, cv2.LINE_AA)
                _label(img, f"{d:.1f} m", (a[0] + b[0]) // 2, (a[1] + b[1]) // 2, RED, 0.45)
        for label, xy in vehicles:
            x, y = self.px(*xy)
            cv2.rectangle(img, (x - 8, y - 8), (x + 8, y + 8), ORANGE, -1)
            cv2.putText(img, label.split("_")[0][:9], (x + 10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ORANGE, 1,
                        cv2.LINE_AA)
        for gid, xy, ncam in workers:
            x, y = self.px(*xy)
            c = RED if gid in active_gids else WORKER
            cv2.circle(img, (x, y), 9 if gid in active_gids else 7, c, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 9 if gid in active_gids else 7, WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), DARK, -1)
        cv2.putText(img, title.encode("ascii", "replace").decode(), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
                    cv2.LINE_AA)
        return img
