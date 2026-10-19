"""Site geometry: calibrated cameras, the floor plan, and zones measured in metres.

Each camera's calibration gives a homography between the floor (world z = 0, metres) and
its image.  A person's feet are the bottom centre of their box, so one inverse homography
turns every camera's detections into floor-plan positions in a single shared frame.  That
is what lets SayGuard treat ten views of one aisle as one place: distances in metres instead
of head-size estimates, zones drawn once on the plan instead of per camera, and one identity
per worker across cameras (sayguard.sitetrack).

Calibration format: NVIDIA PhysicalAI-SmartSpaces 2025/2026 ("calibration.json"): per sensor
an id, a 3x3 "homography" that maps world (x, y, 1) to image pixels at frameWidth x
frameHeight, and a floor-plan transform: plan pixel = (world + translationToGlobalCoordinates)
* scaleFactor, with the plan's y axis pointing down (so y is flipped against the image height).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


HEAD_TOP_M = 1.75   # height of the top of a standing worker's head, for floor points from box tops


@dataclass
class CameraCalib:
    id: str
    H: np.ndarray                    # world (x, y, 1) -> image pixel, at (width, height); sign normalised
    Hinv: np.ndarray
    width: int
    height: int
    position: tuple[float, float] | None = None      # camera location on the floor, metres
    P: np.ndarray | None = None      # 3x4 world (x, y, z, 1) -> image, if the calibration has it

    def __post_init__(self) -> None:
        # The homographies come normalised to H[2,2] = 1, so the sign of w for points in view
        # depends on where the world origin lies; flip it so in-view floor points have w > 0
        # (to_image's "behind the camera" test relies on it).
        p = self.Hinv @ np.array([self.width / 2, self.height - 1.0, 1.0])
        if p[2] < 0:
            self.Hinv = -self.Hinv
            self.H = -self.H
        self._Hh: dict[float, np.ndarray] = {}

    def _scale(self, u: float, v: float, src_wh: tuple[int, int] | None) -> tuple[float, float]:
        if src_wh is not None:
            u, v = u * self.width / src_wh[0], v * self.height / src_wh[1]
        return u, v

    def to_world(self, u: float, v: float, src_wh: tuple[int, int] | None = None) -> tuple[float, float]:
        """Floor position of image point (u, v); src_wh = size of the image the point is in
        (streams are downscaled from the calibrated resolution)."""
        u, v = self._scale(u, v, src_wh)
        p = self.Hinv @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_world_at(self, u: float, v: float, h: float, src_wh: tuple[int, int] | None = None) -> tuple[float, float] | None:
        """Floor position under the point at height h (metres) seen at image point (u, v): e.g.
        the box top of a standing worker at head height.  None without a 3x4 camera matrix."""
        if self.P is None:
            return None
        if h not in self._Hh:
            P = self.P
            self._Hh[h] = np.linalg.inv(np.stack([P[:, 0], P[:, 1], h * P[:, 2] + P[:, 3]], axis=1))
        u, v = self._scale(u, v, src_wh)
        p = self._Hh[h] @ np.array([u, v, 1.0])
        if abs(p[2]) < 1e-12:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_image(self, x: float, y: float, dst_wh: tuple[int, int] | None = None) -> tuple[float, float] | None:
        """Image point of floor position (x, y); None if it is behind the camera."""
        p = self.H @ np.array([x, y, 1.0])
        if p[2] <= 1e-9:
            return None
        u, v = p[0] / p[2], p[1] / p[2]
        if dst_wh is not None:
            u, v = u * dst_wh[0] / self.width, v * dst_wh[1] / self.height
        return float(u), float(v)


@dataclass
class Zone:
    id: str
    name: str
    polygon: np.ndarray              # (N, 2) floor coordinates in metres

    def contains(self, x: float, y: float) -> bool:
        """Even-odd point-in-polygon test."""
        poly, inside = self.polygon, False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
            j = i
        return inside


@dataclass
class Site:
    cameras: dict[str, CameraCalib]
    scale: float                     # plan pixels per metre
    translation: tuple[float, float]
    plan_path: Path | None = None
    plan_size: tuple[int, int] | None = None          # (width, height) of the plan image
    zones: list[Zone] = field(default_factory=list)

    @classmethod
    def load(cls, calibration: str | Path, plan: str | Path | None = None,
             zones: list[dict] | None = None) -> Site:
        data = json.loads(Path(calibration).read_text(encoding="utf-8"))
        cams: dict[str, CameraCalib] = {}
        scale, trans = 1.0, (0.0, 0.0)
        for s in data["sensors"]:
            if s.get("type", "camera") != "camera" or "homography" not in s:
                continue
            attrs = {a["name"]: a["value"] for a in s.get("attributes", [])}
            H = np.array(s["homography"], dtype=float)
            pos = s.get("coordinates")
            P = np.array(s["cameraMatrix"], dtype=float) if "cameraMatrix" in s else None
            cams[s["id"]] = CameraCalib(id=s["id"], H=H, Hinv=np.linalg.inv(H),
                                        width=int(float(attrs.get("frameWidth", 1920))),
                                        height=int(float(attrs.get("frameHeight", 1080))),
                                        position=(float(pos["x"]), float(pos["y"])) if pos else None,
                                        P=P if P is not None and P.shape == (3, 4) else None)
            if "scaleFactor" in s:
                scale = float(s["scaleFactor"])
                t = s.get("translationToGlobalCoordinates", {"x": 0.0, "y": 0.0})
                trans = (float(t["x"]), float(t["y"]))
        plan_size = None
        if plan is not None:
            img = read_image(plan)
            if img is None:
                raise FileNotFoundError(f"floor plan image not readable: {plan}")
            plan_size = (img.shape[1], img.shape[0])
        site = cls(cameras=cams, scale=scale, translation=trans,
                   plan_path=Path(plan) if plan else None, plan_size=plan_size)
        site.zones = [site.zone_from_config(z) for z in (zones or [])]
        return site

    # ---- floor plan <-> world ---------------------------------------------
    def to_plan(self, x: float, y: float) -> tuple[float, float]:
        px = (x + self.translation[0]) * self.scale
        py = (y + self.translation[1]) * self.scale
        return px, (self.plan_size[1] - py) if self.plan_size else -py

    def from_plan(self, px: float, py: float) -> tuple[float, float]:
        y_up = (self.plan_size[1] - py) if self.plan_size else -py
        return px / self.scale - self.translation[0], y_up / self.scale - self.translation[1]

    def zone_from_config(self, z: dict) -> Zone:
        """{id, name, polygon: [[x, y], ...] in metres} or {..., plan_polygon: [[px, py], ...]}."""
        if "polygon" in z:
            pts = np.array(z["polygon"], dtype=float)
        else:
            if self.plan_size is None:   # plan pixels mean nothing without the plan's height (y flip)
                raise ValueError(f"zone {z.get('id')!r} uses plan_polygon but no floor plan image is configured")
            pts = np.array([self.from_plan(px, py) for px, py in z["plan_polygon"]], dtype=float)
        return Zone(id=z["id"], name=z.get("name", z["id"]), polygon=pts)

    def zone_of(self, x: float, y: float) -> str | None:
        """First configured zone containing the point (zones are listed most specific first)."""
        return next((z.id for z in self.zones if z.contains(x, y)), None)

    def zones_of(self, x: float, y: float) -> list[str]:
        return [z.id for z in self.zones if z.contains(x, y)]


def read_image(path: str | Path):
    """cv2.imread that also works for non-ASCII paths on Windows (cv2.imread returns None there)."""
    import cv2
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def foot_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom centre of a person (or vehicle) box: where it touches the floor."""
    return (box[0] + box[2]) / 2, box[3]


def touches_bottom(box: tuple[float, float, float, float], frame_h: int, margin: float = 0.01) -> bool:
    """A box cut off by the bottom edge has no visible feet: its foot point is not on the floor."""
    return box[3] >= frame_h * (1 - margin)
