"""Per-camera tracking of heads (helmet or bare), plus a short label history per
track so one misclassified frame does not raise an alert.

A small greedy tracker (numpy only) replaces supervision's ByteTrack, which
supervision 0.31 removed (it moved to the separate `trackers` package). Heads at
~10 fps need little more than overlap matching, and owning the code lets the
matching fit this job:
- class-agnostic: the detector's helmet/head label for one head can flip between
  frames; that flip is exactly what the label history measures, so it must not
  split the track;
- overlapping helmet+head boxes on one head are merged first (NMS at MERGE_IOU,
  higher score wins), otherwise one worker becomes two tracks, one of them a
  false no-helmet;
- best IoU first (>= MATCH_IOU), then a centre-distance fallback for small heads
  that move more than their own width between frames; a damped constant-velocity
  prediction keeps walking heads overlapping;
- a confident detection starts a track at once (no activation delay, so a
  worker's first frame is already tracked); low-confidence boxes only extend
  existing tracks, so detector flicker does not mint new ids;
- a track unmatched for LOST_S seconds of camera time is dropped. Ids only
  increase, so a dropped track's history can never attach to a new person.
With at most a few dozen heads per camera, greedy matching costs microseconds."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .types import Detection, Person

HISTORY = 15          # observations kept per track
FORGET_S = 10.0       # label history kept this long after a track was last seen
LOST_S = 2.0          # a track unmatched this long (camera time) is dropped
MERGE_IOU = 0.6       # class-agnostic NMS: helmet+head boxes on one head become one
MATCH_IOU = 0.25      # minimum overlap for the primary (IoU) match
CENTRE_FRAC = 0.8     # fallback match: centre distance < CENTRE_FRAC x box diagonal
MAX_SCALE = 2.0       # fallback match only between boxes of similar height
LOW_CONF = 0.35       # below this a detection may extend a track but never start one
VEL_ALPHA = 0.5       # weight of the newest displacement in the velocity estimate


def _centres(b: np.ndarray) -> np.ndarray:
    return (b[:, :2] + b[:, 2:]) / 2


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) xyxy arrays."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _nms(dets: list[Detection], iou: float, contain: float | None = None) -> list[Detection]:
    """Class-agnostic NMS: of boxes overlapping by >= iou, the highest score (and
    its label) survives.  contain: also suppress a box whose overlap covers this share of
    the smaller box (a half-body box inside a full-body box of the same person)."""
    order = sorted(dets, key=lambda d: -d.conf)
    if len(order) < 2:
        return order
    boxes = np.array([d.xyxy for d in order], dtype=float)
    m = _iou(boxes, boxes)
    if contain is not None:
        lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
        rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
        inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
        area = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0, None), axis=1)
        cover = inter / np.maximum(np.minimum(area[:, None], area[None, :]), 1e-9)
        m = np.where(cover >= contain, 1.0, m)
    keep: list[int] = []
    for i in range(len(order)):
        if all(m[i, j] < iou for j in keep):
            keep.append(i)
    return [order[i] for i in keep]


def _take(cands: list[tuple[float, int, int]], pairs: list[tuple[int, int]],
          used_t: set[int], used_d: set[int]) -> None:
    """Accept (cost, track, det) candidates cheapest first, each side at most once."""
    for _, t, d in sorted(cands):
        if t not in used_t and d not in used_d:
            pairs.append((t, d))
            used_t.add(t)
            used_d.add(d)


def _associate(pred: np.ndarray, boxes: np.ndarray) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of predicted track boxes to detections: highest
    IoU first, then, for what is left, nearest centre relative to box size."""
    if not len(pred) or not len(boxes):
        return []
    pairs: list[tuple[int, int]] = []
    used_t: set[int] = set()
    used_d: set[int] = set()
    iou = _iou(pred, boxes)
    _take([(-float(iou[t, d]), int(t), int(d)) for t, d in zip(*np.nonzero(iou >= MATCH_IOU))],
          pairs, used_t, used_d)

    dist = np.linalg.norm(_centres(pred)[:, None] - _centres(boxes)[None], axis=2)
    diag_p = np.linalg.norm(pred[:, 2:] - pred[:, :2], axis=1)
    diag_b = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)
    rel = dist / np.maximum((diag_p[:, None] + diag_b[None]) / 2, 1e-6)
    hp = np.maximum(pred[:, 3] - pred[:, 1], 1e-6)[:, None]
    hb = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)[None]
    similar = np.maximum(hp, hb) / np.minimum(hp, hb) <= MAX_SCALE
    _take([(float(rel[t, d]), int(t), int(d)) for t, d in zip(*np.nonzero((rel < CENTRE_FRAC) & similar))],
          pairs, used_t, used_d)
    return pairs


@dataclass
class _Track:
    tid: int
    box: np.ndarray                     # last matched xyxy
    last_ts: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))   # centre motion, px/s

    def predict(self, ts: float, max_dt: float) -> np.ndarray:
        dt = min(max(0.0, ts - self.last_ts), max_dt)
        return self.box + np.tile(self.vel * dt, 2)

    def update(self, box: np.ndarray, ts: float) -> None:
        dt = ts - self.last_ts
        if dt > 0:
            step = (_centres(box[None]) - _centres(self.box[None]))[0] / dt
            self.vel = VEL_ALPHA * step + (1 - VEL_ALPHA) * self.vel
        self.box, self.last_ts = box, ts


class CameraTracker:
    def __init__(self, fps: float = 10.0, lost_s: float = LOST_S, low_conf: float = LOW_CONF,
                 labels: tuple[str, ...] = ("helmet", "head"), merge_iou: float = MERGE_IOU,
                 contain: float | None = None):
        """labels: the detector classes that are one tracked subject (site mode tracks 'person'
        boxes; the label history then just records that class).  merge_iou / contain: how
        overlapping boxes are merged before tracking (body boxes overlap less than the helmet
        and head boxes of one head, and a partly hidden person can get a half-body box inside
        the full one: two tracks of one person that the floor plan would count twice)."""
        self.lost_s = lost_s
        self.low_conf = low_conf
        self.labels = labels
        self.merge_iou, self.contain = merge_iou, contain
        # extrapolate motion at most ~3 frame intervals; beyond that a stopped
        # worker would be predicted out of reach of their own next detection
        self.max_predict_s = max(0.3, 3.0 / max(fps, 1e-3))
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.hist: dict[int, deque] = {}
        self.seen: dict[int, float] = {}

    def update(self, dets: list[Detection], ts: float) -> tuple[list[Person], list[Detection]]:
        ppe = _nms([d for d in dets if d.label in self.labels], self.merge_iou, self.contain)
        self.tracks = [t for t in self.tracks if ts - t.last_ts <= self.lost_s]
        matched = self._match(ppe, ts)

        people: list[Person] = []
        boxes: list[Detection] = []
        for tr, d in sorted(matched, key=lambda m: m[0].tid):
            bare = d.label == "head"
            h = self.hist.setdefault(tr.tid, deque(maxlen=HISTORY))
            h.append(bare)
            self.seen[tr.tid] = ts
            box = tuple(float(v) for v in d.xyxy)
            people.append(Person(track_id=tr.tid, head=box, helmet_now=not bare,
                                 no_helmet_ratio=sum(h) / len(h), observations=len(h)))
            boxes.append(Detection(box, d.label, float(d.conf), track_id=tr.tid))
        forget = max(FORGET_S, self.lost_s)
        for tid in [t for t, s in self.seen.items() if ts - s > forget]:
            self.seen.pop(tid, None)
            self.hist.pop(tid, None)
        return people, boxes

    def _match(self, ppe: list[Detection], ts: float) -> list[tuple[_Track, Detection]]:
        """Confident detections first (unmatched ones start tracks), then
        low-confidence ones against the tracks still free (unmatched ones dropped)."""
        out: list[tuple[_Track, Detection]] = []
        free = list(self.tracks)
        high = [d for d in ppe if d.conf >= self.low_conf]
        low = [d for d in ppe if d.conf < self.low_conf]
        for group, may_start in ((high, True), (low, False)):
            if not group:
                continue
            pred = np.array([t.predict(ts, self.max_predict_s) for t in free], dtype=float).reshape(-1, 4)
            boxes = np.array([d.xyxy for d in group], dtype=float).reshape(-1, 4)
            pairs = _associate(pred, boxes)
            for ti, di in pairs:
                free[ti].update(boxes[di], ts)
                out.append((free[ti], group[di]))
            if may_start:
                taken = {di for _, di in pairs}
                for di, d in enumerate(group):
                    if di not in taken:
                        tr = _Track(self.next_id, boxes[di], ts)
                        self.next_id += 1
                        self.tracks.append(tr)
                        out.append((tr, d))
            used = {ti for ti, _ in pairs}
            free = [t for i, t in enumerate(free) if i not in used]
        return out
