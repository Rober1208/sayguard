"""One identity per worker across cameras, on the floor plan.

Every camera keeps its own tracks (sayguard.tracking).  Projected onto the floor
(sayguard.site), the views of one worker land close together, so the site-level identity
is geometric first.  One complication shapes everything below: a box's bottom edge is the
worker's feet only when the feet are visible.  A box cut by a shelf, a cart or the frame
edge projects its bottom far beyond the worker along the camera ray (on Warehouse_010 ground
truth, 15 % of views land more than 1.5 m off).  So each camera track carries the floor
segment it can lie on, from the point under the box top at head height to the point under
the box bottom; the worker is somewhere on it, and its feet end only when the feet are seen.

- a worker's position is a robust triangulation: start from the median of the views whose
  feet are visible, move each view to the closest point of its own segment, take the median
  again; one view with hidden feet can no longer drag a worker metres away;
- a camera track keeps the worker it was given (sticky) unless another track of the same
  camera already represents that worker this tick, or it disagrees with the other cameras by
  more than DETACH_M for DETACH_TICKS ticks (a wrong join, or two people who walked together
  and parted; with two members the later link leaves, then the worse view);
- a new camera track joins the nearest worker within MATCH_M of its segment that no other
  track of the same camera represents, so a track-id switch, or a worker walking into the
  next camera's view, keeps the identity;
- leftover tracks start new workers, pairing each with the nearest track of every other
  camera (nearest first, one track per camera); a track no other camera confirms must
  persist SPAWN_TICKS ticks first (SPAWN_TICKS_FAR when its position is uncertain), so a
  flickering false detection does not become a worker;
- every view carries sigma, its floor uncertainty: the floor distance one pixel at its feet
  spans, times the box's localisation error.  Match radii grow with it and the position is a
  weighted median, so a 40-pixel worker at the far end of a camera's view neither drags a
  well-seen worker nor spawns a duplicate of them;
- two workers within MERGE_M with no camera in common for MERGE_TICKS ticks are one person,
  unless a camera recently saw both at once as two tracks standing apart (more than COSEEN_M
  for COSEEN_TICKS ticks, within COSEEN_MEMORY_S: two people who only passed each other; two
  boxes on one person sit at one spot and are no such evidence) - then it takes
  MERGE_TICKS_LONG ticks; the older id survives;
- a worker nobody sees keeps its id for LOST_S: the next track to appear within MATCH_M plus
  REJOIN_SPEED per second hidden (at most REJOIN_MAX) takes it, so walking behind a rack
  does not make a new person; after LOST_S it is dropped; ids are never reused.

Vehicles need no identity for the rules, only "is one near this worker now": each tick
their ground contacts (both bottom corners and the bottom centre of every box) are clustered
across cameras.  No appearance model is involved; the Hailo-10H person re-id networks
(osnet_x1_0, repvgg_a0) are the planned tie-breaker for cameras whose views do not overlap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MATCH_M = 1.2
NEW_CLUSTER_M = 1.0
DETACH_M = 2.0
DETACH_TICKS = 5
MERGE_M = 0.7
MERGE_TICKS = 5
COSEEN_M = 1.2         # two tracks of one camera this far apart on the floor are two people ...
COSEEN_TICKS = 3       # ... once it holds this many ticks (two boxes on one person sit at one spot);
COSEEN_MEMORY_S = 5.0  # that evidence is forgotten after this long without a new co-observation,
MERGE_TICKS_LONG = 20  # and closer than MERGE_M this many ticks merges anyway (passing takes less)
LOST_S = 6.0           # a worker nobody sees is kept this long (hidden behind a rack) ...
REJOIN_SPEED = 1.0     # ... and may be picked up again this much farther away per second hidden,
REJOIN_MAX = 3.0       # up to this much (walking pace, damped: people stop behind racks too)
SMOOTH = 0.6           # weight of this tick's consensus in the worker position
SIGMA_CAP = 2.0        # most a view's uncertainty may widen a match radius (metres)
SPAWN_TICKS = 2        # a track seen by one camera only becomes a worker after this many ticks ...
SPAWN_TICKS_FAR = 4    # ... or this many when its floor position is this uncertain:
FAR_SIGMA = 1.5        # (a few pixels at the far end of a camera's view are metres on the floor)

VEHICLE_CLUSTER_M = 2.5
# subtracted from the distance to a vehicle's nearest ground contact (the near side of its box);
# forklift: chosen so measured worker-forklift distances are unbiased on the validation scene
# (configs/eval/warehouse_020.yaml: -0.38 m with 0.5); the others are rough guesses, not measured
VEHICLE_PAD_M = {"forklift": 0.1, "pallet_truck": 0.35, "transporter": 0.4, "mobile_robot": 0.3}


def _seg_point(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closest point to p on segment ab."""
    ab = b - a
    n = float(ab @ ab)
    t = 0.0 if n < 1e-12 else min(1.0, max(0.0, float((p - a) @ ab) / n))
    return a + t * ab


@dataclass
class Obs:
    """One camera track in one tick.  xy: floor point under the box bottom (the feet when
    they are visible).  seg: (head-height point, feet point), the floor segment the worker
    can be on; None when only xy is known."""
    cam: str
    track_id: int
    xy: tuple[float, float]
    box: tuple[float, float, float, float]        # in the camera's frame pixels
    conf: float = 1.0
    feet_visible: bool = True
    seg: tuple[tuple[float, float], tuple[float, float]] | None = None
    sigma: float = 0.5                            # floor uncertainty of xy, metres

    def near(self, p: np.ndarray) -> np.ndarray:
        """This view's closest possible position to p."""
        if self.seg is None:
            return np.asarray(self.xy, dtype=float)
        return _seg_point(np.asarray(p, dtype=float), np.asarray(self.seg[0], float), np.asarray(self.seg[1], float))

    def dist(self, p: np.ndarray) -> float:
        return float(np.hypot(*(self.near(p) - np.asarray(p, dtype=float))))


@dataclass
class Worker:
    gid: int
    xy: np.ndarray
    first_ts: float
    last_ts: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    members: dict[tuple[str, int], Obs] = field(default_factory=dict)   # this tick's camera tracks

    @property
    def cams(self) -> list[str]:
        return sorted({k[0] for k in self.members})

    def best_view(self) -> Obs | None:
        """The camera track to show for this worker: tallest box with visible feet."""
        obs = list(self.members.values())
        if not obs:
            return None
        return max(obs, key=lambda o: (o.feet_visible, o.box[3] - o.box[1]))


@dataclass
class Vehicle:
    label: str
    xy: np.ndarray                                # cluster centre
    points: list[tuple[str, tuple[float, float], tuple[float, float, float, float]]]   # (cam, xy, box)
    pad: float = 0.0

    @property
    def cams(self) -> list[str]:
        return sorted({p[0] for p in self.points})

    def distance_to(self, xy: np.ndarray) -> float:
        """Distance to the nearest ground contact any camera saw, minus the vehicle's pad (the
        contacts are its near side, but the body can reach past them)."""
        d = min(np.hypot(*(np.asarray(p) - np.asarray(xy))) for _, p, _ in self.points)
        return float(max(0.0, d - self.pad))


def _wmedian(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-axis weighted median."""
    out = np.empty(points.shape[1])
    for ax in range(points.shape[1]):
        order = np.argsort(points[:, ax])
        cw = np.cumsum(weights[order])
        half = cw[-1] / 2
        i = int(np.searchsorted(cw, half))
        v = points[order[i], ax]
        if i + 1 < len(order) and abs(cw[i] - half) <= 1e-9 * cw[-1]:   # exactly between two: average
            v = (v + points[order[i + 1], ax]) / 2
        out[ax] = v
    return out


def _consensus(obs: list[Obs], prior: np.ndarray | None = None) -> np.ndarray:
    """Robust floor position of one worker from its views (see the module docstring)."""
    good = [o for o in obs if o.feet_visible] or obs
    w = lambda os: np.array([1.0 / max(o.sigma, 0.2) ** 2 for o in os])     # noqa: E731
    ref = (_wmedian(np.array([o.xy for o in good], dtype=float), w(good)) if prior is None
           else np.asarray(prior, float))
    usable = [o for o in obs if o.feet_visible or o.seg is not None] or obs   # a hidden-feet point alone is biased
    for _ in range(2):
        ref = _wmedian(np.array([o.near(ref) for o in usable]), w(usable))
    return ref


class SiteTracker:
    def __init__(self, lost_s: float = LOST_S):
        self.lost_s = lost_s
        self.reset()

    def reset(self, keep_ids: bool = False) -> None:
        """Forget everyone (e.g. the demo clips looped back to their start).  keep_ids: new
        workers continue the numbering, so an id never names two different people."""
        nxt = getattr(self, "next_gid", 1) if keep_ids else 1
        self.workers: dict[int, Worker] = {}
        self.assign: dict[tuple[str, int], int] = {}
        self.assign_seen: dict[tuple[str, int], float] = {}
        self.linked_at: dict[tuple[str, int], float] = {}
        self._far: dict[tuple[str, int], int] = {}
        self._close: dict[tuple[int, int], int] = {}
        self._pending: dict[tuple[str, int], int] = {}  # unconfirmed single-camera tracks: ticks seen
        self._coseen: dict[tuple[int, int], float] = {}   # worker pair -> last time one camera saw two people
        self._coseen_n: dict[tuple[int, int], int] = {}
        self.next_gid = nxt

    # ------------------------------------------------------------------ update
    def update(self, obs: list[Obs], ts: float) -> list[Worker]:
        """Feed this tick's camera tracks; returns the workers seen this tick."""
        for w in self.workers.values():
            w.members = {}
        by_key = {(o.cam, o.track_id): o for o in obs}
        # keep only links to camera tracks that still exist (their ids are never reused)
        for k in [k for k, t in self.assign_seen.items() if k not in by_key and ts - t > self.lost_s]:
            self._unlink(k)

        unassigned: list[Obs] = []
        for k, o in by_key.items():
            gid = self.assign.get(k)
            w = self.workers.get(gid) if gid is not None else None
            if w is None:
                unassigned.append(o)
                continue
            clash = next((kk for kk in w.members if kk[0] == o.cam), None)
            if clash is None:
                w.members[k] = o
                self.assign_seen[k] = ts
                continue
            # two tracks of one camera claim this worker (e.g. after a merge remapped stale
            # links): the one nearer the worker's predicted position keeps it
            ref = self._predicted(w, ts)
            keep, lose = (k, clash) if o.dist(ref) < w.members[clash].dist(ref) else (clash, k)
            if keep == k:
                unassigned.append(w.members.pop(clash))
                w.members[k] = o
                self.assign_seen[k] = ts
            else:
                unassigned.append(o)
            self._unlink(lose, keep_seen=True)

        unassigned += self._detach()
        unassigned = self._join(unassigned, ts)
        self._spawn(unassigned, ts)
        self._note_coseen(ts)
        self._move(ts)
        self._merge(ts)
        for gid in [g for g, w in self.workers.items() if ts - w.last_ts > self.lost_s]:
            del self.workers[gid]
        return [w for w in self.workers.values() if w.members]

    def _unlink(self, k: tuple[str, int], keep_seen: bool = False) -> None:
        self.assign.pop(k, None)
        self.linked_at.pop(k, None)
        self._far.pop(k, None)
        if not keep_seen:
            self.assign_seen.pop(k, None)

    def _detach(self) -> list[Obs]:
        """Members that keep disagreeing with the rest of their worker leave it."""
        out = []
        for w in self.workers.values():
            if len(w.members) < 2:
                for k in w.members:
                    self._far.pop(k, None)
                continue
            for k, o in list(w.members.items()):
                others = [m for kk, m in w.members.items() if kk != k]
                d = o.dist(_consensus(others))
                self._far[k] = self._far.get(k, 0) + 1 if d > DETACH_M + min(o.sigma, SIGMA_CAP) else 0
            due = [k for k in w.members if self._far.get(k, 0) >= DETACH_TICKS]
            if len(w.members) == 2 and len(due) == 2:
                due = [max(due, key=lambda k: (self.linked_at.get(k, 0.0), not w.members[k].feet_visible,
                                               -(w.members[k].box[3] - w.members[k].box[1])))]
            for k in due:
                out.append(w.members.pop(k))
                self._unlink(k, keep_seen=True)
        return out

    def _predicted(self, w: Worker, ts: float) -> np.ndarray:
        return w.xy + w.vel * min(max(0.0, ts - w.last_ts), 1.0)

    def _join(self, unassigned: list[Obs], ts: float) -> list[Obs]:
        """Greedy: closest (track, worker) pairs first, one camera track per worker per camera."""
        if not unassigned or not self.workers:
            return unassigned
        refs = {gid: (_consensus(list(w.members.values())) if w.members else self._predicted(w, ts))
                for gid, w in self.workers.items()}
        # a worker hidden for a while may reappear farther from where it was last seen
        grow = {gid: 0.0 if w.members else min(REJOIN_SPEED * max(0.0, ts - w.last_ts), REJOIN_MAX)
                for gid, w in self.workers.items()}
        cands = []
        for i, o in enumerate(unassigned):
            for gid, w in self.workers.items():
                if any(k[0] == o.cam for k in w.members):
                    continue
                d = o.dist(refs[gid])
                if d < MATCH_M + min(o.sigma, SIGMA_CAP) + grow[gid]:
                    cands.append((d - grow[gid], i, gid))
        taken: set[int] = set()
        cam_taken: set[tuple[int, str]] = set()
        for _, i, gid in sorted(cands):
            o = unassigned[i]
            if i in taken or (gid, o.cam) in cam_taken:
                continue
            self._link(o, gid, ts)
            taken.add(i)
            cam_taken.add((gid, o.cam))
        return [o for i, o in enumerate(unassigned) if i not in taken]

    def _spawn(self, rest: list[Obs], ts: float) -> None:
        """New workers from leftover tracks: nearest cross-camera pairs first, one track per camera;
        a cluster only one camera sees waits until it has persisted (see the module docstring)."""
        keys = {(o.cam, o.track_id) for o in rest}
        for k in [k for k in self._pending if k not in keys]:
            del self._pending[k]
        for k in keys:
            self._pending[k] = self._pending.get(k, 0) + 1
        if not rest:
            return
        group_of = list(range(len(rest)))            # union of clusters by index
        members: dict[int, set[int]] = {i: {i} for i in range(len(rest))}
        pairs = []
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                if rest[i].cam == rest[j].cam:
                    continue
                a, b = rest[i], rest[j]
                p = a.near(np.asarray(b.xy, float))
                q = b.near(p)
                p = a.near(q)
                d = float(np.hypot(*(p - q)))
                if d < NEW_CLUSTER_M + min(a.sigma + b.sigma, SIGMA_CAP):
                    pairs.append((d, i, j))
        for _, i, j in sorted(pairs):
            gi, gj = group_of[i], group_of[j]
            if gi == gj:
                continue
            if {rest[x].cam for x in members[gi]} & {rest[x].cam for x in members[gj]}:
                continue                              # would put two tracks of one camera together
            members[gi] |= members.pop(gj)
            for x in members[gi]:
                group_of[x] = gi
        for idx in members.values():
            group = [rest[x] for x in sorted(idx)]
            if len({o.cam for o in group}) < 2:
                o = group[0]
                need = SPAWN_TICKS_FAR if o.sigma > FAR_SIGMA else SPAWN_TICKS
                if self._pending.get((o.cam, o.track_id), 0) < need:
                    continue                          # not confirmed yet: no worker this tick
            for o in group:
                self._pending.pop((o.cam, o.track_id), None)
            gid = self.next_gid
            self.next_gid += 1
            self.workers[gid] = Worker(gid=gid, xy=_consensus(group), first_ts=ts, last_ts=ts)
            for o in group:
                self._link(o, gid, ts)

    def _link(self, o: Obs, gid: int, ts: float) -> None:
        k = (o.cam, o.track_id)
        self.assign[k] = gid
        self.assign_seen[k] = ts
        self.linked_at[k] = ts
        self._far.pop(k, None)
        self.workers[gid].members[k] = o

    def _note_coseen(self, ts: float) -> None:
        """Workers that one camera keeps seeing as two tracks standing apart are separate people."""
        by_cam: dict[str, list[tuple[int, Obs]]] = {}
        for gid, w in self.workers.items():
            for (cam, _), o in w.members.items():
                by_cam.setdefault(cam, []).append((gid, o))
        now: set[tuple[int, int]] = set()
        for entries in by_cam.values():
            for i, (a, oa) in enumerate(entries):
                for b, ob in entries[i + 1:]:
                    if float(np.hypot(*(np.asarray(oa.xy) - np.asarray(ob.xy)))) > COSEEN_M:
                        now.add((min(a, b), max(a, b)))
        for p in now:
            self._coseen_n[p] = self._coseen_n.get(p, 0) + 1
            if self._coseen_n[p] >= COSEEN_TICKS:
                self._coseen[p] = ts
        for p in [p for p, t in self._coseen.items() if ts - t > COSEEN_MEMORY_S]:
            del self._coseen[p]
        for p in [p for p in self._coseen_n if p not in now]:
            del self._coseen_n[p]

    def _move(self, ts: float) -> None:
        for w in self.workers.values():
            if not w.members:
                continue
            dt = ts - w.last_ts
            fresh = w.last_ts == w.first_ts and dt == 0
            c = _consensus(list(w.members.values()), None if fresh else self._predicted(w, ts))
            new = c if fresh else SMOOTH * c + (1 - SMOOTH) * w.xy
            if dt > 0:
                w.vel = 0.5 * (new - w.xy) / dt + 0.5 * w.vel
            w.xy, w.last_ts = new, ts

    def _merge(self, ts: float) -> None:
        active = sorted(g for g, w in self.workers.items() if w.members and w.last_ts == ts)
        seen_pairs = set()
        for a_i, a in enumerate(active):
            for b in active[a_i + 1:]:
                wa, wb = self.workers.get(a), self.workers.get(b)
                if wa is None or wb is None:
                    continue
                pair = (a, b)
                seen_pairs.add(pair)
                # segment-aware: a single view with hidden feet sits at its far end until merged
                d = min(float(np.hypot(*(wa.xy - wb.xy))), min(o.dist(wb.xy) for o in wa.members.values()),
                        min(o.dist(wa.xy) for o in wb.members.values()))
                close = set(wa.cams).isdisjoint(wb.cams) and d < MERGE_M
                self._close[pair] = self._close.get(pair, 0) + 1 if close else 0
                need = MERGE_TICKS_LONG if pair in self._coseen else MERGE_TICKS
                if self._close[pair] >= need:
                    self._absorb(keep=a, drop=b)      # a < b: the older id survives
        for p in [p for p in self._close if p not in seen_pairs]:
            del self._close[p]

    def _absorb(self, keep: int, drop: int) -> None:
        wk, wd = self.workers[keep], self.workers.pop(drop)
        for k, o in wd.members.items():
            self.assign[k] = keep
            wk.members[k] = o
        # links of tracks not seen this tick follow too; a clash with a camera the merged
        # worker already has is resolved on the next update (nearest track keeps it)
        for k, g in list(self.assign.items()):
            if g == drop:
                self.assign[k] = keep
        wk.first_ts = min(wk.first_ts, wd.first_ts)
        wk.xy = _consensus(list(wk.members.values()))
        for p in [p for p in self._close if drop in p]:
            del self._close[p]
        moved: dict[tuple[int, int], float] = {}
        for (x, y), t in self._coseen.items():
            if {x, y} == {keep, drop}:
                continue
            x, y = (keep if x == drop else x), (keep if y == drop else y)
            p = (min(x, y), max(x, y))
            moved[p] = max(t, moved.get(p, t))
        self._coseen = moved
        self._coseen_n = {p: n for p, n in self._coseen_n.items() if drop not in p}


def cluster_vehicles(points: list[tuple[str, str, tuple[float, float], tuple]]) -> list[Vehicle]:
    """points: (label, cam, floor xy, box) -> vehicles, merging same-label points across cameras.
    Several points of one camera may belong to one vehicle (its box's ground contacts), but
    a camera's points from different boxes are not merged."""
    out: list[Vehicle] = []
    owner: dict[tuple[str, tuple], Vehicle] = {}            # (cam, box) -> vehicle
    for label, cam, xy, box in sorted(points, key=lambda p: p[0]):
        p = np.asarray(xy, dtype=float)
        v = owner.get((cam, tuple(box)))
        if v is None:
            best = None
            for c in out:
                if c.label != label or any(pc == cam and tuple(pb) != tuple(box) for pc, _, pb in c.points):
                    continue
                d = float(np.hypot(*(c.xy - p)))
                if d < VEHICLE_CLUSTER_M and (best is None or d < best[0]):
                    best = (d, c)
            if best is None:
                v = Vehicle(label=label, xy=p, points=[], pad=VEHICLE_PAD_M.get(label, 0.3))
                out.append(v)
            else:
                v = best[1]
            owner[(cam, tuple(box))] = v
        v.points.append((cam, tuple(p), box))
        v.xy = np.mean(np.array([q for _, q, _ in v.points]), axis=0)
    return out
