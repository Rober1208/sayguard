"""Evaluate rules on each processed frame and decide when an alert fires.

A condition must hold continuously for `min_duration_s` (short dropouts are
tolerated, so a single missed detection does not reset the timer), and the same
rule/camera/subject cannot fire again within `cooldown_s`.

The dropout grace is measured in wall time between processed frames, so it has
to follow the camera's real frame interval: a fixed 1 s would reset every timer
on each frame once a camera is processed less than once per second (CPU
fallback, model-load stalls, GPU contention) and no rule with a duration could
ever fire.  It is capped so that a genuine stall still resets the timer.

rules.json is the only copy of what the site manager has said, so it is written
atomically and a damaged file is set aside (rules.json.bad-<ts>) instead of
stopping the service from starting.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..types import Detection, Person
from .model import Rule, detector_labels, label_matches

log = logging.getLogger("sayguard.rules")

GRACE_S = 1.0           # minimum dropout grace (and the default when the frame interval is unknown)
GRACE_FRAMES = 2.5      # ...otherwise this many frame intervals
MAX_GRACE_S = 5.0       # ...but never more: a longer gap is a real interruption
NO_HELMET_MIN_OBS = 3
NO_HELMET_RATIO = 0.6
REPLACE_RETRIES = 5     # Windows AV/indexer/OneDrive can hold rules.json open for a moment


def grace_for(frame_dt: float | None) -> float:
    """Dropout grace for a camera whose processed frames are `frame_dt` seconds apart."""
    if frame_dt is None or not math.isfinite(frame_dt) or frame_dt <= 0:
        return GRACE_S
    return min(MAX_GRACE_S, max(GRACE_S, GRACE_FRAMES * frame_dt))


def _replace(src: Path, dst: Path) -> None:
    for i in range(REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == REPLACE_RETRIES - 1:
                raise
            time.sleep(0.05 * 2 ** i)


def _atomic_write(path: Path, data: str) -> None:
    """tmp file + fsync + rename: a crash mid-save leaves the previous file intact.

    Without the fsync the rename can reach the disk before the data does.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@dataclass
class CameraInfo:
    id: str
    name: str
    zone: str
    zones: list[str] = field(default_factory=list)   # site mode: floor-plan zones this camera sees

    @property
    def all_zones(self) -> list[str]:
        return [self.zone, *self.zones]


SITE = "site"   # the pseudo camera that site-mode rules are evaluated on (one subject per worker)


@dataclass
class SiteSubject:
    """A worker on the floor plan: one global id, whichever cameras see them."""
    gid: int
    xy: tuple[float, float]
    zones: list[str]
    cams: list[str]
    box: tuple | None = None         # pixel box, only for per-camera evaluation (cam_id != SITE)


@dataclass
class SiteObject:
    label: str
    xy: tuple[float, float]
    zones: list[str]
    cams: list[str]
    distance_to: object = None       # callable(xy) -> metres to the nearest ground contact, else centre distance

    def dist(self, xy) -> float:
        if callable(self.distance_to):
            return float(self.distance_to(xy))
        return float(((self.xy[0] - xy[0]) ** 2 + (self.xy[1] - xy[1]) ** 2) ** 0.5)


@dataclass
class _Hit:
    first: float
    last: float
    box: tuple[int, int, int, int] | None = None
    reentry: bool = False          # site presence: started after the worker was out for SITE_REENTRY_S


@dataclass
class Fired:
    rule: Rule
    cam_id: str
    key: str
    track_id: int | None
    box: tuple[int, int, int, int] | None
    detail: str = ""


@dataclass
class EvalResult:
    fired: list[Fired] = field(default_factory=list)
    # subject key ("t<track>" or "cam") -> [(rule name, seconds active, severity)]
    active: dict[str, list[tuple[str, float, str]]] = field(default_factory=dict)
    # object boxes that take part in an active proximity rule, with the person box they are near
    links: list[tuple[tuple, tuple, float]] = field(default_factory=list)
    # site mode: (worker gid, object floor xy, metres) for active proximity rules, drawn on the plan
    site_links: list[tuple[int, tuple[float, float], float]] = field(default_factory=list)
    # site mode, site-wide rules (count, object present): rule id -> cameras that see what triggered it
    site_scope: dict[str, list[str]] = field(default_factory=dict)


def rect_gap(a: tuple, b: tuple) -> float:
    """Euclidean gap between two axis-aligned boxes (0 if they overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


SAME_PLACE_IOU = 0.4    # a new track this close to a person alerted within the cooldown is that person
SITE_SAME_PLACE_M = 1.5 # site mode: a new worker id is a worker alerted within the cooldown if it is this
                        # close to them while they are still tracked (a second id on one person), or, once
                        # their id is gone (hidden, re-detected), this close to where they were alerted ...
SITE_REAPPEAR_S = 10.0  # ... within this long of losing them (the tracker itself forgets after 6 s), or
SITE_HANDOVER_S = 3.0   # near where they were last seen within this long, the radius growing at walking
SITE_WALK_MPS = 1.0     # pace.  Anyone else, e.g. the next worker down the same aisle, is someone else
SITE_REENTRY_S = 10.0   # site presence rules: back in the zone after this long outside is a new entry and
                        # alerts again, cooldown or not (the cooldown spaces reminders while someone stays;
                        # it must not carry over to whoever an id is on next)
CAMERA_MIN_GAP_S = 15.0 # per-person alerts on one camera: at most one per rule per 15 s unless more people violate


def box_iou(a: tuple, b: tuple) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class RuleEngine:
    def __init__(self, cameras: list[CameraInfo], rules_path: Path | None = None):
        self.cameras = {c.id: c for c in cameras}
        self.rules: dict[str, Rule] = {}
        self.rules_path = rules_path
        self._hits: dict[tuple[str, str, str], _Hit] = {}
        self._last_fire: dict[tuple[str, str, str], float] = {}
        # (rule, camera) -> [(ts, box)] of recent per-person alerts, to catch track-id switches
        self._fired_boxes: dict[tuple[str, str], list[tuple[float, tuple]]] = {}
        self._cam_fire: dict[tuple[str, str], tuple[float, int]] = {}   # (rule, camera) -> (ts, people) of last alert
        # site alerts: rule -> worker key -> [alert ts, last seen ts, last seen xy, alert xy]
        self._site_fired: dict[str, dict[str, list]] = {}
        self._site_present: set[str] = set()                              # worker keys in the current tick
        self._site_seen: dict[tuple[str, str, str], float] = {}           # presence hit -> last tick it held
        self._lock = threading.RLock()
        self.version = 0

    # ---- rule management -------------------------------------------------
    def add(self, rule: Rule) -> Rule:
        with self._lock:
            self.rules[rule.id] = rule
            self._changed()
        return rule

    def remove(self, rule_id: str) -> bool:
        with self._lock:
            ok = self.rules.pop(rule_id, None) is not None
            if ok:
                for k in [k for k in self._hits if k[0] == rule_id]:
                    del self._hits[k]
                for k in [k for k in self._last_fire if k[0] == rule_id]:
                    del self._last_fire[k]
                for k in [k for k in self._fired_boxes if k[0] == rule_id]:
                    del self._fired_boxes[k]
                for k in [k for k in self._cam_fire if k[0] == rule_id]:
                    del self._cam_fire[k]
                self._site_fired.pop(rule_id, None)
                self._changed()
            return ok

    def set_enabled(self, rule_id: str, enabled: bool) -> Rule | None:
        with self._lock:
            r = self.rules.get(rule_id)
            if r:
                r.enabled = enabled
                self._changed()
            return r

    def list(self) -> list[Rule]:
        with self._lock:
            return sorted(self.rules.values(), key=lambda r: r.created_at)

    def _changed(self) -> None:
        self.version += 1
        self.save()

    def save(self) -> None:
        if not self.rules_path:
            return
        with self._lock:   # one writer at a time: they share the .tmp file
            self.rules_path.parent.mkdir(parents=True, exist_ok=True)
            data = [r.model_dump() for r in self.list()]
            _atomic_write(self.rules_path, json.dumps(data, ensure_ascii=False, indent=1))

    def _set_aside(self, reason: str) -> None:
        """Move the current rules file to rules.json.bad-<unix ts> so nothing the user said is lost."""
        bad = self.rules_path.with_name(f"{self.rules_path.name}.bad-{int(time.time())}")
        try:
            _replace(self.rules_path, bad)
        except OSError as e:   # still start: a service without the old rules beats no service
            log.error("rules file %s: %s; could not keep a copy (%s)", self.rules_path, reason, e)
            return
        log.error("rules file %s: %s; kept the original as %s", self.rules_path, reason, bad.name)

    def _read_saved(self) -> list | None:
        """Items from the rules file, or None when there is no usable file (missing or unreadable)."""
        if not (self.rules_path and self.rules_path.exists()):
            return None
        try:
            items = json.loads(self.rules_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:   # e.g. truncated by a crash mid-write
            reason = f"unreadable ({e})"
        else:
            if isinstance(items, list):
                return items
            reason = f"expected a list of rules, got {type(items).__name__}"
        self._set_aside(reason + "; starting from the default rules")
        return None

    def load(self, defaults: list[dict] | None = None) -> None:
        saved = self._read_saved()
        items = saved if saved is not None else (defaults or [])
        rules: dict[str, Rule] = {}
        skipped = 0
        for i, it in enumerate(items):
            try:
                r = Rule(**it)
            except (ValidationError, TypeError) as e:   # hand edit, old schema, not an object
                log.warning("skipping invalid rule #%d (%s): %s", i, "rules file" if saved is not None else "defaults",
                            str(e).replace("\n", " "))
                skipped += 1
                continue
            rules[r.id] = r
        if skipped and saved is not None:
            self._set_aside(f"{skipped} invalid rule(s) skipped")   # the save below drops them from rules.json
        with self._lock:
            self.rules = rules
            self._changed()

    # ---- queries ---------------------------------------------------------
    def applies(self, rule: Rule, cam: CameraInfo, ts: float | None = None) -> bool:
        if not rule.enabled:
            return False
        if not rule.covers(cam.id, cam.zone):
            return False
        if rule.time_window and not rule.time_window.contains(time.localtime(ts or time.time())):
            return False
        return True

    def scope_cameras(self, rule: Rule) -> list[str]:
        """Cameras the rule's camera/zone filters select, ignoring enabled and time_window.

        Empty means the rule can never fire anywhere, whatever the time.
        """
        return [c.id for c in self.cameras.values() if rule.covers_any(c.id, c.all_zones)]

    def objects_for_site(self) -> set[str]:
        """Object classes that enabled site-mode rules need now.  Every camera looks for them: a
        zone is a region of the floor plan, not a camera."""
        with self._lock:
            now = time.localtime()
            return {lab for r in self.rules.values() if r.object and r.enabled
                    and r.type.startswith("object") and (not r.time_window or r.time_window.contains(now))
                    for lab in detector_labels(r.object)}

    def objects_for_camera(self, cam_id: str) -> set[str]:
        cam = self.cameras[cam_id]
        with self._lock:
            return {r.object for r in self.rules.values()
                    if r.object and r.type.startswith("object") and self.applies(r, cam)}

    def all_objects(self) -> set[str]:
        with self._lock:
            return {r.object for r in self.rules.values() if r.object and r.enabled and r.type.startswith("object")}

    # ---- evaluation ------------------------------------------------------
    def evaluate(self, cam_id: str, ts: float, people: list[Person], objects: list[Detection],
                 frame_wh: tuple[int, int], frame_dt: float | None = None) -> EvalResult:
        """frame_dt: typical seconds between processed frames of this camera (None until known)."""
        cam = self.cameras[cam_id]
        res = EvalResult()
        W, H = frame_wh
        grace = grace_for(frame_dt)
        # held for the whole pass: _update/_expire mutate _hits, which remove() edits from the API thread
        with self._lock:
            rules = [r for r in self.rules.values() if self.applies(r, cam, ts)]
            for rule in rules:
                active: dict[str, tuple[int | None, tuple | None, str]] = {}
                if rule.type == "no_helmet":
                    for p in people:
                        if p.observations >= NO_HELMET_MIN_OBS and p.no_helmet_ratio >= NO_HELMET_RATIO:
                            active[f"t{p.track_id}"] = (p.track_id, p.body_box(W, H), "")
                elif rule.type == "person_present":
                    if people:
                        active["cam"] = (None, None, f"{len(people)} person(s)")
                elif rule.type == "person_count_over":
                    if len(people) > (rule.threshold or 0):
                        active["cam"] = (None, None, f"{len(people)} people > {int(rule.threshold or 0)}")
                elif rule.type == "object_present":
                    hits = [o for o in objects if label_matches(o.label, rule.object)]
                    if hits:
                        b = tuple(int(v) for v in hits[0].xyxy)
                        active["cam"] = (None, b, f"{rule.object} x{len(hits)}")
                elif rule.type == "object_near_person":
                    objs = [o for o in objects if label_matches(o.label, rule.object)]
                    for p in people:
                        body = p.body_box(W, H)
                        best = None
                        for o in objs:
                            d_m = rect_gap(body, o.xyxy) * p.metres_per_px
                            if best is None or d_m < best[0]:
                                best = (d_m, o)
                        if best and best[0] <= (rule.threshold or 0):
                            active[f"t{p.track_id}"] = (p.track_id, body, f"{rule.object} {best[0]:.1f} m")
                            res.links.append((tuple(int(v) for v in best[1].xyxy), body, best[0]))
                self._update(rule, cam_id, ts, active, res, grace)
            self._expire(cam_id, ts, grace)
        return res

    def evaluate_site(self, ts: float, workers: list[SiteSubject], objects: list[SiteObject],
                      frame_dt: float | None = None, cam_id: str = SITE) -> EvalResult:
        """Site mode: rules over workers on the floor plan.  A worker is one subject however many
        cameras see them (key "g<gid>"), so a rule fires once per person, not once per camera;
        zones are floor-plan regions; distances are metres on the floor.  Hard-hat rules need the
        PPE detector's head boxes and are not evaluated here.

        cam_id = a real camera id instead evaluates one camera's own tracks the way the
        per-camera pipeline does (keys "t<track>", one "cam" subject for presence rules, and the
        per-camera duplicate guards): scripts/eval_site.py uses it as the baseline."""
        res = EvalResult()
        grace = grace_for(frame_dt)
        per_cam = cam_id != SITE
        key_of = (lambda w: f"t{w.gid}") if per_cam else (lambda w: f"g{w.gid}")
        place_of = (lambda w: w.box) if per_cam else (lambda w: tuple(w.xy))
        whole = "cam" if per_cam else SITE

        def covers_cam(rule: Rule, cam: str) -> bool:
            return "*" in rule.cameras or cam in rule.cameras

        def in_scope(rule: Rule, cams: list[str], zones: list[str]) -> bool:
            if "*" not in rule.cameras and not any(c in rule.cameras for c in cams):
                return False
            if rule.zones and not any(z in rule.zones for z in zones):
                return False
            return not any(z in rule.zones_exclude for z in zones)

        with self._lock:
            now = time.localtime(ts)
            rules = [r for r in self.rules.values()
                     if r.enabled and (not r.time_window or r.time_window.contains(now))]
            if not per_cam:
                # follow every alerted worker wherever they are, in a rule's scope or not
                present = {key_of(w): tuple(w.xy) for w in workers}
                self._site_present = set(present)
                for entries in self._site_fired.values():
                    for k, e in entries.items():
                        if k in present:
                            e[1:3] = [ts, present[k]]
            for rule in rules:
                active: dict[str, tuple[int | None, tuple | None, str]] = {}
                subs = [w for w in workers if in_scope(rule, w.cams, w.zones)]
                if rule.type == "person_present":
                    if per_cam and subs:
                        active[whole] = (None, None, f"{len(subs)} person(s)")
                    elif not per_cam:
                        for w in subs:
                            active[key_of(w)] = (w.gid, place_of(w), f"in {w.zones[0]}" if w.zones else "")
                elif rule.type == "person_count_over":
                    if len(subs) > (rule.threshold or 0):
                        active[whole] = (None, None, f"{len(subs)} people > {int(rule.threshold or 0)}")
                        res.site_scope[rule.id] = sorted({c for w in subs for c in w.cams if covers_cam(rule, c)})
                elif rule.type == "object_present":
                    hits = [o for o in objects if label_matches(o.label, rule.object) and in_scope(rule, o.cams, o.zones)]
                    if hits:
                        active[whole] = (None, None, f"{rule.object} x{len(hits)}")
                        res.site_scope[rule.id] = sorted({c for o in hits for c in o.cams if covers_cam(rule, c)})
                elif rule.type == "object_near_person":
                    objs = [o for o in objects if label_matches(o.label, rule.object)]
                    for w in subs:
                        best = min(((o.dist(w.xy), o) for o in objs), default=None, key=lambda t: t[0])
                        if best and best[0] <= (rule.threshold or 0):
                            active[key_of(w)] = (w.gid, place_of(w), f"{rule.object} {best[0]:.1f} m")
                            res.site_links.append((w.gid, tuple(best[1].xy), best[0]))
                self._update(rule, cam_id, ts, active, res, grace)
            self._expire(cam_id, ts, grace)
        return res

    def _update(self, rule: Rule, cam_id: str, ts: float, active: dict, res: EvalResult,
                grace: float = GRACE_S) -> None:
        n_people = sum(1 for k in active if k != "cam")
        for key, (track_id, box, detail) in active.items():
            hk = (rule.id, cam_id, key)
            hit = self._hits.get(hk)
            if hit is None or ts - hit.last > grace:
                first = ts
                if cam_id == SITE and key != SITE and box is not None:
                    # a worker whose global id just changed carries on the condition it was already
                    # in: its duration counts from when the previous id at this spot started
                    prev = [h for (r, c, k), h in self._hits.items()
                            if r == rule.id and c == SITE and k != key and k != SITE and h.box is not None
                            and ts - h.last <= max(grace, 2.0)
                            and ((h.box[0] - box[0]) ** 2 + (h.box[1] - box[1]) ** 2) ** 0.5 < SITE_SAME_PLACE_M]
                    if prev:
                        first = min(h.first for h in prev)
                reentry = (cam_id == SITE and rule.type == "person_present" and key != SITE
                           and ts - self._site_seen.get(hk, -1e9) >= SITE_REENTRY_S)
                hit = self._hits[hk] = _Hit(first=first, last=ts, box=box, reentry=reentry)
            hit.last, hit.box = ts, box
            if cam_id == SITE and rule.type == "person_present":
                self._site_seen[hk] = ts
            dur = ts - hit.first
            res.active.setdefault(key, []).append((rule.name, dur, rule.severity))
            last_fire = self._last_fire.get(hk, -1e9)
            new_entry = hit.reentry and last_fire < hit.first
            if dur >= rule.min_duration_s and (ts - last_fire >= rule.cooldown_s or new_entry):
                self._last_fire[hk] = ts
                if cam_id == SITE and key != SITE and box is not None:
                    # a worker whose global id changed (hidden behind a rack, re-detected, or a
                    # second id on the same person) is still the worker alerted moments ago
                    def near(p, slack=0.0):
                        return ((box[0] - p[0]) ** 2 + (box[1] - p[1]) ** 2) ** 0.5 < SITE_SAME_PLACE_M + slack

                    def same_worker(k, e):
                        if k in self._site_present:              # still tracked: only a second id on them
                            return near(e[2])
                        gone = ts - e[1]
                        return ((gone <= SITE_REAPPEAR_S and near(e[3]))
                                or (gone <= SITE_HANDOVER_S and near(e[2], SITE_WALK_MPS * gone)))
                    recent = self._site_fired.setdefault(rule.id, {})
                    same = [e for k, e in recent.items()
                            if k != key and ts - e[0] < rule.cooldown_s and same_worker(k, e)]
                    if same:
                        first = max(same, key=lambda e: e[0])
                        self._last_fire[hk] = first[0]     # their cooldown runs from that alert, not from now
                        recent[key] = [first[0], ts, box, first[3]]
                        continue
                    recent[key] = [ts, ts, box, box]
                if key not in ("cam", SITE) and cam_id != SITE:   # site keys are global ids already
                    if box is not None and self._same_place(rule, cam_id, ts, box):
                        continue   # same person under a new track id (occlusion, id switch, looped clip)
                    last_t, last_n = self._cam_fire.get((rule.id, cam_id), (-1e9, 0))
                    if ts - last_t < CAMERA_MIN_GAP_S and n_people <= last_n:
                        continue   # this camera already alerted on this rule moments ago and nobody new joined
                    self._cam_fire[(rule.id, cam_id)] = (ts, n_people)
                    if box is not None:
                        self._fired_boxes.setdefault((rule.id, cam_id), []).append((ts, box))
                res.fired.append(Fired(rule=rule, cam_id=cam_id, key=key, track_id=track_id, box=box, detail=detail))

    def _same_place(self, rule: Rule, cam_id: str, ts: float, box: tuple) -> bool:
        return any(ts - t < rule.cooldown_s and box_iou(b, box) >= SAME_PLACE_IOU
                   for t, b in self._fired_boxes.get((rule.id, cam_id), []))

    def _expire(self, cam_id: str, ts: float, grace: float = GRACE_S) -> None:
        for hk in [k for k, h in self._hits.items() if k[1] == cam_id and ts - h.last > grace]:
            del self._hits[hk]
        if cam_id == SITE:
            for hk in [k for k, t in self._site_seen.items() if ts - t > SITE_REENTRY_S]:
                del self._site_seen[hk]
        # fire history only matters within a rule's cooldown; drop it afterwards (long-running service)
        for hk in [k for k, t in self._last_fire.items() if k[1] == cam_id and
                   ts - t > (self.rules[k[0]].cooldown_s if k[0] in self.rules else 0)]:
            del self._last_fire[hk]
        for fk in [k for k in self._fired_boxes if k[1] == cam_id]:
            cd = self.rules[fk[0]].cooldown_s if fk[0] in self.rules else 0
            kept = [(t, b) for t, b in self._fired_boxes[fk] if ts - t < cd]
            if kept:
                self._fired_boxes[fk] = kept
            else:
                del self._fired_boxes[fk]
        for ck in [k for k, (t, _) in self._cam_fire.items() if k[1] == cam_id and ts - t > CAMERA_MIN_GAP_S]:
            del self._cam_fire[ck]
        if cam_id == SITE:
            for rid in list(self._site_fired):
                cd = self.rules[rid].cooldown_s if rid in self.rules else 0
                kept = {k: e for k, e in self._site_fired[rid].items() if ts - e[0] < cd}
                if kept:
                    self._site_fired[rid] = kept
                else:
                    del self._site_fired[rid]
