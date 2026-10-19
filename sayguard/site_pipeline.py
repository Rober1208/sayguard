"""Site mode: calibrated cameras of one site fused on its floor plan.

Same loop as the per-camera pipeline (one batched detector call per tick), but after the
per-camera tracks every person is placed on the floor and sayguard.sitetrack gives each
worker one id across cameras.  Rules are then evaluated per worker, not per camera
(RuleEngine.evaluate_site): a worker seen by four cameras next to a forklift raises one
alert, distances are metres on the floor, and zones are regions drawn once on the plan.

The demo footage is synchronized multi-camera video (NVIDIA PhysicalAI-SmartSpaces), played
frame-locked by a SyncedFileGroup; when it loops, the floor-plan state starts over (worker
ids keep counting up, so an id never means two people).  The detector is trained for the
site (person, forklift, pallet truck, transporter, mobile robot, humanoid robot: robots are
their own classes, so they are never counted as workers).  Every class it knows can be named
in a rule; any other object a rule names falls back to the open-vocabulary detector.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import resolve
from .pipeline import CamState, Pipeline, source_arg
from .rules.engine import Fired, SiteObject, SiteSubject
from .rules.model import canon_label, label_matches
from .site import HEAD_TOP_M, Site, foot_point, touches_bottom
from .site_view import PlanRenderer, draw_site_tile, head_of
from .sitetrack import Obs, SiteTracker, Worker, cluster_vehicles
from .sources import SyncedFileGroup, VideoSource, is_camera_index, is_network
from .tracking import CameraTracker
from .types import Detection, Event, FramePacket

log = logging.getLogger("sayguard.site")

PERSON = "person"
MAX_SEGMENT_M = 15.0     # longer head-to-feet floor segments are geometry blow-ups near the horizon
# Confidence the site detector needs, chosen on the validation scene (configs/eval/warehouse_020.yaml),
# never on the test scenes: most worker positions and forklift boxes below these were not real.
SHOW_CONF = 0.7          # a worker is reported (map, head count, rules) once a camera sees it this surely,
KEEP_CONF = 0.6          # and while one still sees it this surely; fainter tracks keep their id, hidden
OBJECT_CONF = 0.6        # vehicles and robots (pallet trucks and empty floor were taken for forklifts below)


def person_tracker(fps: float) -> CameraTracker:
    """Per-camera tracker for whole-person boxes: merges overlapping and nested boxes of one
    person first, and keeps a track through a longer occlusion behind a rack."""
    return CameraTracker(fps, lost_s=3.0, labels=(PERSON,), merge_iou=0.45, contain=0.8)


def load_site(cfg: dict) -> tuple[Site, dict]:
    """The config's `site:` block and top-level `zones:` -> (Site, {camera id: CameraCalib})."""
    sc = cfg["site"]
    zones = [{"id": zid, "name": z.get("en", zid), **{k: z[k] for k in ("polygon", "plan_polygon") if k in z}}
             for zid, z in cfg.get("zones", {}).items() if "polygon" in z or "plan_polygon" in z]
    site = Site.load(resolve(sc["calibration"]), resolve(sc["plan"]) if sc.get("plan") else None, zones)
    missing = [c.get("calib", c["id"]) for c in cfg["cameras"] if c.get("calib", c["id"]) not in site.cameras]
    if missing:
        raise ValueError(f"cameras without calibration: {missing}")
    return site, {c["id"]: site.cameras[c.get("calib", c["id"])] for c in cfg["cameras"]}


@dataclass
class FusionResult:
    workers: list[Worker]
    vehicles: list
    subjects: list[SiteSubject]
    objects: list[SiteObject]
    tracked: dict[str, list[Detection]] = field(default_factory=dict)    # per camera: person tracks
    objs: dict[str, list[Detection]] = field(default_factory=dict)       # per camera: object boxes
    raw: dict[str, list[Detection]] = field(default_factory=dict)        # per camera: every person box (privacy)

    @property
    def camera_sum(self) -> int:
        return sum(len(v) for v in self.tracked.values())


class SiteFusion:
    """Per-camera detections of one tick -> workers and objects on the floor plan.  Shared by
    the live SitePipeline and scripts/eval_site.py, so what is measured is what runs."""

    def __init__(self, site: Site, calib_of: dict, fps: float, site_zone: str | None = None):
        self.site, self.calib_of, self.fps, self.site_zone = site, calib_of, fps, site_zone
        self.trackers = {cam: person_tracker(fps) for cam in calib_of}
        self.tracker = SiteTracker()
        self._shown: set[int] = set()

    def reset(self) -> None:
        """The clips looped: nobody carries over (ids keep counting up)."""
        self.trackers = {cam: person_tracker(self.fps) for cam in self.calib_of}
        self.tracker.reset(keep_ids=True)
        self._shown = set()

    def zones(self, xy) -> list[str]:
        zs = self.site.zones_of(float(xy[0]), float(xy[1]))
        return zs + [self.site_zone] if self.site_zone else zs

    def person_obs(self, cam_id: str, b: Detection, W: int, H: int) -> Obs:
        """A camera track as a floor observation: the point under the box bottom (the feet, when
        visible) and the segment to the point under the box top at head height."""
        cam = self.calib_of[cam_id]
        feet = cam.to_world(*foot_point(b.xyxy), src_wh=(W, H))
        head = None if b.xyxy[1] <= 1 else cam.to_world_at((b.xyxy[0] + b.xyxy[2]) / 2, b.xyxy[1], HEAD_TOP_M,
                                                           src_wh=(W, H))
        seg = (head, feet) if head and np.hypot(head[0] - feet[0], head[1] - feet[1]) < MAX_SEGMENT_M else None
        feet_visible = not touches_bottom(b.xyxy, H)
        use_head = not feet_visible and seg is not None             # feet cut by the frame edge: head point
        xy = seg[0] if use_head else feet
        # floor uncertainty: metres one pixel spans at the point used, times the box's
        # localisation error (a few pixels plus ~5 % of its height)
        u = (b.xyxy[0] + b.xyxy[2]) / 2
        if use_head:
            nxt = cam.to_world_at(u, b.xyxy[1] + 1, HEAD_TOP_M, src_wh=(W, H)) or xy
        else:
            nxt = cam.to_world(u, b.xyxy[3] + 1, src_wh=(W, H))
        per_px = float(np.hypot(nxt[0] - xy[0], nxt[1] - xy[1]))
        sigma = float(np.clip(per_px * (1.5 + 0.05 * (b.xyxy[3] - b.xyxy[1])), 0.15, 5.0))
        return Obs(cam_id, b.track_id, xy, b.xyxy, b.conf, feet_visible, seg, sigma)

    def step(self, ts: float, inputs: list[tuple[str, tuple[int, int], list[Detection], list[Detection]]]) -> FusionResult:
        """inputs: [(camera id, (frame w, h), detections, extra object detections)]; every
        non-person detection is an object."""
        obs: list[Obs] = []
        pts = []
        res = FusionResult([], [], [], [])
        for cam_id, (W, H), dets, extra in inputs:
            cam = self.calib_of[cam_id]
            persons = [x for x in dets if x.label == PERSON]
            _, tracked = self.trackers[cam_id].update(persons, ts)
            obs += [self.person_obs(cam_id, b, W, H) for b in tracked]
            objs = [Detection(x.xyxy, canon_label(x.label), x.conf, source=x.source)
                    for x in [d for d in dets if d.conf >= OBJECT_CONF] + list(extra) if x.label != PERSON]
            for v in objs:
                if touches_bottom(v.xyxy, H):     # its ground contact is outside the frame
                    continue
                x1, _, x2, y2 = v.xyxy
                for u in (x1, (x1 + x2) / 2, x2):  # both bottom corners and the centre: the near side
                    pts.append((v.label, cam_id, cam.to_world(u, y2, src_wh=(W, H)), v.xyxy))
            res.tracked[cam_id], res.objs[cam_id], res.raw[cam_id] = tracked, objs, persons
        shown = set()
        for w in self.tracker.update(obs, ts):
            c = max((o.conf for o in w.members.values()), default=0.0)
            if c >= SHOW_CONF or (w.gid in self._shown and c >= KEEP_CONF):
                shown.add(w.gid)
                res.workers.append(w)
        self._shown = shown
        res.vehicles = cluster_vehicles(pts)
        res.subjects = [SiteSubject(w.gid, (float(w.xy[0]), float(w.xy[1])), self.zones(w.xy), w.cams)
                        for w in res.workers]
        res.objects = [SiteObject(v.label, (float(v.xy[0]), float(v.xy[1])), self.zones(v.xy), v.cams,
                                  distance_to=v.distance_to) for v in res.vehicles]
        return res


class SitePipeline(Pipeline):
    unsupported_rule_types = {"no_helmet"}        # needs the PPE detector's head boxes (per-camera mode)

    def __init__(self, cfg: dict):
        sc = cfg["site"]
        self.site, calib_of = load_site(cfg)
        self.site_zone = sc.get("zone")                 # whole-site zone id every worker is in, e.g. warehouse
        fps = float(cfg["streams"]["fps"])
        srcs = {c["id"]: source_arg(c["source"]) for c in cfg["cameras"]}
        # recorded files are played frame-locked; live cameras are already simultaneous
        files = not any(is_network(s) or is_camera_index(s) for s in srcs.values())
        self._group = SyncedFileGroup(srcs, fps=fps) if files else None
        super().__init__(cfg)
        self.calib_of = calib_of
        self.fusion = SiteFusion(self.site, calib_of, fps, self.site_zone)
        labels = getattr(self.detector, "labels", None) or {}
        names = labels.values() if isinstance(labels, dict) else labels
        self.known = {canon_label(n) for n in names} - {PERSON}   # objects the site detector finds itself
        self.plan = PlanRenderer(self.site, out_width=int(sc.get("plan_width", 960)))
        self.map_jpeg: bytes | None = None
        self.map_seq = 0
        self._loop_no = 0
        self._site_stats = {"unique_people": 0, "camera_sum": 0, "zones": {}, "vehicles": 0}
        # which floor zones each camera sees: only for the scope check of a drafted rule ("no camera
        # covers aisle 7") and the dashboard's coverage preview; evaluation uses worker positions
        for c in cfg["cameras"]:
            seen = self._zones_seen(self.calib_of[c["id"]])
            self.engine.cameras[c["id"]].zones = seen
            c["zones"] = [c["zone"], *seen]

    # ------------------------------------------------------------------ hooks
    def _make_source(self, cam_cfg: dict):
        if self._group is not None:
            return self._group.views[cam_cfg["id"]]
        return VideoSource(cam_cfg["id"], source_arg(cam_cfg["source"]))

    def _make_tracker(self) -> CameraTracker:
        return person_tracker(self.fps_target)                        # unused: SiteFusion tracks

    def _zones_seen(self, cam, step_m: float = 0.5) -> list[str]:
        """Zones with any floor point visible to the camera (sampled on a grid inside the polygon)."""
        out = []
        for z in self.site.zones:
            (x0, y0), (x1, y1) = z.polygon.min(axis=0), z.polygon.max(axis=0)
            found = False
            for x in np.arange(x0, x1 + 1e-9, step_m):
                for y in np.arange(y0, y1 + 1e-9, step_m):
                    if not z.contains(float(x), float(y)):
                        continue
                    uv = cam.to_image(float(x), float(y))
                    if uv and 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
                        found = True
                        break
                if found:
                    break
            if found:
                out.append(z.id)
        return out

    def _ov_wanted(self) -> set[str]:
        return {o for o in self.engine.objects_for_site() if o not in self.known}

    # ------------------------------------------------------------------ loop
    def _run_open_vocab(self, due: list[tuple[CamState, FramePacket]]) -> None:
        """Only objects the site detector does not know go to YOLO-World, on every due camera."""
        wanted = self._ov_wanted()
        ov = self._ov_for(wanted)
        if ov is None or not due:
            if not wanted:
                for st in self.states.values():
                    st.ov_last = []
            return
        t1 = time.perf_counter()
        try:
            res = ov.infer([p.frame for _, p in due])
        except Exception as e:
            log.warning("open-vocabulary inference failed", exc_info=True)
            for st, _ in due:
                st.ov_last = []
            self._ov_failed(f"{type(e).__name__}: {e}"[:300], permanent=False, retry_s=60.0)
            return
        self.stats["ov_ms"] = 0.8 * self.stats["ov_ms"] + 0.2 * (time.perf_counter() - t1) * 1000
        for (st, p), r in zip(due, res):
            st.ov_last, st.ov_ts = [d for d in r if canon_label(d.label) in wanted], p.ts
        if self._ov_error is not None:
            with self._ov_lock:
                self._ov_error = None

    def _process(self, batch: list[tuple[CamState, FramePacket]], tick: float, every_n: int) -> None:
        t0 = time.perf_counter()
        dets = self.detector.infer([p.frame for _, p in batch])
        t_det = time.perf_counter() - t0
        ov_due = self._schedule(batch, tick, every_n)
        loop_no = max(p.loop for _, p in batch)
        if loop_no != self._loop_no:                    # the clips wrapped: nobody carries over
            self._loop_no = loop_no
            self.fusion.reset()
        try:
            self._run_open_vocab(ov_due)
        except Exception:
            self._log_failure("ov", "open-vocabulary step failed")

        inputs = []
        for (st, pkt), d in zip(batch, dets):
            H, W = pkt.frame.shape[:2]
            inputs.append((st.info.id, (W, H), d, st.ov_last if pkt.ts - st.ov_ts < 1.5 else []))
        fr = self.fusion.step(tick, inputs)
        workers, vehicles, subjects, objects = fr.workers, fr.vehicles, fr.subjects, fr.objects
        per_cam = {}
        for st, pkt in batch:
            cid = st.info.id
            st.people = len(fr.tracked.get(cid, []))
            per_cam[cid] = (st, pkt, fr.tracked.get(cid, []), fr.objs.get(cid, []), fr.raw.get(cid, []))
            # every person box this frame plus where one was in the last second (detector misses)
            st.blur_now = [b for b, _ in st.face_hold.update(pkt.ts, [(x.xyxy, False) for x in fr.raw.get(cid, [])])]
        fps = np.median([st.fps for st, _ in batch if st.fps > 0.5]) if any(st.fps > 0.5 for st, _ in batch) else 0
        res = self.engine.evaluate_site(tick, subjects, objects, frame_dt=1.0 / fps if fps else None)
        by_gid = {w.gid: w for w in workers}
        for f in res.fired:
            try:
                self._emit_site(f, by_gid, per_cam, vehicles, res.site_scope.get(f.rule.id, []))
            except Exception:
                self._log_failure("emit-site", "could not record site alert %r", f.rule.name)

        gid_of = {k: w.gid for w in workers for k in w.members}
        rule_types = {r.name: r.type for r in self.engine.list()}
        for cam_id, (st, pkt, tracked, objs, raw) in per_cam.items():
            try:
                people = [(b.xyxy, gid_of.get((cam_id, b.track_id)), b.conf) for b in tracked]
                tile = draw_site_tile(pkt.frame, st.title, people, [(o.xyxy, o.label) for o in objs], res.active,
                                      rule_types, st.fps, blur=self.blur, out_width=self.tile_w,
                                      blur_boxes=st.blur_now)
                ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    st.jpeg, st.jpeg_seq = buf.tobytes(), st.jpeg_seq + 1
            except Exception:
                self._log_failure(f"tile-{cam_id}", "camera %s: tile failed", cam_id)
        try:
            active_gids = {int(k[1:]) for k in res.active if k.startswith("g")}
            n_sum = fr.camera_sum
            title = f"FLOOR PLAN  {len(workers)} workers (cameras count {n_sum})  {len(vehicles)} vehicles"
            img = self.plan.render([(w.gid, (float(w.xy[0]), float(w.xy[1])), len(w.cams)) for w in workers],
                                   [(v.label, (float(v.xy[0]), float(v.xy[1]))) for v in vehicles],
                                   res.site_links, active_gids, title)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                self.map_jpeg, self.map_seq = buf.tobytes(), self.map_seq + 1
            zones: dict[str, int] = {}
            for s in subjects:
                for z in s.zones:
                    zones[z] = zones.get(z, 0) + 1
            self._site_stats = {"unique_people": len(workers), "camera_sum": n_sum, "zones": zones,
                                "vehicles": len(vehicles)}
        except Exception:
            self._log_failure("plan", "floor plan failed")
        with self.cond:
            self.cond.notify_all()
        self._record_throughput(len(batch), t_det)

    def _emit_site(self, f: Fired, by_gid: dict[int, Worker], per_cam: dict, vehicles, scope: list[str]) -> None:
        rule = f.rule
        w = by_gid.get(f.track_id) if f.track_id is not None else None
        best = w.best_view(rule.cameras) if w else None
        if best is not None:
            cam_id = best.cam
        else:                                             # site-wide rule: a camera that sees its cause
            cands = [c for c in scope if c in per_cam] or list(per_cam)
            cam_id = max(cands, key=lambda c: len(per_cam[c][2]) + len(per_cam[c][3])) if cands \
                else next(iter(self.states))
        st, pkt, tracked, objs, raw = per_cam.get(cam_id, (self.states[cam_id], None, [], [], []))
        dist = None
        if w is not None and rule.type == "object_near_person":
            near = [v.distance_to(w.xy) for v in vehicles if label_matches(v.label, rule.object)]
            dist = round(min(near), 2) if near else None
        if w is not None:
            zone = next((z for z in self.fusion.zones(w.xy) if not rule.zones or z in rule.zones), None)
        else:
            zone = rule.zones[0] if rule.zones else self.site_zone
        ev = Event(id=uuid.uuid4().hex[:12], ts=pkt.ts if pkt else time.time(), cam_id=cam_id, cam_name=st.info.name,
                   zone=zone or st.info.zone, rule_id=rule.id, rule_name=rule.name, rule_type=rule.type,
                   severity=rule.severity, message=rule.message or rule.name, track_id=f.track_id,
                   box=tuple(int(v) for v in best.box) if best else None,
                   extra={"detail": f.detail, "source_text": rule.source_text, "global_id": f.track_id,
                          "cameras": w.cams if w else (scope or sorted(per_cam)), "distance_m": dist,
                          "world": [round(float(w.xy[0]), 2), round(float(w.xy[1]), 2)] if w else None})
        if pkt is not None:
            ev.snapshot = self._snapshot_site(ev, pkt.frame, st.blur_now, objs)
        self.store.add(ev)
        log.info("ALERT site %s W-%s %s (%s)", rule.name, f.track_id, f.detail, ", ".join(ev.extra["cameras"]))

    def _snapshot_site(self, ev: Event, frame: np.ndarray, blur_boxes: list[tuple], objs: list[Detection]) -> str | None:
        try:
            H, W = frame.shape[:2]
            snap = frame.copy()
            if self.blur:
                from .privacy import anonymise
                anonymise(snap, [(head_of(b), False) for b in blur_boxes])
            lw = max(2, W // 320)
            for o in objs:
                cv2.rectangle(snap, tuple(map(int, o.xyxy[:2])), tuple(map(int, o.xyxy[2:])), (0, 160, 255), lw)
            if ev.box:
                cv2.rectangle(snap, ev.box[:2], ev.box[2:], (40, 40, 235), lw)
            snap = cv2.resize(snap, (640, int(H * 640 / W)), interpolation=cv2.INTER_AREA)
            return self.store.save_snapshot(ev, snap)
        except Exception:
            self._log_failure("snapshot", "could not save the snapshot for alert %s", ev.id)
            return None

    # ---------------------------------------------------------------- status
    def _ov_status(self, now: float) -> dict:
        wanted = self._ov_wanted()
        with self._ov_lock:
            ov, err, retry_at, busy = self._ov, self._ov_error, self._ov_retry_at, self._ov_busy()
        ready = bool(wanted) and ov is not None and not busy and ov.ready_for(wanted)
        if not wanted or busy:
            err = None
        return {"loaded": ready, "busy": busy, "error": err,
                "retry_in_s": round(max(0.0, retry_at - now)) if err and retry_at != float("inf") else None,
                "classes": sorted(wanted), "ms": round(self.stats["ov_ms"], 1)}

    def status(self) -> dict:
        s = super().status()
        now = time.time()
        if all(self._stale(st, now) for st in self.states.values()):   # no video: nothing is live
            s["site"] = {"unique_people": 0, "camera_sum": 0, "zones": {}, "vehicles": 0, "offline": True}
        else:
            s["site"] = dict(self._site_stats)
        return s
