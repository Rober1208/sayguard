"""The multi-camera loop: one batched detector call serves every camera that has a
new frame, then per camera: track -> rules -> alerts -> annotated tile.

Batching across cameras is what lets one accelerator serve many streams; it is
also how the UGen300 would be fed (HailoRT batch 8: 252 FPS for YOLOv8s).  So the
loop runs on one global tick at the target fps and takes every camera with a new
frame into that tick's batch: cameras never drift into separate, smaller batches.

Everything slow or fallible sits off this loop or behind its own guard:
YOLO-World/CLIP load in a background thread (object rules wait, hard hats do not),
an open-vocabulary error only pauses object rules for a while, one camera's or
one alert's failure stays with that camera or alert, and the retention purge runs
in its own short-lived thread."""
from __future__ import annotations

import gc
import logging
import math
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .annotate import draw
from .config import resolve
from .detectors import make_detector
from .events import EventStore
from .llm import LLMClient
from .privacy import FaceHold, anonymise
from .rules import CameraInfo, Rule, RuleEngine
from .rules.engine import Fired
from .rules.model import label_matches
from .sources import VideoSource
from .tracking import CameraTracker
from .types import Detection, Event, FramePacket

log = logging.getLogger("sayguard.pipeline")

OV_RETRY_S = 120.0          # after a failed YOLO-World load / class encoding (memory refusal, OOM, I/O)
OV_INFER_RETRY_S = 60.0     # after a failed open-vocabulary inference
OV_PERMANENT = (ImportError, FileNotFoundError)   # will not fix itself: needs an install or the weights
RETENTION_CHECK_S = 3600.0  # how often old alerts and snapshots are purged
LOG_EVERY_S = 30.0          # a failure repeating every frame is logged at most this often


def source_arg(src: str | int) -> str | int:
    """Camera indexes and URLs (rtsp://, http://, ...) pass through unchanged; only
    local file paths are resolved ('@work/...' or repo-relative)."""
    s = str(src)
    if isinstance(src, int) or s.isdigit() or "://" in s:
        return src
    return str(resolve(s))


def make_pipeline(cfg: dict) -> Pipeline:
    """Site mode (calibrated cameras on a floor plan) when the config has a `site:` block."""
    if cfg.get("site"):
        from .site_pipeline import SitePipeline
        return SitePipeline(cfg)
    return Pipeline(cfg)


def _release_partial() -> None:
    """Return memory a failed model load left behind (never imports torch just for this)."""
    torch = sys.modules.get("torch")
    try:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class CamState:
    info: CameraInfo
    source: VideoSource
    tracker: CameraTracker
    title: str
    jpeg: bytes | None = None
    jpeg_seq: int = 0
    fps: float = 0.0
    last_idx: int = -1
    last_proc: float = 0.0
    n_proc: int = 0
    ov_last: list[Detection] = field(default_factory=list)
    ov_ts: float = 0.0
    people: int = 0
    no_helmet: int = 0
    face_hold: FaceHold = field(default_factory=FaceHold)   # keeps faces masked through detector misses
    blur_now: list = field(default_factory=list)            # site mode: this frame's boxes to pixelate


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fps_target = float(cfg["streams"]["fps"])
        self.stale_s = max(2.0, 3.0 / self.fps_target)   # no frame processed this long -> camera offline
        self.tile_w = int(cfg["streams"].get("tile_width", 640))
        self.blur = bool(cfg.get("privacy", {}).get("blur_faces", True))
        rd = cfg.get("retention_days", 14)
        self.retention_days = float(rd) if rd else 0.0    # 0 / null keeps everything
        self.zones = cfg.get("zones", {})
        self.cam_cfg = cfg["cameras"]
        self.cam_infos = [CameraInfo(c["id"], c["name"], c["zone"]) for c in self.cam_cfg]
        self.engine = RuleEngine(self.cam_infos, resolve(cfg["rules_file"]))
        self.engine.load(cfg.get("default_rules"))
        self.store = EventStore(resolve(cfg["events_db"]), resolve(cfg["snapshots_dir"]))
        self.detector = make_detector(cfg)
        from .verify import Verifier, VlmBackend, make_backend
        vcfg = cfg.get("verifier") or cfg.get("vlm", {})
        v_enabled = vcfg.get("enabled", True)
        v_backend = make_backend(vcfg, resolve)         # cheap: nothing is loaded until the first alert
        # transformers is only needed by Whisper and the VLM backend (the CLIP probe uses torch + clip)
        if (v_enabled and isinstance(v_backend, VlmBackend)) or cfg.get("asr"):
            from .resources import preload_model_libs
            err = preload_model_libs()   # main thread, before the VLM worker or an ASR request can race
            if err:
                log.warning("transformers not available (VLM/ASR will report it): %s", err)
        self.llm = LLMClient(cfg["llm"]["base_url"], cfg["llm"]["model"], cfg["llm"].get("timeout_s", 60))
        # open vocabulary: built and re-classed by a background thread, read by the loop
        self._ov = None
        self._ov_enabled = bool(cfg.get("open_vocab", {}).get("enabled", True))
        self._ov_error: str | None = None
        self._ov_retry_at = 0.0           # time.time() before which no load/inference is tried; inf = never
        self._ov_thread: threading.Thread | None = None
        self._ov_lock = threading.Lock()
        self.states: dict[str, CamState] = {}
        for c, info in zip(self.cam_cfg, self.cam_infos):
            zname = self.zones.get(info.zone, {}).get("en", info.zone)
            self.states[info.id] = CamState(info=info, source=self._make_source(c), tracker=self._make_tracker(),
                                            title=f"{info.id.upper()}  {info.name} | {zname}")
        self.verifier = Verifier(v_backend, self.store, enabled=v_enabled)
        self.stats = {"frames_per_s": 0.0, "det_batch_ms": 0.0, "det_capacity_fps": 0.0,
                      "batch_size": 0.0, "ov_ms": 0.0, "started": time.time()}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)
        self._win = deque(maxlen=200)   # (time, frames) for throughput
        self._next_purge = 0.0          # first retention pass on the first tick
        self._purge_thread: threading.Thread | None = None
        self._last_log: dict[str, float] = {}
        self.cond = threading.Condition()

    def _make_source(self, cam_cfg: dict):
        return VideoSource(cam_cfg["id"], source_arg(cam_cfg["source"]))

    def _make_tracker(self) -> CameraTracker:
        return CameraTracker(self.fps_target)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        from .resources import keep_full_speed
        if keep_full_speed():     # Windows: never moved onto the efficient cores as "background work"
            log.info("Windows power throttling is off for this process (HighQoS)")
        for st in self.states.values():
            st.source.start()
        self.verifier.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for st in self.states.values():
            st.source.stop()
        self.verifier.stop()

    def _log_failure(self, key: str, msg: str, *args) -> None:
        """log.exception, at most once per LOG_EVERY_S per key (call inside an except block)."""
        now = time.time()
        if now - self._last_log.get(key, 0.0) >= LOG_EVERY_S:
            self._last_log[key] = now
            log.exception(msg, *args)

    # ------------------------------------------------------- open vocabulary
    def _ov_busy(self) -> bool:
        return self._ov_thread is not None and self._ov_thread.is_alive()

    def _ov_for(self, wanted: set[str]):
        """The open-vocabulary detector if it is ready for exactly `wanted`, else None.

        Never blocks.  Building YOLO-World and CLIP-encoding the class names take
        seconds (CLIP is rebuilt on every class change), so that runs in a background thread,
        started here whenever the class set is not the loaded one, at most one at a
        time.  Until it finishes the cameras run on without object detections."""
        if not wanted or not self._ov_enabled:
            return None
        with self._ov_lock:
            if self._ov_busy() or time.time() < self._ov_retry_at:
                return None
            ov = self._ov
        if ov is not None and ov.ready_for(wanted):
            return ov
        t = threading.Thread(target=self._ov_prepare, args=(set(wanted),), name="open-vocab", daemon=True)
        with self._ov_lock:
            self._ov_thread = t
        t.start()
        return None

    def _ov_prepare(self, names: set[str]) -> None:
        """Background worker: build YOLO-World on first use, then encode `names`."""
        fail: tuple[str, bool] | None = None
        try:
            ov = self._ov if self._ov is not None else self._build_ov()
            ov.set_classes(names)
            if not ov.ready_for(names):   # would otherwise restart this worker on every tick
                raise RuntimeError(f"class set {sorted(names)} did not load")
        except Exception as e:  # no memory, missing weights/CLIP, CUDA trouble: object rules wait
            fail = (f"{type(e).__name__}: {e}"[:300], isinstance(e, OV_PERMANENT))
            log.warning("open-vocabulary detector could not be prepared", exc_info=True)
        if fail is None:
            with self._ov_lock:
                self._ov_error, self._ov_retry_at = None, 0.0
            log.info("open-vocabulary detector ready for: %s", ", ".join(sorted(names)))
            return
        self._ov_failed(*fail, retry_s=OV_RETRY_S)
        # outside the except block, so its traceback no longer pins a half-built model
        _release_partial()

    def _build_ov(self):
        from .detectors.open_vocab import OpenVocabDetector
        from .resources import require

        oc = self.cfg.get("open_vocab", {})
        require("YOLO-World", gpu_gb=0.8, ram_gb=2.1)   # incl. ~0.6 GB for the detector's host copy of CLIP
        ov = OpenVocabDetector(str(resolve("@work/models/" + oc.get("weights", "yolov8s-worldv2.pt"))),
                               conf=oc.get("conf", 0.2))
        with self._ov_lock:
            self._ov = ov
        return ov

    def _ov_failed(self, msg: str, permanent: bool, retry_s: float) -> None:
        with self._ov_lock:
            self._ov_error = msg
            self._ov_retry_at = math.inf if permanent else time.time() + retry_s
        log.warning("object rules paused: %s (%s)", msg,
                    "fix it and restart" if permanent else f"retrying in {retry_s:.0f} s")

    def _run_open_vocab(self, due: list[tuple[CamState, FramePacket]]) -> None:
        """Object detections for the cameras that are due this frame and have an
        object rule.  A failure clears their objects and pauses object rules for
        OV_INFER_RETRY_S; the hard-hat path is not affected."""
        ov = self._ov_for(self.engine.all_objects())   # also (re)starts the background prepare
        if ov is None:
            return
        jobs = [(st, p, objs) for st, p in due if (objs := self.engine.objects_for_camera(st.info.id))]
        if not jobs:
            return
        t1 = time.perf_counter()
        try:
            res = ov.infer([p.frame for _, p, _ in jobs])
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:300]
            log.warning("open-vocabulary inference failed", exc_info=True)
            for st, _, _ in jobs:
                st.ov_last = []
            self._ov_failed(msg, permanent=False, retry_s=OV_INFER_RETRY_S)
            return
        self.stats["ov_ms"] = 0.8 * self.stats["ov_ms"] + 0.2 * (time.perf_counter() - t1) * 1000
        for (st, p, objs), r in zip(jobs, res):
            st.ov_last, st.ov_ts = [d for d in r if any(label_matches(d.label, o) for o in objs)], p.ts
        if self._ov_error is not None:
            with self._ov_lock:
                self._ov_error = None

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        period = 1.0 / self.fps_target
        every_n = max(1, int(self.cfg.get("open_vocab", {}).get("every_n_frames", 3)))
        next_tick = time.time()
        while not self._stop.is_set():
            next_tick += period
            delay = next_tick - time.time()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period:      # a slow batch left us over a period behind: restart the grid
                next_tick = time.time()
            if self._stop.is_set():
                break
            tick = next_tick
            self._maybe_purge(tick)
            batch: list[tuple[CamState, FramePacket]] = []
            for st in self.states.values():
                pkt = st.source.latest()
                if pkt is not None and pkt.index != st.last_idx:
                    batch.append((st, pkt))
            if not batch:
                continue
            try:
                self._process(batch, tick, every_n)
            except Exception:
                self._log_failure("step", "pipeline step failed")

    def _process(self, batch: list[tuple[CamState, FramePacket]], tick: float, every_n: int) -> None:
        t0 = time.perf_counter()
        dets = self.detector.infer([p.frame for _, p in batch])
        t_det = time.perf_counter() - t0
        ov_due = self._schedule(batch, tick, every_n)

        # open-vocabulary objects only where a rule needs them, every Nth frame
        try:
            self._run_open_vocab(ov_due)
        except Exception:
            self._log_failure("ov", "open-vocabulary step failed")

        rule_types = {r.name: r.type for r in self.engine.list()}
        for (st, pkt), d in zip(batch, dets):
            try:
                self._process_camera(st, pkt, d, rule_types)
            except Exception:
                self._log_failure(f"cam-{st.info.id}", "camera %s: frame failed", st.info.id)
        with self.cond:
            self.cond.notify_all()
        self._record_throughput(len(batch), t_det)

    def _schedule(self, batch: list[tuple[CamState, FramePacket]], tick: float,
                  every_n: int) -> list[tuple[CamState, FramePacket]]:
        """Per-camera rate and frame bookkeeping, done before anything that can fail so no error
        can freeze a camera's counters; returns the cameras due for open-vocabulary detection."""
        ov_due: list[tuple[CamState, FramePacket]] = []
        for st, pkt in batch:
            gap = tick - st.last_proc
            if st.last_proc and gap <= self.stale_s:
                inst = 1.0 / max(1e-3, gap)
                st.fps = inst if st.fps <= 0 else 0.85 * st.fps + 0.15 * inst
            else:
                st.fps = 0.0           # first frame, or back from an outage: measure afresh
            st.last_idx, st.last_proc = pkt.index, tick
            if st.n_proc % every_n == 0:
                ov_due.append((st, pkt))
            st.n_proc += 1
        return ov_due

    def _record_throughput(self, n: int, t_det: float) -> None:
        now = time.time()
        self._win.append((now, n))
        while self._win and now - self._win[0][0] > 2.0:
            self._win.popleft()
        s = self.stats
        # frames per second over the window: the batches after the oldest one, over the time since it
        # (counting the oldest batch too divided N batches by N-1 intervals: 105 for 10 x 10 fps)
        span = now - self._win[0][0] if len(self._win) > 1 else 0.0
        s["frames_per_s"] = sum(k for _, k in list(self._win)[1:]) / span if span > 0 else 0.0
        s["det_batch_ms"] = 0.9 * s["det_batch_ms"] + 0.1 * t_det * 1000
        s["batch_size"] = 0.9 * s["batch_size"] + 0.1 * n
        s["det_capacity_fps"] = s["batch_size"] / max(1e-4, s["det_batch_ms"] / 1000)

    def _process_camera(self, st: CamState, pkt: FramePacket, d: list[Detection], rule_types: dict[str, str]) -> None:
        people, heads = st.tracker.update(d, pkt.ts)
        st.people = len(people)
        st.no_helmet = sum(1 for h in heads if h.label == "head")
        objects = st.ov_last if pkt.ts - st.ov_ts < 1.5 else []
        H, W = pkt.frame.shape[:2]
        frame_dt = 1.0 / st.fps if st.fps > 0.5 else None    # None until this camera's rate is known
        res = self.engine.evaluate(st.info.id, pkt.ts, people, objects, (W, H), frame_dt=frame_dt)
        # anonymise every head the detector saw in this frame (the tracker leaves out low-confidence
        # and unmatched boxes, and those faces must not be shown either), plus where a face was in
        # the last second: a detector miss of a few frames must not show a face
        blur_boxes = st.face_hold.update(pkt.ts, [(x.xyxy, x.label == "helmet") for x in d
                                                  if x.label in ("helmet", "head")])
        for f in res.fired:
            try:
                self._emit(st, pkt, f, heads, objects, blur_boxes)
            except Exception:
                self._log_failure(f"emit-{st.info.id}", "camera %s: could not record alert %r", st.info.id, f.rule.name)
        tile = draw(pkt.frame, st.title, heads, objects, res.active, rule_types, res.links, st.fps,
                    blur=self.blur, out_width=self.tile_w, blur_boxes=blur_boxes)
        ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            st.jpeg, st.jpeg_seq = buf.tobytes(), st.jpeg_seq + 1

    def _emit(self, st: CamState, pkt: FramePacket, f: Fired, heads: list[Detection], objects: list[Detection],
              blur_boxes: list[tuple[tuple, bool]]) -> None:
        rule: Rule = f.rule
        ev = Event(id=uuid.uuid4().hex[:12], ts=pkt.ts, cam_id=st.info.id, cam_name=st.info.name,
                   zone=st.info.zone, rule_id=rule.id, rule_name=rule.name, rule_type=rule.type,
                   severity=rule.severity, message=rule.message or rule.name, track_id=f.track_id,
                   box=f.box, extra={"detail": f.detail, "source_text": rule.source_text})
        frame = pkt.frame
        H, W = frame.shape[:2]
        # raw crop for the double-check (kept in memory only, never written to disk)
        crop = None
        if rule.type == "no_helmet" and f.track_id is not None:
            head = next((h for h in heads if h.track_id == f.track_id), None)
            if head is not None:
                x1, y1, x2, y2 = head.xyxy
                w, h = x2 - x1, y2 - y1
                cx1, cy1 = int(max(0, x1 - 1.0 * w)), int(max(0, y1 - 0.6 * h))
                cx2, cy2 = int(min(W, x2 + 1.0 * w)), int(min(H, y2 + 2.0 * h))
                if cx2 - cx1 > 8 and cy2 - cy1 > 8:
                    crop = frame[cy1:cy2, cx1:cx2].copy()
        ev.snapshot = self._snapshot(ev, frame, f, objects, blur_boxes)
        self.store.add(ev)
        log.info("ALERT %s %s %s", st.info.id, rule.name, f.detail)
        if crop is not None:
            self.verifier.submit(ev, crop)

    def _snapshot(self, ev: Event, frame: np.ndarray, f: Fired, objects: list[Detection],
                  blur_boxes: list[tuple[tuple, bool]]) -> str | None:
        """Anonymised 640-px JPEG of the alert; None if it cannot be written (the alert
        itself is still recorded: losing the picture must not lose the event)."""
        try:
            H, W = frame.shape[:2]
            snap = frame.copy()
            if self.blur:
                anonymise(snap, blur_boxes)
            if f.box:
                cv2.rectangle(snap, f.box[:2], f.box[2:], (40, 40, 235), max(2, W // 320))
            for o in objects:
                cv2.rectangle(snap, tuple(map(int, o.xyxy[:2])), tuple(map(int, o.xyxy[2:])), (0, 160, 255),
                              max(2, W // 320))
            snap = cv2.resize(snap, (640, int(H * 640 / W)), interpolation=cv2.INTER_AREA)
            return self.store.save_snapshot(ev, snap)
        except Exception:
            self._log_failure("snapshot", "could not save the snapshot for alert %s", ev.id)
            return None

    # ------------------------------------------------------------- retention
    def _maybe_purge(self, now: float) -> None:
        """Hourly: drop alerts and snapshots older than retention_days.  The check is
        a comparison; the delete runs in its own thread, so a large first purge
        (thousands of files) does not stall the cameras."""
        if now < self._next_purge:
            return
        self._next_purge = now + RETENTION_CHECK_S
        if self.retention_days <= 0 or (self._purge_thread is not None and self._purge_thread.is_alive()):
            return
        cutoff = now - self.retention_days * 86400
        self._purge_thread = threading.Thread(target=self._purge, args=(cutoff,), name="retention", daemon=True)
        self._purge_thread.start()

    def _purge(self, cutoff: float) -> None:
        try:
            n = self.store.purge_older_than(cutoff)
        except Exception:
            log.exception("retention pass failed")
            return
        if n:
            log.info("retention: removed %d alerts older than %g days", n, self.retention_days)

    # ---------------------------------------------------------------- status
    def _stale(self, st: CamState, now: float) -> bool:
        """No frame of this camera processed recently (source down, or not started)."""
        return now - st.last_proc > self.stale_s

    def hailo_estimate(self) -> dict:
        """Demand against Hailo's published YOLOv8s speeds.  Those are for a 640 px input; a detector
        run at another size (warehouse mode: 960 px) has no reference, so its load is not estimated."""
        ref = self.cfg.get("hailo_reference", {})
        now = time.time()
        demand = sum(st.fps for st in self.states.values() if not self._stale(st, now))
        b1, b8 = float(ref.get("fps_batch1", 166)), float(ref.get("fps_batch8", 252))
        imgsz, ref_imgsz = int(self.cfg.get("detector", {}).get("imgsz", 640)), int(ref.get("imgsz", 640))
        comparable = imgsz == ref_imgsz
        return {"model": ref.get("model", "yolov8s"), "demand_fps": round(demand, 1),
                "capacity_batch1": b1, "capacity_batch8": b8, "comparable": comparable,
                "imgsz": imgsz, "reference_imgsz": ref_imgsz,
                "load_batch8_pct": round(100 * demand / b8, 1) if comparable else None,
                "load_batch1_pct": round(100 * demand / b1, 1) if comparable else None}

    def _camera_status(self, st: CamState, now: float) -> dict:
        if self._stale(st, now):    # frozen numbers from an offline camera would look live
            return {"fps": 0.0, "online": False, "people": 0, "no_helmet": 0}
        return {"fps": round(st.fps, 1), "online": st.source.online, "people": st.people, "no_helmet": st.no_helmet}

    def _ov_status(self, now: float) -> dict:
        wanted = self.engine.all_objects()
        with self._ov_lock:
            ov, err, retry_at, busy = self._ov, self._ov_error, self._ov_retry_at, self._ov_busy()
        if not self._ov_enabled:
            err, retry_at = "disabled in config", math.inf
        ready = bool(wanted) and ov is not None and not busy and ov.ready_for(wanted)
        if not wanted or (busy and self._ov_enabled):
            err = None   # nothing asks for objects, or a retry is loading right now
        return {"loaded": ready, "busy": busy, "error": err,
                "retry_in_s": round(max(0.0, retry_at - now)) if err and retry_at != math.inf else None,
                "classes": sorted(wanted), "ms": round(self.stats["ov_ms"], 1)}

    def status(self) -> dict:
        now = time.time()
        c = self.store.counts(now - 24 * 3600, now + 1)
        return {
            "detector": self.detector.name,
            "stats": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.stats.items()},
            "cameras": {cid: self._camera_status(st, now) for cid, st in self.states.items()},
            "hailo": self.hailo_estimate(),
            "rules_version": self.engine.version,
            "events_version": self.store.version,
            # unchecked = hard-hat alerts still waiting for the double-check; other rule types are
            # never double-checked and are counted separately, not as "pending"
            "alerts_24h": {"total": c["total"], "confirmed": c["confirmed"], "false_alarms": c["false_alarms"],
                           "unchecked": c["pending"], "not_applicable": c["not_applicable"]},
            "open_vocab": self._ov_status(now),
            "verifier": self.verifier.status(),
        }
