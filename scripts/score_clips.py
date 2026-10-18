"""Rank the downloaded stock clips and pick ten of them as the demo's "CCTV cameras".

Stock footage is shot for advertising: hand-held pans, close-ups, and few bare
heads.  A convincing PPE demo needs the opposite: static, CCTV-like shots with
several small-to-medium heads, some of them without a hard hat, and a forklift
for the spoken-rule demo.  So every clip is run through the same detectors the
demo uses (the fine-tuned helmet/head YOLOv8s and YOLO-World for "forklift")
on ~1 frame per second, and scored on what the dashboard will actually show.

Clips longer than --max-seconds are sampled at a coarser stride so the per-clip
cost stays bounded and the best-60-s-window search still sees the whole clip.

    python scripts/score_clips.py --dry-run
    python scripts/score_clips.py
    python scripts/score_clips.py --select-only     # re-pick from clip_scores.json, no GPU
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import load_config, resolve, work_dir  # noqa: E402

OV_CLASSES = ["forklift", "person"]
GRAY_W = 160                 # motion is measured on tiny grey frames: cheap and noise-tolerant
MOTION_REF = 12.0            # mean |diff| (0-255) at 1 s lag of a slow pan; static shots sit at ~1-4
TINY_HEAD = 0.025            # head box below 2.5 % of frame height: too small to read on a tile
CLOSE_UP = 0.35              # head box above 35 % of frame height: a portrait, not CCTV
HEAD_MIN_FRAC = 0.10         # a clip "has violations" if bare heads show in >= 10 % of frames
FORKLIFT_MIN_FRAC = 0.10
STOPWORDS = {"a", "an", "the", "in", "at", "of", "on", "his", "her", "their", "and", "with",
             "near", "to", "for", "seen", "up", "is", "are", "while", "from", "by"}
# Title hints that make a clip a natural fit for a camera zone in configs/site.yaml.
ZONE_HINTS = {
    "office": r"office|architect|meeting|business|desk|laptop|plan",
    "warehouse": r"warehouse|forklift|inventory|logistic|storage|dock|box|pallet|shelf",
    "workshop": r"workshop|lathe|weld|machine|engineer|carpenter|factory|industr|repair|metal",
    "entrance": r"gate|entrance|walk|arriv|street|road|phone",
    "zone_a": r"scaffold|construction|build|house|site|crane|roof",
    "zone_b": r"construction|concrete|rebar|brick|site|cement",
}
ZONE_ORDER = ["office", "warehouse", "workshop", "entrance", "zone_a", "zone_b"]


# ---------------------------------------------------------------- clips + frames

def load_manifest(raw: Path) -> dict[str, dict]:
    p = raw / "manifest.json"
    if not p.exists():
        return {}
    return {str(x["id"]): x for x in json.loads(p.read_text(encoding="utf-8"))}


def clip_id(path: Path) -> str:
    m = re.search(r"(\d+)$", path.stem)
    return m.group(1) if m else path.stem


def list_clips(raw: Path, manifest: dict[str, dict]) -> list[dict]:
    clips = []
    for p in sorted(raw.glob("*.mp4")):
        cid = clip_id(p)
        meta = manifest.get(cid, {})
        clips.append({"id": cid, "file": p.name, "path": p,
                      "title": (meta.get("title") or p.stem).strip(),
                      "category": meta.get("category", ""), "page": meta.get("page", ""),
                      "license": meta.get("license", "")})
    return clips


def clip_meta(cap: cv2.VideoCapture) -> tuple[float, int, int, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (fps if 1 <= fps <= 240 else 30.0), n, w, h


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        fps, n, w, h = clip_meta(cap)
        return {"ok": cap.isOpened() and n > 0, "fps": fps, "frames_total": n,
                "duration_s": round(n / fps, 2) if n else 0.0, "width": w, "height": h}
    finally:
        cap.release()


def sample_plan(n_frames: int, fps: float, max_samples: int) -> tuple[list[int], int]:
    """Frame indices to analyse (1/s, or evenly spread if the clip is longer than
    max_samples seconds) and the motion lag in frames (1 s)."""
    dur = n_frames / fps
    step_s = max(1.0, dur / max_samples)
    n = max(1, min(max_samples, int(dur / step_s)))
    idx = sorted({min(n_frames - 1, int(round(k * step_s * fps))) for k in range(n)})
    return idx, max(1, int(round(fps)))


def small_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    g = cv2.cvtColor(cv2.resize(frame, (GRAY_W, max(1, round(h * GRAY_W / w))),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0)


def iter_samples(path: Path, max_samples: int) -> Iterator[tuple[float, np.ndarray, float | None]]:
    """Yield (t_seconds, BGR frame, motion vs. the frame 1 s earlier) by reading
    sequentially; seeking in H.264 is slow and imprecise with OpenCV."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps, n, _, _ = clip_meta(cap)
        if not cap.isOpened() or n <= 0:
            return
        wanted, lag = sample_plan(n, fps, max_samples)
        want, refs = set(wanted), {i - lag for i in wanted if i >= lag}
        grays: dict[int, np.ndarray] = {}
        for idx in range(wanted[-1] + 1):
            if not cap.grab():
                break
            if idx not in want and idx not in refs:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            g = small_gray(frame)
            if idx in refs:
                grays[idx] = g
            if idx in want:
                ref = grays.pop(idx - lag, None)
                motion = float(np.mean(cv2.absdiff(g, ref))) if ref is not None else None
                yield idx / fps, frame, motion
    finally:
        cap.release()


# ---------------------------------------------------------------- detection

class Detectors:
    """The demo's two detectors, loaded once.  FP16 on CUDA, batch inference."""

    def __init__(self, weights: Path, world: Path | None, imgsz: int, conf: float, ov_conf: float,
                 fp32: bool = False):
        import torch
        from ultralytics import YOLO, YOLOWorld

        self.cuda = torch.cuda.is_available()
        self.device = 0 if self.cuda else "cpu"
        self.quantize = 16 if self.cuda and not fp32 else None
        self.imgsz, self.conf, self.ov_conf = imgsz, conf, ov_conf
        self.det = YOLO(str(weights))
        self.world = None
        if world is not None:
            self.world = YOLOWorld(str(world))
            self.world.set_classes(OV_CLASSES)

    def _predict(self, model, frames: list[np.ndarray], conf: float) -> list:
        return model.predict(frames, imgsz=self.imgsz, conf=conf, device=self.device,
                             quantize=self.quantize, verbose=False)

    def count(self, frames: list[np.ndarray]) -> list[dict]:
        out = []
        for r in self._predict(self.det, frames, self.conf):
            labels = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
            xyxy = r.boxes.xyxy.cpu().numpy()
            fh = float(r.orig_shape[0])
            out.append({"helmets": labels.count("helmet"), "heads": labels.count("head"),
                        "head_h": [float(b[3] - b[1]) for b in xyxy], "frame_h": fh,
                        "forklifts": 0, "persons": 0})
        if self.world is not None:
            for rec, r in zip(out, self._predict(self.world, frames, self.ov_conf)):
                labels = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
                rec["forklifts"], rec["persons"] = labels.count("forklift"), labels.count("person")
        return out

    def close(self) -> None:
        self.det = self.world = None
        if self.cuda:
            import torch
            torch.cuda.empty_cache()


def analyse_clip(clip: dict, dets: Detectors, max_samples: int, batch: int) -> dict:
    samples: list[dict] = []
    motions: list[float] = []
    frames: list[np.ndarray] = []
    times: list[float] = []

    def flush() -> None:
        for t, rec in zip(times, dets.count(frames)):
            samples.append({"t": round(t, 2), **rec})
        frames.clear()
        times.clear()

    for t, frame, motion in iter_samples(clip["path"], max_samples):
        frames.append(frame)
        times.append(t)
        if motion is not None:
            motions.append(motion)
        if len(frames) >= batch:
            flush()
    if frames:
        flush()
    return summarize(clip, probe(clip["path"]), samples, motions)


# ---------------------------------------------------------------- scoring

def summarize(clip: dict, meta: dict, samples: list[dict], motions: list[float]) -> dict:
    n = len(samples)
    heights_px = [h for s in samples for h in s["head_h"]]
    heights_rel = [h / s["frame_h"] for s in samples for h in s["head_h"] if s["frame_h"]]

    def frac(key_fn) -> float:
        return round(sum(1 for s in samples if key_fn(s)) / n, 3) if n else 0.0

    def mean(key: str) -> float:
        return round(sum(s[key] for s in samples) / n, 3) if n else 0.0

    out = {
        "id": clip["id"], "file": clip["file"], "title": clip["title"], "category": clip["category"],
        "page": clip["page"], "duration_s": meta["duration_s"], "fps": round(meta["fps"], 2),
        "width": meta["width"], "height": meta["height"], "frames": n,
        "mean_helmets": mean("helmets"), "mean_heads": mean("heads"),
        "head_frac": frac(lambda s: s["heads"] > 0),
        "people_frac": frac(lambda s: s["helmets"] + s["heads"] > 0),
        "ov_person_frac": frac(lambda s: s["persons"] > 0),
        "forklift_frac": frac(lambda s: s["forklifts"] > 0),
        "median_head_h_px": round(statistics.median(heights_px), 1) if heights_px else None,
        "median_head_h_rel": round(statistics.median(heights_rel), 4) if heights_rel else None,
        "motion": round(sum(motions) / len(motions), 2) if motions else None,
        "samples": [[s["t"], s["helmets"], s["heads"], s["forklifts"]] for s in samples],
    }
    out["score"] = score_clip(out)
    return out


def score_clip(s: dict) -> float:
    """Higher = better demo camera.  Presence and density of heads first, bare heads
    (violations) as a bonus, static camera as a bonus; tiny heads, portraits and
    empty scenes are penalised."""
    if not s["frames"]:
        return -9.0
    density = min(s["mean_helmets"] + s["mean_heads"], 4.0) / 4.0
    static = 0.5 if s["motion"] is None else max(0.0, 1.0 - s["motion"] / MOTION_REF)
    score = 1.0 * s["people_frac"] + 0.5 * density + 0.8 * s["head_frac"] + 0.5 * static
    if s["people_frac"] < 0.1:
        score -= 1.0
    h = s["median_head_h_rel"]
    if h is not None and h < TINY_HEAD:
        score -= 0.5
    if h is not None and h > CLOSE_UP:
        score -= 0.3
    return round(score, 3)


def best_window(samples: list[list], duration: float, window: float = 60.0) -> float:
    """Start time of the `window`-second span with the most people per sample."""
    if duration <= window or not samples:
        return 0.0
    times = [s[0] for s in samples]
    people = [s[1] + s[2] for s in samples]
    starts = [t for t in times if t + window <= duration] + [duration - window]
    best_t, best = 0.0, -1.0
    for t0 in starts:
        vals = [p for t, p in zip(times, people) if t0 <= t < t0 + window]
        m = sum(vals) / len(vals) if vals else 0.0
        if m > best + 1e-9:
            best_t, best = t0, m
    return round(max(0.0, best_t), 1)


# ---------------------------------------------------------------- selection

def title_tokens(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in STOPWORDS}


def title_sim(a: str, b: str) -> float:
    ta, tb = title_tokens(a), title_tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def is_violation(c: dict) -> bool:
    return c["head_frac"] >= HEAD_MIN_FRAC


def choose(scored: list[dict], n: int, min_violation: int = 3, max_sim: float = 0.5) -> list[tuple[dict, str]]:
    """>= min_violation clips with bare heads, >= 1 forklift clip if any exists, then
    the best remaining scores; near-duplicate titles are only used as a last resort."""
    pool = sorted((c for c in scored if c["frames"] > 0), key=lambda c: -c["score"])
    picked: list[tuple[dict, str]] = []

    def taken(c: dict) -> bool:
        return any(p["id"] == c["id"] for p, _ in picked)

    def diverse(c: dict) -> bool:
        return all(title_sim(c["title"], p["title"]) < max_sim for p, _ in picked)

    def take(cands: list[dict], k: int, why) -> None:
        for strict in (True, False):
            for c in cands:
                if k <= 0 or len(picked) >= n:
                    return
                if not taken(c) and (diverse(c) or not strict):
                    picked.append((c, why(c)))
                    k -= 1

    take([c for c in pool if is_violation(c)], min_violation,
         lambda c: f"violations: bare heads in {c['head_frac']:.0%} of frames (score {c['score']})")
    if not any(p["forklift_frac"] >= FORKLIFT_MIN_FRAC for p, _ in picked):
        take([c for c in pool if c["forklift_frac"] >= FORKLIFT_MIN_FRAC], 1,
             lambda c: f"forklift in {c['forklift_frac']:.0%} of frames (score {c['score']})")
    take(pool, n - len(picked), lambda c: f"score {c['score']}")
    base = list(picked)
    for i in range(n - len(picked) if base else 0):     # fewer usable clips than cameras: reuse
        picked.append((base[i % len(base)][0], f"reused (only {len(pool)} usable clips)"))
    return picked


def assign_cameras(picked: list[tuple[dict, str]], cameras: list[dict], n: int) -> list[tuple[dict, dict, str]]:
    """Give zone-specific cameras a clip whose title fits (forklift -> warehouse);
    the office camera (excluded from the hard-hat rule) gets a non-violation clip."""
    cams = cameras[:n] if len(cameras) >= n else [{"id": f"cam{i + 1:02d}", "zone": "", "name": ""}
                                                   for i in range(n)]
    left = list(range(len(picked)))
    out: dict[str, tuple[dict, dict, str]] = {}
    order = sorted(cams, key=lambda c: ZONE_ORDER.index(c["zone"]) if c["zone"] in ZONE_ORDER else 99)
    for cam in order:
        if not left:
            break
        hint = re.compile(ZONE_HINTS.get(cam["zone"], r"(?!x)x"), re.I)

        def key(i: int) -> tuple:
            c = picked[i][0]
            match = bool(hint.search(c["title"] + " " + c.get("category", "")))
            if cam["zone"] == "office":
                return (not is_violation(c), match, c["score"])
            return (match, c["score"])

        best = max(left, key=key)
        left.remove(best)
        out[cam["id"]] = (cam, *picked[best])
    return [out[c["id"]] for c in cams if c["id"] in out]


def selection_rows(assigned: list[tuple[dict, dict, str]], window: float) -> list[dict]:
    return [{"cam": cam["id"], "clip_id": c["id"], "file": c["file"], "title": c["title"],
             "start_s": best_window(c["samples"], c["duration_s"], window), "reason": why,
             "camera_name": cam.get("name", ""), "zone": cam.get("zone", ""),
             "duration_s": c["duration_s"], "score": c["score"], "page": c["page"]}
            for cam, c, why in assigned]


# ---------------------------------------------------------------- output

def fmt(v, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def print_scores(scored: list[dict]) -> None:
    print(f"{'#':>2} {'id':>6} {'score':>6} {'n':>3} {'helm/f':>6} {'head/f':>6} {'head%':>5} "
          f"{'headH%':>6} {'fork%':>5} {'motion':>6} {'dur':>5}  title")
    for i, c in enumerate(scored, 1):
        rel = None if c["median_head_h_rel"] is None else 100 * c["median_head_h_rel"]
        print(f"{i:>2} {c['id']:>6} {c['score']:>6.2f} {c['frames']:>3} {c['mean_helmets']:>6.2f} "
              f"{c['mean_heads']:>6.2f} {100 * c['head_frac']:>5.0f} {fmt(rel, '6.1f'):>6} "
              f"{100 * c['forklift_frac']:>5.0f} {fmt(c['motion'], '6.1f'):>6} {c['duration_s']:>5.1f}  "
              f"{c['title'][:48]}")


def print_selection(rows: list[dict]) -> None:
    print(f"\n{'cam':<6}{'clip':>7} {'start':>6}  {'camera':<16} title / reason")
    for r in rows:
        print(f"{r['cam']:<6}{r['clip_id']:>7} {r['start_s']:>6.1f}  {r['camera_name'][:16]:<16} "
              f"{r['title'][:40]}  [{r['reason']}]")


def dry_run(clips: list[dict]) -> None:
    print(f"{len(clips)} clips")
    for c in clips:
        m = probe(c["path"])
        flag = "" if m["ok"] else "  UNREADABLE"
        print(f"{c['id']:>6} {m['duration_s']:>6.1f}s {m['width']:>4}x{m['height']:<4} {m['fps']:>5.1f}fps  "
              f"{c['category']:<13} {c['title'][:60]}{flag}")


def score_all(clips: list[dict], a: argparse.Namespace) -> list[dict]:
    if not a.skip_resource_check:
        from sayguard.resources import require
        require("score_clips (YOLOv8s + YOLO-World)", gpu_gb=1.0, ram_gb=2.0)
    world = None if a.no_world else resolve(a.world)
    dets = Detectors(resolve(a.weights), world, a.imgsz, a.conf, a.ov_conf, a.fp32)
    scored = []
    try:
        for i, clip in enumerate(clips, 1):
            try:
                s = analyse_clip(clip, dets, a.max_seconds, a.batch)
            except Exception as e:  # one broken file must not lose the whole run
                print(f"[{i}/{len(clips)}] {clip['file']}: FAILED {e}", flush=True)
                continue
            print(f"[{i}/{len(clips)}] {clip['file']}: score {s['score']:.2f} "
                  f"({s['frames']} frames, heads {s['mean_heads']:.2f}/f)", flush=True)
            scored.append(s)
    finally:
        dets.close()
    return sorted(scored, key=lambda c: -c["score"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default="@work/videos/raw")
    ap.add_argument("--weights", default="@work/models/helmet_yolov8s.pt")
    ap.add_argument("--world", default="@work/models/yolov8s-worldv2.pt")
    ap.add_argument("--no-world", action="store_true", help="skip YOLO-World (no forklift scoring)")
    ap.add_argument("--max-seconds", type=int, default=40, help="max frames analysed per clip (1/s)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--ov-conf", type=float, default=0.20)
    ap.add_argument("--fp32", action="store_true", help="disable FP16 on CUDA")
    ap.add_argument("--cams", type=int, default=10)
    ap.add_argument("--min-violation", type=int, default=3)
    ap.add_argument("--window", type=float, default=60.0, help="stream length (s) for start_s")
    ap.add_argument("--dry-run", action="store_true", help="only list the clips")
    ap.add_argument("--select-only", action="store_true", help="re-select from existing clip_scores.json")
    ap.add_argument("--skip-resource-check", action="store_true")
    a = ap.parse_args()

    raw = resolve(a.raw)
    vids = work_dir() / "videos"
    scores_path, sel_path = vids / "clip_scores.json", vids / "selection.json"
    clips = list_clips(raw, load_manifest(raw))
    if a.dry_run:
        dry_run(clips)
        return
    if not clips:
        sys.exit(f"no clips in {raw}")

    if a.select_only:
        scored = json.loads(scores_path.read_text(encoding="utf-8"))
    else:
        scored = score_all(clips, a)
        scores_path.write_text(json.dumps(scored, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", scores_path)
    print_scores(scored)

    picked = choose(scored, a.cams, a.min_violation)
    rows = selection_rows(assign_cameras(picked, load_config().get("cameras", []), a.cams), a.window)
    sel_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print_selection(rows)
    n_viol = sum(1 for c, _ in picked if is_violation(c))
    n_fork = sum(1 for c, _ in picked if c["forklift_frac"] >= FORKLIFT_MIN_FRAC)
    print(f"\n{len(rows)} cameras, {n_viol} with bare heads, {n_fork} with a forklift -> {sel_path}")


if __name__ == "__main__":
    main()
