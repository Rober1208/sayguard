"""Measure site mode on a warehouse scene the detector never trained on (Warehouse_010, or
Warehouse_018 with --config configs/eval/warehouse_018.yaml).

Runs the exact live fusion (sayguard.site_pipeline.SiteFusion + RuleEngine.evaluate_site) frame
by frame over the synchronized streams and scores it against NVIDIA's ground truth, which knows
every person's floor position and the same id in every camera:

  identity   MOTA / IDF1 / id switches on the floor plan (match radius 1 m), and how many
             global ids each real person was split into
  counting   unique workers vs. the sum of per-camera detections vs. the true head count
  alerts     ground-truth events (a person within the rule's distance of a forklift for the
             rule's duration; a person in the restricted aisle for its duration; with --all-zones
             also a person in each other floor zone for 3 s) against the
             alerts raised, in site mode (one subject per worker) and in per-camera mode (each
             camera alerts on its own tracks with the per-camera duplicate guards SayGuard used
             before: same-place IoU, 15 s camera gap, one presence subject per camera); both
             modes get the same detections and floor distances
  distance   worker-to-forklift distance as measured vs. the ground-truth gap to the forklift's
             3D footprint

    python scripts/eval_site.py [--config configs/warehouse.yaml] [--limit 600] [--dets-cache dets.pkl]

--dets-cache stores the detector output on the first run and replays it afterwards, so tracker and
rule changes can be measured without the GPU (the cache is tied to the weights it was made with).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import load_config, resolve, work_dir  # noqa: E402
from sayguard.rules import CameraInfo, Rule, RuleEngine  # noqa: E402
from sayguard.rules.engine import SiteObject, SiteSubject  # noqa: E402
from sayguard.site_pipeline import SiteFusion, load_site  # noqa: E402
from sayguard.site import touches_bottom  # noqa: E402
from sayguard.sitetrack import cluster_vehicles  # noqa: E402

MATCH_M = 1.0          # a worker counts as a true person within this floor distance
ALERT_MATCH_M = 2.5    # an alert belongs to an event if its worker is this close to the event's person
EVENT_GAP_S = 1.0      # gaps shorter than this do not split a ground-truth event
IN_VIEW_PX = 40.0      # "in view": at least this tall (px, 1280-wide stream) in some camera.  Chosen on the
                       # validation scene (Warehouse_020), where person detection went from 68 % of people
                       # 20-40 px tall to 92 % at 40-80 px; smaller people are mostly not detected at all


def load_gt(path: Path) -> dict[int, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(f): [{k.replace(" ", "_"): v for k, v in o.items()} for o in objs] for f, objs in raw.items()}


def footprint_gap(p: np.ndarray, o: dict) -> float:
    """Floor distance from point p to an object's oriented 3D box footprint (0 inside)."""
    c = np.asarray(o["3d_location"][:2], float)
    w, l = o["3d_bounding_box_scale"][:2]
    yaw = o["3d_bounding_box_rotation"][2]
    d = p - c
    dx = math.cos(-yaw) * d[0] - math.sin(-yaw) * d[1]
    dy = math.sin(-yaw) * d[0] + math.cos(-yaw) * d[1]
    return math.hypot(max(abs(dx) - w / 2, 0.0), max(abs(dy) - l / 2, 0.0))


def segments(frames: list[float], min_s: float) -> list[tuple[float, float]]:
    """Sorted timestamps -> [(start, end)] runs lasting >= min_s (gaps < EVENT_GAP_S joined)."""
    out = []
    if not frames:
        return out
    start = prev = frames[0]
    for t in frames[1:] + [None]:
        if t is None or t - prev > EVENT_GAP_S:
            if prev - start >= min_s:
                out.append((start, prev))
            if t is not None:
                start = t
        if t is not None:
            prev = t
    return out


def best_event(a: dict, events: list[dict]) -> int | None:
    """The event of the alert's rule, around its time, whose person was closest to it (<= ALERT_MATCH_M)."""
    best = None
    for i, e in enumerate(events):
        if e["rule"] != a["rule"] or not (e["start"] - 0.5 <= a["ts"] <= e["end"] + 3.0):
            continue
        pos = e["pos"].get(round(a["ts"], 1)) or e["pos"][min(e["pos"], key=lambda t: abs(t - a["ts"]))]
        d = float(np.hypot(*(np.asarray(pos) - np.asarray(a["xy"])))) if a["xy"] is not None else 0.0
        if d <= ALERT_MATCH_M and (best is None or d < best[0]):
            best = (d, i)
    return None if best is None else best[1]


def match_alerts(events: list[dict], alerts: list[dict], cooldown_s: float = 0.0,
                 gt_pos: dict | None = None) -> dict:
    """Each alert -> the event of the same rule whose person was closest at alert time.

    recall counts events with an alert of their own.  An event whose person was alerted for this
    rule less than cooldown_s before it started gets no new alert by design (the rule's cooldown):
    recall_with_cooldown also counts those (the earlier alert's position must match that person
    at that time)."""
    hit = defaultdict(int)
    false = 0
    for a in alerts:
        i = best_event(a, events)
        if i is None:
            false += 1
        else:
            hit[i] += 1
    n_ev = len(events)
    found = len(hit)
    covered = set(hit)
    if cooldown_s > 0 and gt_pos is not None:
        for i, e in enumerate(events):
            if i in covered:
                continue
            for a in alerts:
                if not (e["start"] - cooldown_s <= a["ts"] < e["start"]) or a["xy"] is None:
                    continue
                track = gt_pos.get(e["pid"], {})
                pos = track.get(round(a["ts"], 1))
                if pos is not None and float(np.hypot(*(np.asarray(pos) - np.asarray(a["xy"])))) <= ALERT_MATCH_M:
                    covered.add(i)
                    break
    return {"events": n_ev, "events_alerted": found, "recall": round(found / n_ev, 3) if n_ev else None,
            "events_in_cooldown": len(covered) - found,
            "recall_with_cooldown": round(len(covered) / n_ev, 3) if n_ev else None,
            "alerts": len(alerts), "alerts_on_events": sum(hit.values()), "false_alerts": false,
            "alerts_per_alerted_event": round(sum(hit.values()) / found, 2) if found else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "warehouse.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N stream frames")
    ap.add_argument("--weights", default=None, help="override detector.weights")
    ap.add_argument("--dets-cache", default=None, help="save detections here / replay them if present")
    ap.add_argument("--tag", default=None, help="experiment: write results/site_eval_<tag>.json only")
    ap.add_argument("--all-zones", action="store_true",
                    help="also score a presence rule (3 s, 60 s cooldown) for every floor zone, for more events")
    a = ap.parse_args()

    import motmetrics as mm

    from sayguard.detectors import make_detector

    cfg = load_config(a.config)
    if a.weights:
        cfg["detector"]["weights"] = a.weights
    site, calib_of = load_site(cfg)
    sc = cfg["site"]
    cams = [c["id"] for c in cfg["cameras"]]
    calib_ids = {c.get("calib", c["id"]) for c in cfg["cameras"]}
    src = {c["id"]: resolve(c["source"]) for c in cfg["cameras"]}
    meta = json.loads((Path(next(iter(src.values()))).parent / "streams.json").read_text(encoding="utf-8"))
    step, fps = int(meta["step"]), float(meta["fps"])
    gt = load_gt(Path(resolve(sc["calibration"])).parent / "ground_truth.json")
    fusion = SiteFusion(site, calib_of, fps, sc.get("zone"))

    rules = [Rule(**r) for r in cfg.get("default_rules", [])]
    if a.all_zones:
        covered = {z for r in rules if r.type == "person_present" for z in r.zones}
        rules += [Rule(name=f"Zone check: {z.id}", type="person_present", zones=[z.id], min_duration_s=3.0,
                       cooldown_s=60.0) for z in site.zones if z.id not in covered]
    infos = [CameraInfo(c["id"], c["name"], c["zone"]) for c in cfg["cameras"]]
    eng_site, eng_cam = RuleEngine(infos), RuleEngine(infos)
    for r in rules:
        eng_site.add(r)
        eng_cam.add(Rule(**r.model_dump()))
    prox = next((r for r in rules if r.type == "object_near_person"), None)
    zone_rules = [r for r in rules if r.type == "person_present"]

    n_frames = min(int(meta["frames"]), a.limit or 10 ** 9)
    cache_path = Path(a.dets_cache) if a.dets_cache else None
    wfile = Path(resolve(cfg["detector"]["weights"]))
    made_with = [cfg["detector"]["weights"], *((wfile.stat().st_size, wfile.stat().st_mtime_ns) if wfile.exists() else ())]
    replay = None
    if cache_path is not None and cache_path.exists():
        replay = pickle.loads(cache_path.read_bytes())
        if replay["weights"] != made_with or replay["cams"] != cams:
            raise SystemExit(f"{cache_path} was made with {replay['weights']} / {replay['cams']}")
        n_frames = min(n_frames, len(replay["frames"]))
        detector, caps = None, {}
    else:
        detector = make_detector(cfg)
        caps = {c: cv2.VideoCapture(str(src[c])) for c in cams}
    recorded = []
    acc = mm.MOTAccumulator(auto_id=True)
    counts = {"gt": [], "unique": [], "camera_sum": []}
    alerts = {"site": [], "camera": []}
    gt_prox: dict[int, list[float]] = defaultdict(list)
    gt_zone: dict[str, dict[int, list[float]]] = {r.name: defaultdict(list) for r in zone_rules}
    # the same, counting a person only while in view (IN_VIEW_PX)
    gt_prox_v: dict[int, list[float]] = defaultdict(list)
    gt_zone_v: dict[str, dict[int, list[float]]] = {r.name: defaultdict(list) for r in zone_rules}
    px_scale = float(meta.get("width", 1280)) / 1920.0          # ground-truth boxes are 1920-wide source pixels
    gt_pos: dict[int, dict[float, tuple]] = defaultdict(dict)
    dist_err, dist_pairs = [], []
    near_missed = 0                                    # person near a forklift that was not detected
    fl_seen = fl_found = 0                             # forklift-frames visible in a camera / detected
    hyp_of_gt: dict[int, set] = defaultdict(set)
    near_ids: dict[int, set] = defaultdict(set)        # every worker id ever nearest to a person
    dup_frames = person_frames = 0
    t0 = time.time()
    for k in range(n_frames):
        if replay is not None:
            dets, sizes = replay["frames"][k]
        else:
            frames = {}
            for c in cams:
                ok, f = caps[c].read()
                if not ok:
                    break
                frames[c] = f
            if len(frames) < len(cams):
                break
            dets = detector.infer([frames[c] for c in cams])
            sizes = {c: (frames[c].shape[1], frames[c].shape[0]) for c in cams}
            recorded.append((dets, sizes))
        ts = round(k / fps, 3)
        inputs = [(c, sizes[c], d, []) for c, d in zip(cams, dets)]
        fr = fusion.step(ts, inputs)

        # site mode: one subject per worker
        res = eng_site.evaluate_site(ts, fr.subjects, fr.objects, frame_dt=1 / fps)
        pos_of = {s.gid: s.xy for s in fr.subjects}
        for f in res.fired:
            alerts["site"].append({"rule": f.rule.name, "ts": ts, "xy": pos_of.get(f.track_id)})
        # per-camera mode: each camera alerts on its own tracks and the vehicles it sees, with the
        # per-camera duplicate guards (evaluate_site with cam_id = that camera)
        for c in cams:
            W, H = sizes[c]
            cs = []
            for b in fr.tracked.get(c, []):
                o = fusion.person_obs(c, b, W, H)
                cs.append(SiteSubject(b.track_id, tuple(o.xy), fusion.zones(o.xy), [c], box=tuple(b.xyxy)))
            pts = []
            for ob in fr.objs.get(c, []):
                if touches_bottom(ob.xyxy, H):            # as SiteFusion: no ground contact in view
                    continue
                x1, _, x2, y2 = ob.xyxy
                pts += [(ob.label, c, calib_of[c].to_world(u, y2, src_wh=(W, H)), ob.xyxy) for u in (x1, (x1 + x2) / 2, x2)]
            co = [SiteObject(v.label, tuple(v.xy), fusion.zones(v.xy), [c], distance_to=v.distance_to)
                  for v in cluster_vehicles(pts)]
            r2 = eng_cam.evaluate_site(ts, cs, co, frame_dt=1 / fps, cam_id=c)
            cpos = {s.gid: s.xy for s in cs}
            for f in r2.fired:                       # a whole-camera ("cam") alert has no position
                alerts["camera"].append({"rule": f.rule.name, "ts": ts, "xy": cpos.get(f.track_id)})

        # ground truth at the matching source frame
        objs_gt = gt.get(k * step, [])
        # people no configured camera sees cannot be detected: they are not counted as misses
        persons = [(o["object_id"], np.asarray(o["3d_location"][:2], float)) for o in objs_gt
                   if o["object_type"] == "Person" and calib_ids & set(o.get("2d_bounding_box_visible", {}))]
        in_view = {o["object_id"] for o in objs_gt if o["object_type"] == "Person" and any(
            (b[3] - b[1]) * px_scale >= IN_VIEW_PX for c, b in o.get("2d_bounding_box_visible", {}).items()
            if c in calib_ids)}
        forklifts = [o for o in objs_gt if o["object_type"] == "Forklift"]
        det_fl = [v for v in fr.vehicles if v.label == "forklift"]

        def found(o):
            """A detected forklift whose ground contacts touch this forklift's footprint (2.5 m slack)."""
            return [v for v in det_fl if min(footprint_gap(np.asarray(q), o) for _, q, _ in v.points) <= 2.5]
        for o in forklifts:
            if calib_ids & set(o.get("2d_bounding_box_visible", {})):
                fl_seen += 1
                fl_found += bool(found(o))
        hyp = [(s.gid, np.asarray(s.xy)) for s in fr.subjects]
        dmat = np.full((len(persons), len(hyp)), np.nan)
        for i, (_, p) in enumerate(persons):
            for j, (_, h) in enumerate(hyp):
                d = float(np.hypot(*(p - h)))
                if d <= MATCH_M:
                    dmat[i, j] = d          # metres, so MOTP is the mean position error in metres
        acc.update([pid for pid, _ in persons], [g for g, _ in hyp], dmat)
        # duplicates: every worker goes to its nearest person (many-to-one), so two ids on one
        # person at the same time are counted, which one-to-one MOT matching hides
        per_person: dict[int, set] = defaultdict(set)
        for g, h in hyp:
            if persons:
                i = int(np.argmin([np.hypot(*(p - h)) for _, p in persons]))
                if np.hypot(*(persons[i][1] - h)) <= MATCH_M:
                    per_person[persons[i][0]].add(g)
        for pid, ids in per_person.items():
            near_ids[pid] |= ids
        person_frames += len(persons)
        dup_frames += sum(1 for ids in per_person.values() if len(ids) > 1)
        counts["gt"].append(len(persons))
        counts["unique"].append(len(fr.workers))
        counts["camera_sum"].append(fr.camera_sum)
        for pid, p in persons:
            gt_pos[pid][round(ts, 1)] = tuple(p)
            if forklifts and prox is not None:
                gap = min(footprint_gap(p, o) for o in forklifts)
                if gap <= prox.threshold:
                    gt_prox[pid].append(ts)
                    if pid in in_view:
                        gt_prox_v[pid].append(ts)
                if gap <= 6.0 and hyp:
                    j = int(np.argmin([np.hypot(*(p - h)) for _, h in hyp]))
                    if np.hypot(*(p - hyp[j][1])) <= MATCH_M:
                        # the measured distance must be to the same forklift; if that one was not
                        # detected the rule cannot see it: counted separately, not as a distance error
                        f_near = min(forklifts, key=lambda o: footprint_gap(p, o))
                        same = found(f_near)
                        if same:
                            m = min(v.distance_to(hyp[j][1]) for v in same)
                            dist_err.append(m - gap)
                            dist_pairs.append((round(gap, 2), round(m, 2)))
                        else:
                            near_missed += 1
            inside = site.zones_of(*p)
            for zr in zone_rules:
                if any(z in zr.zones for z in inside):
                    gt_zone[zr.name][pid].append(ts)
                    if pid in in_view:
                        gt_zone_v[zr.name][pid].append(ts)
        if k % 300 == 0:
            print(f"frame {k}/{n_frames}  workers {len(fr.workers)} (gt {len(persons)})  "
                  f"{(time.time() - t0) / max(1, k):.2f} s/frame", flush=True)
    for cap in caps.values():
        cap.release()
    if cache_path is not None and replay is None:
        cache_path.write_bytes(pickle.dumps({"weights": made_with, "cams": cams,
                                             "frames": recorded}))

    # identity
    mh = mm.metrics.create()
    summ = mh.compute(acc, metrics=["mota", "motp", "idf1", "idp", "idr", "num_switches", "num_fragmentations",
                                    "precision", "recall", "num_unique_objects", "mostly_tracked"], name="site")
    ev = acc.mot_events
    for _, row in ev[ev["Type"].isin(["MATCH", "SWITCH"])].iterrows():
        hyp_of_gt[row["OId"]].add(row["HId"])
    split = [len(v) for v in hyp_of_gt.values()]
    ident = {m: (round(float(summ[m].iloc[0]), 3) if isinstance(summ[m].iloc[0], float) else int(summ[m].iloc[0]))
             for m in summ.columns}
    ident["sequential_ids_per_person"] = {"mean": round(float(np.mean(split)), 2) if split else None,
                                          "median": float(np.median(split)) if split else None}
    near = [len(v) for v in near_ids.values()]
    ident["ids_ever_on_a_person"] = {"mean": round(float(np.mean(near)), 2) if near else None,
                                     "median": float(np.median(near)) if near else None}
    ident["duplicate_person_frames_pct"] = round(100 * dup_frames / max(1, person_frames), 2)

    # events
    def build(prox_ts: dict, zone_ts: dict) -> list[dict]:
        out = []
        per_rule = ([(prox, prox_ts)] if prox is not None else []) + [(zr, zone_ts[zr.name]) for zr in zone_rules]
        for rule, by_pid in per_rule:
            for pid, ts_list in by_pid.items():
                for s, e in segments(sorted(ts_list), rule.min_duration_s):
                    out.append({"rule": rule.name, "pid": pid, "start": s, "end": e,
                                "pos": {t: p for t, p in gt_pos[pid].items() if s - 0.5 <= t <= e + 3.0}})
        return out
    events, events_v = build(gt_prox, gt_zone), build(gt_prox_v, gt_zone_v)
    alert_eval, alert_eval_v = {}, {}
    for mode in ("site", "camera"):
        alert_eval[mode], alert_eval_v[mode] = {}, {}
        for r in [prox, *zone_rules]:
            if r is None:
                continue
            ev_r, ev_v = [e for e in events if e["rule"] == r.name], [e for e in events_v if e["rule"] == r.name]
            al = [x for x in alerts[mode] if x["rule"] == r.name]
            alert_eval[mode][r.name] = match_alerts(ev_r, al, r.cooldown_s, gt_pos)
            # in view: events of people out of view are dropped, and so are the alerts on them
            kept = [x for x in al if best_event(x, ev_v) is not None or best_event(x, ev_r) is None]
            alert_eval_v[mode][r.name] = match_alerts(ev_v, kept, r.cooldown_s, gt_pos)

    g, u, cs = (np.asarray(counts[x], float) for x in ("gt", "unique", "camera_sum"))
    out = {
        "scene": meta["scene"], "frames": len(g), "fps": fps, "cameras": len(cams),
        "detector": cfg["detector"]["weights"],
        "identity": ident,
        "counting": {"true_mean": round(float(g.mean()), 2), "unique_workers_mean": round(float(u.mean()), 2),
                     "camera_sum_mean": round(float(cs.mean()), 2),
                     "unique_abs_error_mean": round(float(np.abs(u - g).mean()), 2),
                     "camera_sum_abs_error_mean": round(float(np.abs(cs - g).mean()), 2)},
        "alerts": alert_eval,
        "alerts_in_view": alert_eval_v,
        "in_view_px": IN_VIEW_PX,
        "forklift_detection": {"forklift_frames_visible": fl_seen, "detected": fl_found,
                               "rate": round(fl_found / fl_seen, 3) if fl_seen else None},
        "forklift_distance": {"pairs": len(dist_err), "pairs_forklift_not_detected": near_missed,
                              "mae_m": round(float(np.mean(np.abs(dist_err))), 2) if dist_err else None,
                              "median_abs_error_m": round(float(np.median(np.abs(dist_err))), 2) if dist_err else None,
                              "mean_signed_error_m": round(float(np.mean(dist_err)), 2) if dist_err else None},
        "note": "Ground truth never enters the pipeline. Per-camera mode = same detections and floor distances, "
                "but each camera track alerts on its own (no cross-camera identity).",
    }
    stem = "site_eval" if Path(a.config).stem == "warehouse" else f"site_eval_{Path(a.config).stem}"
    name = f"{stem}_{a.tag}.json" if a.tag else f"{stem}.json"
    for d in (work_dir() / "results",) if a.tag else (work_dir() / "results", ROOT / "docs" / "results"):
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    if a.tag:                                    # experiments: keep every alert for a closer look
        (work_dir() / "results" / f"{Path(name).stem}_alerts.json").write_text(
            json.dumps({"alerts": alerts, "events": [{k: v for k, v in e.items() if k != "pos"} for e in events]},
                       default=lambda o: [float(x) for x in o]), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
