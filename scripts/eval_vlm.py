"""How good is the VLM second opinion?  Crop labelled heads from the held-out test split
the same way the pipeline does for an alert, ask Qwen2-VL-2B the production prompt, and
compare with the label.

For a hard-hat alert the VLM's job is to say "yes, wearing one" on false alarms (so they
are filtered) and "no" on real violations (so they are confirmed).

    python scripts/eval_vlm.py --per-class 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import load_config, resolve, work_dir  # noqa: E402


def alert_crop(img, box):
    """Same expansion as Pipeline._emit (head box -> head + shoulders context)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - 1.0 * w)), int(max(0, y1 - 0.6 * h))
    cx2, cy2 = int(min(W, x2 + 1.0 * w)), int(min(H, y2 + 2.0 * h))
    return img[cy1:cy2, cx1:cx2]


def sample(split_dir: Path, per_class: int, min_px: int, seed: int) -> list[tuple[Path, int, tuple]]:
    """Up to per_class boxes of each class (0 helmet, 1 head), at least min_px tall, one per image."""
    rng = random.Random(seed)
    labels = sorted((split_dir.parent.parent / "labels" / split_dir.name).glob("*.txt"))
    rng.shuffle(labels)
    out, count = [], {0: 0, 1: 0}
    for lf in labels:
        img = split_dir / (lf.stem + ".jpg")
        if not img.exists():
            continue
        rows = [l.split() for l in lf.read_text().splitlines() if l.strip()]
        rng.shuffle(rows)
        for c, cx, cy, w, h in rows:
            c = int(c)
            if count[c] >= per_class:
                continue
            out.append((img, c, (float(cx), float(cy), float(w), float(h))))
            count[c] += 1
            break
        if all(v >= per_class for v in count.values()):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--min-px", type=int, default=24, help="skip heads smaller than this (pixels)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from sayguard.resources import require
    require("eval_vlm", gpu_gb=5.0, ram_gb=1.5)
    from sayguard.verify import Verifier, bgr_to_pil, parse_yes_no

    cfg = load_config()
    v = Verifier(str(resolve(cfg["vlm"]["model"])), store=None, enabled=True)
    v._load()                                  # synchronous load, no worker thread
    items = sample(work_dir() / "datasets" / "helmet" / "images" / "test", a.per_class, a.min_px, a.seed)
    tally = {"helmet": {"yes": 0, "no": 0, "unclear": 0}, "head": {"yes": 0, "no": 0, "unclear": 0}}
    ms = []
    for path, c, (cx, cy, w, h) in items:
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        box = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
        if box[3] - box[1] < a.min_px:
            continue
        t0 = time.perf_counter()
        ans = parse_yes_no(v._ask(bgr_to_pil(alert_crop(img, box))))
        ms.append((time.perf_counter() - t0) * 1000)
        key = "helmet" if c == 0 else "head"
        tally[key]["yes" if ans is True else "no" if ans is False else "unclear"] += 1
    hel, head = tally["helmet"], tally["head"]
    n_hel, n_head = sum(hel.values()), sum(head.values())
    res = {
        "model": cfg["vlm"]["model"], "split": "test (held out)", "crops": {"helmet": n_hel, "head": n_head},
        "tally": tally,
        # would be filtered as a false alarm if the detector had wrongly flagged this helmet as bare
        "false_alarm_filter_rate": round(hel["yes"] / max(1, n_hel), 3),
        # real violations the VLM keeps (confirms)
        "violation_confirm_rate": round(head["no"] / max(1, n_head), 3),
        "accuracy": round((hel["yes"] + head["no"]) / max(1, n_hel + n_head), 3),
        "latency_ms": {"mean": round(sum(ms) / max(1, len(ms))), "p95": round(sorted(ms)[int(0.95 * (len(ms) - 1))]) if ms else None},
        "device": v._device if hasattr(v, "_device") else None,
    }
    for d in (work_dir() / "results", ROOT / "docs" / "results"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "vlm_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
