"""Evaluate the fine-tuned hard-hat detector on the held-out test split.

Training only ever looked at train/val, so the test split is the honest number
for the write-up.  The output JSON is also the baseline to compare against once
Stage II re-compiles the same YOLOv8s to a Hailo-10H HEF (INT8): the accuracy
drop from quantisation is measured against exactly these figures.

Dataloader workers are kept at 2: on Windows each spawned worker commits
~1.2 GB, and this machine is close to its commit limit.

    python scripts/eval_detector.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import resolve, work_dir  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET = "keremberke/hard-hat-detection (Roboflow Hard Hats v2, CC BY 4.0), classes helmet/head"


def split_counts(data_yaml: Path, split: str) -> dict:
    """Images and label instances on disk, independent of what the validator reports."""
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    img_dir = Path(d.get("path", data_yaml.parent)) / d[split]
    parts = list(img_dir.parts)
    if "images" in parts:   # same rule as Ultralytics: the last "images" path component -> "labels"
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    lbl_dir = Path(*parts)
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT] if img_dir.is_dir() else []
    names = d["names"] if isinstance(d["names"], dict) else dict(enumerate(d["names"]))
    inst = {str(n): 0 for n in names.values()}
    for p in images:
        lbl = lbl_dir / (p.stem + ".txt")
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                inst[str(names[int(line.split()[0])])] += 1
    return {"images": len(images), "instances": inst, "image_dir": str(img_dir)}


def model_stats(model, weights: Path, imgsz: int) -> dict:
    out = {"size_mb": round(weights.stat().st_size / 1e6, 2),
           "params": int(sum(p.numel() for p in model.model.parameters()))}
    out["params_m"] = round(out["params"] / 1e6, 2)
    try:
        from ultralytics.utils.torch_utils import get_flops
        out["gflops"] = round(float(get_flops(model.model, imgsz)), 1) or None
    except Exception:
        out["gflops"] = None
    return out


def to_float(x) -> float:
    return round(float(x), 4)


def collect(metrics, names: dict) -> dict:
    box = metrics.box
    per_class = {}
    for i, c in enumerate(metrics.ap_class_index):
        c = int(c)
        per_class[str(names[c])] = {
            "precision": to_float(box.p[i]), "recall": to_float(box.r[i]),
            "mAP50": to_float(box.ap50[i]), "mAP50-95": to_float(box.ap[i]),
            "images": int(metrics.nt_per_image[c]), "instances": int(metrics.nt_per_class[c]),
        }
    cm = getattr(metrics, "confusion_matrix", None)
    labels = [str(names[k]) for k in sorted(names)] + ["background"]
    return {
        "mAP50": to_float(box.map50), "mAP50-95": to_float(box.map), "mAP75": to_float(box.map75),
        "precision": to_float(box.mp), "recall": to_float(box.mr),
        "per_class": per_class,
        "images_evaluated": len(getattr(box, "image_metrics", None) or {}) or None,
        "instances": int(metrics.nt_per_class.sum()),
        "speed_ms_per_image": {k: round(float(v), 3) for k, v in metrics.speed.items()},
        "confusion_matrix": None if cm is None else {
            "labels": labels, "rows": "predicted", "cols": "true",
            "matrix": [[int(v) for v in row] for row in cm.matrix]},
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="@work/models/helmet_yolov8s.pt")
    ap.add_argument("--data", default="@work/datasets/helmet/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fp32", action="store_true", help="disable FP16 on CUDA")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--skip-resource-check", action="store_true")
    a = ap.parse_args()

    weights, data = resolve(a.weights), resolve(a.data)
    counts = split_counts(data, a.split)
    if not a.skip_resource_check:
        from sayguard.resources import require
        require("eval_detector (YOLOv8s val)", gpu_gb=1.5, ram_gb=2.0 + 1.2 * a.workers)

    import torch
    import ultralytics
    from ultralytics import YOLO

    cuda = torch.cuda.is_available()
    quantize = 16 if cuda and not a.fp32 else None
    model = YOLO(str(weights))
    stats = model_stats(model, weights, a.imgsz)
    t0 = time.time()
    metrics = model.val(data=str(data), split=a.split, batch=a.batch, workers=a.workers, imgsz=a.imgsz,
                        device=0 if cuda else "cpu", quantize=quantize, plots=not a.no_plots,
                        project=str(work_dir() / "runs"), name=f"helmet_eval_{a.split}", exist_ok=True)
    wall = time.time() - t0

    names = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {"weights": str(weights), "architecture": "YOLOv8s (fine-tuned, 2 classes)", **stats},
        "dataset": {"source": DATASET, "data_yaml": str(data), "split": a.split, **counts},
        "settings": {"imgsz": a.imgsz, "batch": a.batch, "workers": a.workers,
                     "precision": "fp16" if quantize == 16 else "fp32", "conf": 0.001, "iou": 0.7,
                     "note": "Ultralytics val defaults; P/R are at the max-F1 confidence"},
        "hardware": {"device": torch.cuda.get_device_name(0) if cuda else "cpu",
                     "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "metrics": collect(metrics, names),
        "wall_time_s": round(wall, 1),
    }
    if cuda:
        torch.cuda.empty_cache()

    text = json.dumps(report, ensure_ascii=False, indent=1)
    for p in (work_dir() / "results" / "detector_metrics.json", ROOT / "docs" / "results" / "detector_metrics.json"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print("wrote", p)
    m = report["metrics"]
    print(f"\nmAP50 {m['mAP50']:.3f}  mAP50-95 {m['mAP50-95']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}")
    for name, c in m["per_class"].items():
        print(f"  {name:<8} P {c['precision']:.3f}  R {c['recall']:.3f}  mAP50 {c['mAP50']:.3f}  "
              f"mAP50-95 {c['mAP50-95']:.3f}  ({c['instances']} instances, {c['images']} images)")
    print(f"  speed (ms/img): {m['speed_ms_per_image']}  model {stats['size_mb']} MB, {stats['params_m']} M params")


if __name__ == "__main__":
    main()
